package com.edgesense.respiratory

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class InferenceResult(
    val className: String,
    val confidence: Float,
    val allProbabilities: FloatArray
)

class ModelInference(context: Context, modelFile: String) {
    
    private var interpreter: Interpreter
    
    private val labels = arrayOf(
        "Normal",
        "Asthma",
        "COPD",
        "Pneumonia",
        "Bronchitis",
        "Tuberculosis",
        "Long-COVID"
    )
    
    init {
        val model = loadModelFile(context, modelFile)
        val options = Interpreter.Options()
        options.setNumThreads(4)
        interpreter = Interpreter(model, options)
    }
    
    private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun predict(features: Array<FloatArray>): InferenceResult {
        // Prepare input
        val inputShape = interpreter.getInputTensor(0).shape()
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputShape[1] * inputShape[2] * inputShape[3])
        inputBuffer.order(ByteOrder.nativeOrder())
        
        // Fill input buffer
        for (i in features.indices) {
            for (j in features[i].indices) {
                inputBuffer.putFloat(features[i][j])
            }
        }
        
        // Prepare output
        val outputShape = interpreter.getOutputTensor(0).shape()
        val output = Array(1) { FloatArray(outputShape[1]) }
        
        // Run inference
        val startTime = System.currentTimeMillis()
        interpreter.run(inputBuffer, output)
        val inferenceTime = System.currentTimeMillis() - startTime
        
        // Get results
        val probabilities = output[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val maxConfidence = probabilities[maxIndex]
        
        return InferenceResult(
            className = labels[maxIndex],
            confidence = maxConfidence,
            allProbabilities = probabilities
        )
    }
    
    fun close() {
        interpreter.close()
    }
}
