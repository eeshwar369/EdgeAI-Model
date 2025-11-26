package com.edgesense.respiratory

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.ProgressBar
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {
    
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var modelInference: ModelInference
    
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var tvPrediction: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvStatus: TextView
    private lateinit var progressBar: ProgressBar
    
    private var isRecording = false
    private val scope = CoroutineScope(Dispatchers.Main + Job())
    
    companion object {
        private const val REQUEST_RECORD_AUDIO = 1
        private const val MODEL_FILE = "quantized_model.tflite"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initViews()
        checkPermissions()
        initModel()
    }
    
    private fun initViews() {
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        tvPrediction = findViewById(R.id.tvPrediction)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvStatus = findViewById(R.id.tvStatus)
        progressBar = findViewById(R.id.progressBar)
        
        btnStart.setOnClickListener { startDetection() }
        btnStop.setOnClickListener { stopDetection() }
        btnStop.isEnabled = false
    }
    
    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_RECORD_AUDIO
            )
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            REQUEST_RECORD_AUDIO -> {
                if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    tvStatus.text = "Microphone permission required"
                }
            }
        }
    }
    
    private fun initModel() {
        try {
            modelInference = ModelInference(this, MODEL_FILE)
            audioRecorder = AudioRecorder()
            tvStatus.text = "Ready"
        } catch (e: Exception) {
            tvStatus.text = "Error loading model: ${e.message}"
        }
    }
    
    private fun startDetection() {
        if (!isRecording) {
            isRecording = true
            btnStart.isEnabled = false
            btnStop.isEnabled = true
            tvStatus.text = "Recording..."
            
            scope.launch {
                withContext(Dispatchers.IO) {
                    audioRecorder.startRecording()
                    
                    while (isRecording) {
                        val audioData = audioRecorder.getAudioBuffer()
                        
                        if (audioData != null) {
                            // Extract features
                            val features = FeatureExtractor.extractFeatures(audioData)
                            
                            // Run inference
                            val result = modelInference.predict(features)
                            
                            // Update UI
                            withContext(Dispatchers.Main) {
                                updatePrediction(result)
                            }
                        }
                        
                        delay(500) // Update every 500ms
                    }
                }
            }
        }
    }
    
    private fun stopDetection() {
        if (isRecording) {
            isRecording = false
            audioRecorder.stopRecording()
            btnStart.isEnabled = true
            btnStop.isEnabled = false
            tvStatus.text = "Stopped"
        }
    }
    
    private fun updatePrediction(result: InferenceResult) {
        tvPrediction.text = "Prediction: ${result.className}"
        tvConfidence.text = "Confidence: ${String.format("%.2f%%", result.confidence * 100)}"
        
        // Update progress bar
        progressBar.progress = (result.confidence * 100).toInt()
        
        // Color code based on risk
        val color = when {
            result.className == "Normal" -> ContextCompat.getColor(this, android.R.color.holo_green_dark)
            result.confidence > 0.8 -> ContextCompat.getColor(this, android.R.color.holo_red_dark)
            result.confidence > 0.6 -> ContextCompat.getColor(this, android.R.color.holo_orange_dark)
            else -> ContextCompat.getColor(this, android.R.color.holo_blue_dark)
        }
        tvPrediction.setTextColor(color)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) {
            stopDetection()
        }
        scope.cancel()
        modelInference.close()
    }
}
