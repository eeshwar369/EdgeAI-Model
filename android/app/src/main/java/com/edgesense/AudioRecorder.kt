package com.edgesense.respiratory

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.math.abs

class AudioRecorder {
    
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
    
    private val audioBuffer = mutableListOf<Short>()
    private val targetBufferSize = sampleRate * 3 // 3 seconds
    
    fun startRecording() {
        if (isRecording) return
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )
        
        audioRecord?.startRecording()
        isRecording = true
        
        Thread {
            val buffer = ShortArray(bufferSize)
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, bufferSize) ?: 0
                if (read > 0) {
                    synchronized(audioBuffer) {
                        audioBuffer.addAll(buffer.take(read))
                        
                        // Keep only last 3 seconds
                        if (audioBuffer.size > targetBufferSize) {
                            audioBuffer.subList(0, audioBuffer.size - targetBufferSize).clear()
                        }
                    }
                }
            }
        }.start()
    }
    
    fun stopRecording() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }
    
    fun getAudioBuffer(): FloatArray? {
        synchronized(audioBuffer) {
            if (audioBuffer.size < targetBufferSize) {
                return null
            }
            
            // Convert to float and normalize
            val floatBuffer = FloatArray(targetBufferSize)
            var maxAbs = 0f
            
            for (i in 0 until targetBufferSize) {
                floatBuffer[i] = audioBuffer[i].toFloat()
                maxAbs = maxOf(maxAbs, abs(floatBuffer[i]))
            }
            
            // Normalize
            if (maxAbs > 0) {
                for (i in floatBuffer.indices) {
                    floatBuffer[i] /= maxAbs
                }
            }
            
            return floatBuffer
        }
    }
}
