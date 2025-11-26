"""
Real-time respiratory disease detection on Raspberry Pi.
"""

import numpy as np
import pyaudio
import tensorflow as tf
import librosa
from collections import deque
import time


class RealTimeDetector:
    """Real-time audio detection on Raspberry Pi."""
    
    def __init__(self, model_path='models/quantized_model.tflite'):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_size)
        
        # Labels
        self.labels = [
            'Normal',
            'Asthma',
            'COPD',
            'Pneumonia',
            'Bronchitis',
            'Tuberculosis',
            'Long-COVID'
        ]
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
    
    def extract_features(self, audio):
        """Extract MFCC features."""
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=40,
            n_fft=2048,
            hop_length=160
        )
        
        # Extract Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=160
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Combine features
        min_time = min(mfcc.shape[1], mel_spec_db.shape[1])
        mfcc = mfcc[:, :min_time]
        mel_spec_db = mel_spec_db[:, :min_time]
        
        features = np.vstack([mfcc, mel_spec_db])
        features = features.T
        features = np.expand_dims(features, axis=-1)
        features = np.expand_dims(features, axis=0)
        
        return features.astype(np.float32)
    
    def predict(self, audio):
        """Run inference."""
        features = self.extract_features(audio)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Get prediction
        pred_idx = np.argmax(output)
        confidence = output[pred_idx]
        
        return self.labels[pred_idx], confidence, output
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback."""
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def start_stream(self):
        """Start audio stream."""
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
    
    def stop_stream(self):
        """Stop audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
    
    def run(self):
        """Run real-time detection."""
        print("=" * 60)
        print("EdgeSense Real-Time Respiratory Detection")
        print("=" * 60)
        print("\nStarting audio stream...")
        print("Speak, breathe, or cough near the microphone...")
        print("Press Ctrl+C to stop\n")
        
        self.start_stream()
        
        try:
            while True:
                if len(self.audio_buffer) >= self.chunk_size:
                    # Get audio chunk
                    audio = np.array(list(self.audio_buffer))
                    
                    # Predict
                    start_time = time.time()
                    prediction, confidence, probabilities = self.predict(audio)
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Display results
                    print(f"\r[{time.strftime('%H:%M:%S')}] "
                          f"Prediction: {prediction:15s} | "
                          f"Confidence: {confidence:.3f} | "
                          f"Latency: {inference_time:.1f}ms", end='')
                    
                    # Alert for high-risk conditions
                    if prediction != 'Normal' and confidence > 0.7:
                        print(f"\n⚠️  ALERT: {prediction} detected with {confidence:.1%} confidence")
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop_stream()
            print("Stopped.")


if __name__ == '__main__':
    detector = RealTimeDetector()
    detector.run()
