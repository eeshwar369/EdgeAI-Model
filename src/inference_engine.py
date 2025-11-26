"""
Real-time inference engine for respiratory disease detection.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from .feature_extractor import RespiratoryFeatureExtractor, preprocess_audio
from .anomaly_detector import RespiratoryAnomalyDetector


class RespiratoryInferenceEngine:
    """Real-time inference for respiratory disease detection."""
    
    def __init__(
        self,
        model_path: str,
        anomaly_detector_path: str = None,
        use_tflite: bool = False
    ):
        self.use_tflite = use_tflite
        self.feature_extractor = RespiratoryFeatureExtractor()
        
        # Load classification model
        if use_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.interpreter = None
        
        # Load anomaly detector
        self.anomaly_detector = None
        if anomaly_detector_path:
            self.anomaly_detector = RespiratoryAnomalyDetector()
            self.anomaly_detector.load(anomaly_detector_path)
        
        self.label_names = [
            'Normal',
            'Asthma',
            'COPD',
            'Pneumonia',
            'Bronchitis',
            'Tuberculosis',
            'Long-COVID'
        ]
    
    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_features: bool = False
    ) -> Dict:
        """
        Predict respiratory condition from audio.
        
        Returns:
            dict with keys:
                - prediction: class name
                - confidence: probability
                - probabilities: all class probabilities
                - is_anomaly: whether pattern is anomalous
                - anomaly_score: anomaly score
        """
        
        # Preprocess audio
        audio_processed = preprocess_audio(audio, sample_rate)
        
        # Extract features
        features = self.feature_extractor.prepare_model_input(audio_processed)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        
        # Classification
        if self.use_tflite:
            # TFLite inference
            self.interpreter.set_tensor(self.input_details[0]['index'], features.astype(np.float32))
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            # Keras inference
            probabilities = self.model.predict(features, verbose=0)[0]
        
        # Get prediction
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        prediction = self.label_names[predicted_class]
        
        result = {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.label_names, probabilities)
            }
        }
        
        # Anomaly detection
        if self.anomaly_detector:
            anomaly_pred = self.anomaly_detector.predict(features)
            anomaly_score = self.anomaly_detector.score_samples(features)[0]
            
            result['is_anomaly'] = bool(anomaly_pred[0] == -1)
            result['anomaly_score'] = float(anomaly_score)
        
        if return_features:
            result['features'] = features
        
        return result
    
    def predict_batch(
        self,
        audio_batch: np.ndarray,
        sample_rate: int = 16000
    ) -> list:
        """Predict on batch of audio samples."""
        results = []
        
        for audio in audio_batch:
            result = self.predict(audio, sample_rate)
            results.append(result)
        
        return results
    
    def get_risk_level(self, prediction: str, confidence: float) -> str:
        """Determine risk level based on prediction."""
        if prediction == 'Normal':
            return 'Low'
        elif confidence > 0.8:
            return 'High'
        elif confidence > 0.6:
            return 'Medium'
        else:
            return 'Low'
