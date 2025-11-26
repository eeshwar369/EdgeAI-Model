"""
Anomaly detection for unknown respiratory patterns.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


class RespiratoryAnomalyDetector:
    """Detect anomalous respiratory patterns using Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, features: np.ndarray):
        """Train anomaly detector on normal patterns."""
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        self.model.fit(features_scaled)
        self.is_fitted = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict if samples are anomalies.
        Returns: 1 for normal, -1 for anomaly
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        features_scaled = self.scaler.transform(features)
        scores = self.model.score_samples(features_scaled)
        
        return scores
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
