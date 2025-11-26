"""EdgeSense: Multi-Disease Respiratory Detection via Acoustic Biomarkers"""

__version__ = "1.0.0"
__author__ = "EdgeSense Team"

from .data_loader import RespiratoryDataLoader
from .feature_extractor import RespiratoryFeatureExtractor
from .model_builder import build_crnn_model, compile_model
from .inference_engine import RespiratoryInferenceEngine
from .anomaly_detector import RespiratoryAnomalyDetector

__all__ = [
    'RespiratoryDataLoader',
    'RespiratoryFeatureExtractor',
    'build_crnn_model',
    'compile_model',
    'RespiratoryInferenceEngine',
    'RespiratoryAnomalyDetector'
]
