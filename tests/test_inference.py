"""
Unit tests for inference engine.
"""

import pytest
import numpy as np
from src.feature_extractor import RespiratoryFeatureExtractor, preprocess_audio


def test_preprocess_audio():
    """Test audio preprocessing."""
    # Create dummy audio
    audio = np.random.randn(32000)  # 2 seconds at 16kHz
    sample_rate = 16000
    
    # Preprocess
    processed = preprocess_audio(audio, sample_rate, duration=3.0)
    
    # Check output
    assert len(processed) == 48000  # 3 seconds at 16kHz
    assert np.max(np.abs(processed)) <= 1.0  # Normalized


def test_feature_extraction():
    """Test feature extraction."""
    extractor = RespiratoryFeatureExtractor()
    
    # Create dummy audio
    audio = np.random.randn(48000)  # 3 seconds at 16kHz
    
    # Extract MFCC
    mfcc = extractor.extract_mfcc(audio)
    assert mfcc.shape[0] == 120  # 40 MFCC + 40 delta + 40 delta-delta
    
    # Extract Mel-Spectrogram
    mel_spec = extractor.extract_mel_spectrogram(audio)
    assert mel_spec.shape[0] == 128  # 128 mel bins
    
    # Extract all features
    features = extractor.extract_all_features(audio)
    assert 'mfcc' in features
    assert 'mel_spectrogram' in features
    assert 'breathing_cadence' in features


def test_model_input_preparation():
    """Test model input preparation."""
    extractor = RespiratoryFeatureExtractor()
    
    # Create dummy audio
    audio = np.random.randn(48000)
    
    # Prepare model input
    model_input = extractor.prepare_model_input(audio)
    
    # Check shape (time, features, channels)
    assert len(model_input.shape) == 3
    assert model_input.shape[-1] == 1  # Single channel


if __name__ == '__main__':
    pytest.main([__file__])
