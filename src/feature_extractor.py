"""
Feature extraction module for respiratory audio analysis.
Extracts MFCC, Mel-Spectrogram, and other acoustic features.
"""

import numpy as np
import librosa
import scipy.signal as signal
from typing import Dict, Tuple


class RespiratoryFeatureExtractor:
    """Extract acoustic features from respiratory audio."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 160,
        fmin: int = 20,
        fmax: int = 8000
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        # Add delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        return np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract Mel-Spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features."""
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85
        )[0]
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        return features
    
    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chroma
    
    def extract_breathing_cadence(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract breathing rhythm patterns."""
        # Apply bandpass filter for breathing frequency (0.1-1 Hz)
        sos = signal.butter(4, [0.1, 1.0], btype='band', fs=self.sample_rate, output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        # Detect peaks (breathing cycles)
        peaks, properties = signal.find_peaks(
            np.abs(filtered),
            distance=self.sample_rate * 0.5,  # Min 0.5s between breaths
            prominence=0.1
        )
        
        # Calculate breathing rate
        duration = len(audio) / self.sample_rate
        breathing_rate = len(peaks) / duration * 60  # breaths per minute
        
        # Calculate regularity (coefficient of variation)
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sample_rate
            regularity = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        else:
            regularity = 0
        
        return {
            'breathing_rate': breathing_rate,
            'regularity': regularity,
            'num_cycles': len(peaks)
        }
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features for model input."""
        features = {}
        
        # Core features
        features['mfcc'] = self.extract_mfcc(audio)
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        features['chroma'] = self.extract_chroma_features(audio)
        
        # Spectral features
        spectral = self.extract_spectral_features(audio)
        features.update(spectral)
        
        # Breathing patterns
        cadence = self.extract_breathing_cadence(audio)
        features['breathing_cadence'] = cadence
        
        return features
    
    def prepare_model_input(self, audio: np.ndarray) -> np.ndarray:
        """Prepare features for model inference (MFCC + Mel-Spec combined)."""
        mfcc = self.extract_mfcc(audio)
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Ensure same time dimension
        min_time = min(mfcc.shape[1], mel_spec.shape[1])
        mfcc = mfcc[:, :min_time]
        mel_spec = mel_spec[:, :min_time]
        
        # Stack features
        combined = np.vstack([mfcc, mel_spec])
        
        # Reshape for CNN input (height, width, channels)
        combined = combined.T  # (time, features)
        combined = np.expand_dims(combined, axis=-1)  # (time, features, 1)
        
        return combined


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int = 16000,
    duration: float = 3.0,
    normalize: bool = True
) -> np.ndarray:
    """Preprocess audio: resample, trim/pad, normalize."""
    
    # Resample if needed
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    
    # Target length
    target_length = int(target_sr * duration)
    
    # Trim or pad
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    
    # Normalize
    if normalize and np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio
