"""
Data loading and preprocessing utilities for respiratory audio datasets.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class RespiratoryDataLoader:
    """Load and preprocess respiratory audio datasets."""
    
    def __init__(
        self,
        data_dir: str = 'data/raw',
        sample_rate: int = 16000,
        duration: float = 3.0
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.label_map = {
            'normal': 0,
            'asthma': 1,
            'copd': 2,
            'pneumonia': 3,
            'bronchitis': 4,
            'tuberculosis': 5,
            'long_covid': 6
        }
    
    def load_audio_file(self, file_path: str) -> np.ndarray:
        """Load and preprocess single audio file."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad if too short
            target_length = int(self.sample_rate * self.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_dataset_from_directory(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load dataset from directory structure:
        data/raw/
            normal/
            asthma/
            copd/
            ...
        """
        audio_data = []
        labels = []
        file_paths = []
        
        for label_name, label_id in self.label_map.items():
            label_dir = self.data_dir / label_name
            
            if not label_dir.exists():
                print(f"Warning: Directory {label_dir} not found")
                continue
            
            audio_files = list(label_dir.glob('*.wav')) + list(label_dir.glob('*.mp3'))
            
            print(f"Loading {len(audio_files)} files from {label_name}...")
            
            for audio_file in tqdm(audio_files, desc=label_name):
                audio = self.load_audio_file(str(audio_file))
                if audio is not None:
                    audio_data.append(audio)
                    labels.append(label_id)
                    file_paths.append(str(audio_file))
        
        return audio_data, labels, file_paths
    
    def load_from_csv(self, csv_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load dataset from CSV manifest.
        CSV format: file_path, label
        """
        df = pd.read_csv(csv_path)
        audio_data = []
        labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading from CSV"):
            audio = self.load_audio_file(row['file_path'])
            if audio is not None:
                audio_data.append(audio)
                labels.append(self.label_map[row['label']])
        
        return audio_data, labels
    
    def create_train_val_test_split(
        self,
        audio_data: List[np.ndarray],
        labels: List[int],
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple:
        """Split data into train, validation, and test sets."""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            audio_data, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        return (
            np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test)
        )
    
    def save_processed_data(
        self,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        output_dir: str = 'data/processed'
    ):
        """Save processed data to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'X_train.npy', X_train)
        np.save(output_path / 'y_train.npy', y_train)
        np.save(output_path / 'X_val.npy', X_val)
        np.save(output_path / 'y_val.npy', y_val)
        np.save(output_path / 'X_test.npy', X_test)
        np.save(output_path / 'y_test.npy', y_test)
        
        print(f"Processed data saved to {output_dir}")
    
    def load_processed_data(
        self,
        data_dir: str = 'data/processed'
    ) -> Tuple:
        """Load preprocessed data from disk."""
        data_path = Path(data_dir)
        
        X_train = np.load(data_path / 'X_train.npy')
        y_train = np.load(data_path / 'y_train.npy')
        X_val = np.load(data_path / 'X_val.npy')
        y_val = np.load(data_path / 'y_val.npy')
        X_test = np.load(data_path / 'X_test.npy')
        y_test = np.load(data_path / 'y_test.npy')
        
        return X_train, y_train, X_val, y_val, X_test, y_test


def augment_audio(audio: np.ndarray, sr: int = 16000) -> List[np.ndarray]:
    """Apply audio augmentation techniques."""
    augmented = [audio]  # Original
    
    # Time stretch
    augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
    augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    
    # Pitch shift
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))
    
    # Add noise
    noise = np.random.normal(0, 0.005, audio.shape)
    augmented.append(audio + noise)
    
    return augmented
