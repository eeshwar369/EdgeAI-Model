"""
Preprocess respiratory audio datasets.
"""

import sys
sys.path.append('.')

from src.data_loader import RespiratoryDataLoader
from src.feature_extractor import RespiratoryFeatureExtractor
import numpy as np
from pathlib import Path


def main():
    print("=" * 60)
    print("RESPIRATORY AUDIO PREPROCESSING")
    print("=" * 60)
    
    # Initialize data loader
    loader = RespiratoryDataLoader(data_dir='data/raw')
    
    # Load dataset
    print("\nLoading audio files...")
    audio_data, labels, file_paths = loader.load_dataset_from_directory()
    
    print(f"\nLoaded {len(audio_data)} audio samples")
    print(f"Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label_id, count in zip(unique, counts):
        label_name = [k for k, v in loader.label_map.items() if v == label_id][0]
        print(f"  {label_name}: {count}")
    
    # Split data
    print("\nSplitting into train/val/test...")
    X_train, y_train, X_val, y_val, X_test, y_test = loader.create_train_val_test_split(
        audio_data, labels
    )
    
    print(f"Train: {len(X_train)}")
    print(f"Val: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    # Extract features
    print("\nExtracting features...")
    feature_extractor = RespiratoryFeatureExtractor()
    
    def extract_features_batch(audio_batch):
        features = []
        for audio in audio_batch:
            feat = feature_extractor.prepare_model_input(audio)
            features.append(feat)
        return np.array(features)
    
    X_train_features = extract_features_batch(X_train)
    X_val_features = extract_features_batch(X_val)
    X_test_features = extract_features_batch(X_test)
    
    print(f"Feature shape: {X_train_features.shape}")
    
    # Save processed data
    print("\nSaving processed data...")
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'X_train.npy', X_train_features)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'X_val.npy', X_val_features)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'X_test.npy', X_test_features)
    np.save(output_dir / 'y_test.npy', y_test)
    
    print(f"\nProcessed data saved to {output_dir}")
    print("=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
