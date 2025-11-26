"""
Train respiratory disease classification model.
"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
from tensorflow import keras
from src.model_builder import (
    build_crnn_model,
    compile_model,
    get_callbacks,
    convert_to_tflite
)


def main():
    print("=" * 60)
    print("RESPIRATORY DISEASE MODEL TRAINING")
    print("=" * 60)
    
    # Load processed data
    print("\nLoading processed data...")
    data_dir = Path('data/processed')
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=7)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=7)
    
    # Build model
    print("\nBuilding CRNN model...")
    input_shape = X_train.shape[1:]
    model = build_crnn_model(input_shape, num_classes=7)
    
    print(f"Input shape: {input_shape}")
    model.summary()
    
    # Compile model
    print("\nCompiling model...")
    model = compile_model(model, learning_rate=0.0005, optimizer='adamw')
    
    # Prepare callbacks
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    callbacks = get_callbacks(
        model_path=str(models_dir / 'crnn_best.h5'),
        patience=15
    )
    
    # Train model
    print("\nTraining model...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(models_dir / 'crnn_final.h5')
    print(f"\nFinal model saved to {models_dir / 'crnn_final.h5'}")
    
    # Convert to TFLite
    print("\nConverting to TensorFlow Lite...")
    convert_to_tflite(
        model,
        output_path=str(models_dir / 'quantized_model.tflite'),
        quantize=True
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Print final metrics
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")


if __name__ == '__main__':
    main()
