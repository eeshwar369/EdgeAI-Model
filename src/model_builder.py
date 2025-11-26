"""
Model architecture definitions for respiratory disease classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple


def build_crnn_model(
    input_shape: Tuple[int, int, int],
    num_classes: int = 7,
    lstm_units: int = 128,
    dense_units: int = 256,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Build CRNN (CNN + LSTM) model for respiratory classification.
    Best performing architecture from experiments.
    """
    
    inputs = layers.Input(shape=input_shape, name='audio_input')
    
    # CNN layers for feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Reshape for LSTM
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    
    # LSTM for temporal patterns
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense layers
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CRNN_Respiratory')
    
    return model


def build_baseline_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int = 7
) -> keras.Model:
    """Build baseline CNN model."""
    
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Baseline_CNN')
    
    return model


def build_deep_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int = 7
) -> keras.Model:
    """Build deeper CNN model (Iteration 2)."""
    
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Deep_CNN')
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.0005,
    optimizer: str = 'adamw'
) -> keras.Model:
    """Compile model with optimizer and loss."""
    
    if optimizer == 'adamw':
        opt = keras.optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def get_callbacks(
    model_path: str = 'models/best_model.h5',
    patience: int = 15
) -> list:
    """Get training callbacks."""
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    return callbacks


def convert_to_tflite(
    model: keras.Model,
    output_path: str = 'models/quantized_model.tflite',
    quantize: bool = True
) -> None:
    """Convert Keras model to TensorFlow Lite for edge deployment."""
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Post-training quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
