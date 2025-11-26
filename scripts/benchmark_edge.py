"""
Benchmark model performance on edge devices.
"""

import sys
sys.path.append('.')

import time
import numpy as np
import argparse
from pathlib import Path
import tensorflow as tf
from src.feature_extractor import RespiratoryFeatureExtractor


def benchmark_model(model_path: str, num_iterations: int = 100):
    """Benchmark model inference speed."""
    
    print("=" * 60)
    print("EDGE DEVICE BENCHMARK")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    
    if model_path.endswith('.tflite'):
        # TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        print(f"Input shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm-up
        print("\nWarming up...")
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # Benchmark
        print(f"\nRunning {num_iterations} iterations...")
        times = []
        
        for i in range(num_iterations):
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            end = time.time()
            
            times.append((end - start) * 1000)  # Convert to ms
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
        
        # Get model size
        model_size = Path(model_path).stat().st_size / 1024  # KB
        
    else:
        # Keras model
        model = tf.keras.models.load_model(model_path)
        
        input_shape = model.input_shape
        print(f"Input shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(1, *input_shape[1:]).astype(np.float32)
        
        # Warm-up
        print("\nWarming up...")
        for _ in range(10):
            model.predict(dummy_input, verbose=0)
        
        # Benchmark
        print(f"\nRunning {num_iterations} iterations...")
        times = []
        
        for i in range(num_iterations):
            start = time.time()
            output = model.predict(dummy_input, verbose=0)
            end = time.time()
            
            times.append((end - start) * 1000)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
        
        # Estimate model size
        model_size = sum([np.prod(w.shape) * 4 for w in model.get_weights()]) / 1024  # KB
    
    # Calculate statistics
    times = np.array(times)
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\nModel: {Path(model_path).name}")
    print(f"Model Size: {model_size:.2f} KB")
    print(f"\nInference Time:")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min: {np.min(times):.2f} ms")
    print(f"  Max: {np.max(times):.2f} ms")
    print(f"  Std: {np.std(times):.2f} ms")
    
    print(f"\nThroughput:")
    print(f"  FPS: {1000 / np.mean(times):.2f}")
    print(f"  Samples/sec: {1000 / np.mean(times):.2f}")
    
    # Estimate RAM usage (rough approximation)
    ram_usage = model_size * 2  # Model + activations
    print(f"\nEstimated RAM Usage: {ram_usage:.2f} KB")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Benchmark edge device performance')
    parser.add_argument('--model', type=str, default='models/quantized_model.tflite',
                       help='Path to model file')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    benchmark_model(args.model, args.iterations)


if __name__ == '__main__':
    main()
