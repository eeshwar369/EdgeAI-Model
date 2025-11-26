"""
Test inference on sample audio file.
"""

import sys
sys.path.append('.')

import argparse
import librosa
import numpy as np
from src.inference_engine import RespiratoryInferenceEngine


def main():
    parser = argparse.ArgumentParser(description='Test respiratory disease inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/crnn_best.h5', help='Model path')
    parser.add_argument('--tflite', action='store_true', help='Use TFLite model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RESPIRATORY DISEASE INFERENCE TEST")
    print("=" * 60)
    
    # Load audio
    print(f"\nLoading audio: {args.audio}")
    audio, sr = librosa.load(args.audio, sr=16000)
    print(f"Audio duration: {len(audio) / sr:.2f}s")
    
    # Initialize inference engine
    print(f"\nLoading model: {args.model}")
    if args.tflite:
        model_path = 'models/quantized_model.tflite'
    else:
        model_path = args.model
    
    engine = RespiratoryInferenceEngine(model_path, use_tflite=args.tflite)
    
    # Predict
    print("\nRunning inference...")
    result = engine.predict(audio, sample_rate=sr)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    risk_level = engine.get_risk_level(result['prediction'], result['confidence'])
    print(f"Risk Level: {risk_level}")
    
    print("\nAll Probabilities:")
    for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 50)
        print(f"  {label:15s}: {prob:.4f} {bar}")
    
    if 'is_anomaly' in result:
        print(f"\nAnomaly Detection:")
        print(f"  Is Anomaly: {result['is_anomaly']}")
        print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
