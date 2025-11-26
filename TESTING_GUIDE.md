# EdgeSense Testing Guide

Complete guide for testing all components of EdgeSense.

---

## 🧪 Testing Overview

EdgeSense includes multiple testing levels:
1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Component interaction testing
3. **System Tests** - End-to-end testing
4. **Performance Tests** - Benchmarking and profiling

---

## 🚀 Quick Test

```bash
# Create synthetic sample data
python scripts/create_sample_data.py

# Test inference on sample
python scripts/test_inference.py --audio samples/asthma/asthma_001.wav

# Run unit tests
pytest tests/ -v
```

---

## 📋 Pre-Testing Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Sample data available (run `create_sample_data.py`)
- [ ] Model trained or downloaded

---

## 1️⃣ Unit Tests

### Run All Unit Tests

```bash
pytest tests/ -v
```

### Run Specific Test

```bash
pytest tests/test_inference.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Test Individual Components

**Feature Extraction:**
```python
from src.feature_extractor import RespiratoryFeatureExtractor
import numpy as np

extractor = RespiratoryFeatureExtractor()
audio = np.random.randn(48000)  # 3 seconds

# Test MFCC
mfcc = extractor.extract_mfcc(audio)
assert mfcc.shape[0] == 120  # 40 + 40 + 40

# Test Mel-Spectrogram
mel_spec = extractor.extract_mel_spectrogram(audio)
assert mel_spec.shape[0] == 128

print("✅ Feature extraction tests passed")
```

**Data Loading:**
```python
from src.data_loader import RespiratoryDataLoader

loader = RespiratoryDataLoader(data_dir='samples')
audio_data, labels, paths = loader.load_dataset_from_directory()

assert len(audio_data) > 0
assert len(audio_data) == len(labels)

print(f"✅ Loaded {len(audio_data)} samples")
```

---

## 2️⃣ Integration Tests

### Test Complete Pipeline

```bash
# 1. Create sample data
python scripts/create_sample_data.py

# 2. Test preprocessing
python scripts/preprocess_audio.py

# 3. Test training (quick test with few epochs)
# Edit train_model.py to set epochs=5 for testing

# 4. Test evaluation
python scripts/evaluate_model.py

# 5. Test inference
python scripts/test_inference.py --audio samples/asthma/asthma_001.wav
```

### Test Feature Extraction Pipeline

```python
import numpy as np
from src.data_loader import RespiratoryDataLoader
from src.feature_extractor import RespiratoryFeatureExtractor

# Load data
loader = RespiratoryDataLoader(data_dir='samples')
audio_data, labels, _ = loader.load_dataset_from_directory()

# Extract features
extractor = RespiratoryFeatureExtractor()
features = []

for audio in audio_data[:5]:  # Test first 5
    feat = extractor.prepare_model_input(audio)
    features.append(feat)
    print(f"Feature shape: {feat.shape}")

print("✅ Feature extraction pipeline works")
```

---

## 3️⃣ System Tests

### End-to-End Test

```bash
# Complete workflow test
make pipeline
```

Or manually:

```bash
# 1. Create samples
python scripts/create_sample_data.py

# 2. Preprocess
python scripts/preprocess_audio.py

# 3. Train (quick)
python scripts/train_model.py

# 4. Evaluate
python scripts/evaluate_model.py

# 5. Test inference
python scripts/test_inference.py --audio samples/normal/normal_001.wav
```

### Test All Sample Classes

```bash
# Test each disease class
for class in normal asthma copd pneumonia bronchitis tuberculosis long_covid; do
    echo "Testing $class..."
    python scripts/test_inference.py --audio samples/$class/${class}_001.wav
done
```

---

## 4️⃣ Performance Tests

### Benchmark Inference Speed

```bash
python scripts/benchmark_edge.py --model models/quantized_model.tflite --iterations 100
```

Expected output:
```
Model Size: 567 KB
Inference Time:
  Mean: 34.2 ms
  Median: 33.8 ms
Throughput:
  FPS: 29.4
```

### Memory Profiling

```bash
pip install memory_profiler

python -m memory_profiler scripts/test_inference.py --audio samples/asthma/asthma_001.wav
```

### CPU Profiling

```bash
pip install py-spy

py-spy record -o profile.svg -- python scripts/test_inference.py --audio samples/asthma/asthma_001.wav
```

---

## 5️⃣ Model Tests

### Test Model Architecture

```python
from src.model_builder import build_crnn_model

input_shape = (None, 168, 1)
model = build_crnn_model(input_shape)

# Check output shape
assert model.output_shape == (None, 7)

# Check parameter count
params = model.count_params()
print(f"Model parameters: {params:,}")
assert params < 1_000_000  # Should be under 1M

print("✅ Model architecture tests passed")
```

### Test Model Inference

```python
import numpy as np
from tensorflow import keras

model = keras.models.load_model('models/crnn_best.h5')

# Create dummy input
dummy_input = np.random.randn(1, 300, 168, 1).astype(np.float32)

# Run inference
output = model.predict(dummy_input)

# Check output
assert output.shape == (1, 7)
assert np.isclose(np.sum(output), 1.0, atol=0.01)  # Softmax sums to 1

print("✅ Model inference tests passed")
```

### Test TFLite Model

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/quantized_model.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create dummy input
input_shape = input_details[0]['shape']
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# Check output
assert output.shape[1] == 7
print("✅ TFLite model tests passed")
```

---

## 6️⃣ Edge Device Tests

### Raspberry Pi Test

```bash
# SSH into Raspberry Pi
ssh pi@raspberrypi.local

# Run test
cd ~/edgesense
python3 scripts/test_inference.py --audio samples/asthma/asthma_001.wav --tflite

# Benchmark
python3 scripts/benchmark_edge.py --model models/quantized_model.tflite
```

### Android Test

1. Build and install APK
2. Grant microphone permission
3. Start detection
4. Verify predictions appear
5. Check latency in logs

```bash
adb logcat | grep EdgeSense
```

---

## 7️⃣ API Tests

### Start API Server

```bash
python api_server.py
```

### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Test Prediction Endpoint

```bash
curl -X POST -F "audio=@samples/asthma/asthma_001.wav" http://localhost:8000/predict
```

Expected:
```json
{
  "prediction": "Asthma",
  "confidence": 0.87,
  "probabilities": {...},
  "risk_level": "High",
  "inference_time_ms": 34.2
}
```

### Load Test

```bash
pip install locust

# Create locustfile.py
cat > locustfile.py << 'EOF'
from locust import HttpUser, task, between

class EdgeSenseUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        with open('samples/asthma/asthma_001.wav', 'rb') as f:
            self.client.post('/predict', files={'audio': f})
EOF

# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

---

## 8️⃣ Data Quality Tests

### Test Audio Files

```python
import librosa
from pathlib import Path

def test_audio_file(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Check duration
        duration = len(audio) / sr
        assert duration >= 1.0, f"Audio too short: {duration}s"
        
        # Check amplitude
        max_amp = np.max(np.abs(audio))
        assert max_amp > 0, "Silent audio"
        assert max_amp <= 1.0, "Clipping detected"
        
        # Check for NaN/Inf
        assert not np.any(np.isnan(audio)), "NaN values found"
        assert not np.any(np.isinf(audio)), "Inf values found"
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test all samples
samples_dir = Path('samples')
for audio_file in samples_dir.rglob('*.wav'):
    result = test_audio_file(audio_file)
    print(f"{'✅' if result else '❌'} {audio_file}")
```

---

## 9️⃣ Regression Tests

### Save Baseline Results

```bash
# Run evaluation and save results
python scripts/evaluate_model.py > baseline_results.txt
```

### Compare with Baseline

```bash
# After changes, run again
python scripts/evaluate_model.py > new_results.txt

# Compare
diff baseline_results.txt new_results.txt
```

---

## 🔟 Continuous Integration Tests

### GitHub Actions

Tests run automatically on push/PR. Check `.github/workflows/ci.yml`

### Local CI Simulation

```bash
# Run all CI tests locally
pip install -r requirements.txt
flake8 src/ scripts/
pytest tests/ -v --cov=src
```

---

## 🐛 Debugging Tests

### Verbose Output

```bash
pytest tests/ -v -s
```

### Run Single Test

```bash
pytest tests/test_inference.py::test_feature_extraction -v
```

### Debug with pdb

```python
import pdb; pdb.set_trace()
```

### Check Logs

```bash
tail -f logs/edgesense.log
```

---

## ✅ Test Checklist

### Before Committing
- [ ] All unit tests pass
- [ ] Code coverage > 80%
- [ ] No linting errors
- [ ] Documentation updated

### Before Deploying
- [ ] Integration tests pass
- [ ] Performance benchmarks acceptable
- [ ] Edge device tests successful
- [ ] API tests pass

### Before Competition
- [ ] End-to-end test successful
- [ ] Demo works on target hardware
- [ ] All sample classes tested
- [ ] Documentation complete

---

## 📊 Test Results Format

### Expected Test Output

```
================================ test session starts =================================
platform linux -- Python 3.9.0, pytest-7.1.0
collected 10 items

tests/test_inference.py::test_preprocess_audio PASSED                         [ 10%]
tests/test_inference.py::test_feature_extraction PASSED                       [ 20%]
tests/test_inference.py::test_mfcc_shape PASSED                              [ 30%]
tests/test_inference.py::test_mel_spectrogram_shape PASSED                   [ 40%]
tests/test_inference.py::test_model_input_preparation PASSED                 [ 50%]
tests/test_inference.py::test_inference_engine PASSED                        [ 60%]
tests/test_inference.py::test_anomaly_detector PASSED                        [ 70%]
tests/test_inference.py::test_data_loader PASSED                             [ 80%]
tests/test_inference.py::test_model_builder PASSED                           [ 90%]
tests/test_inference.py::test_end_to_end PASSED                              [100%]

================================ 10 passed in 12.34s =================================
```

---

## 🚨 Common Test Failures

### Issue: "Model not found"
**Solution:**
```bash
# Train model first
python scripts/train_model.py
```

### Issue: "No audio data"
**Solution:**
```bash
# Create sample data
python scripts/create_sample_data.py
```

### Issue: "Import error"
**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt
```

### Issue: "TensorFlow error"
**Solution:**
```bash
# Reinstall TensorFlow
pip install --upgrade tensorflow==2.10.0
```

---

## 📞 Getting Help

If tests fail:
1. Check error messages carefully
2. Review relevant documentation
3. Check GitHub issues
4. Ask in discussions
5. Contact: your-email@example.com

---

**Happy Testing! 🧪✅**
