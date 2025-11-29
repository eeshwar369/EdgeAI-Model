# Edge Impulse Setup Guide

Complete guide for training and deploying EdgeSense using Edge Impulse.

## Prerequisites

- Edge Impulse account (free at [edgeimpulse.com](https://edgeimpulse.com))
- Node.js installed
- Python 3.8+
- Audio dataset prepared

## Step 1: Install Edge Impulse CLI

```bash
npm install -g edge-impulse-cli
```

Verify installation:
```bash
edge-impulse-cli --version
```

## Step 2: Create New Project

1. Go to [studio.edgeimpulse.com](https://studio.edgeimpulse.com)
2. Click "Create new project"
3. Name: "EdgeSense Respiratory Detection"
4. Project type: "Audio"

## Step 3: Prepare Data

### Organize Audio Files

Structure your data:
```
data/
├── normal/
│   ├── sample001.wav
│   ├── sample002.wav
│   └── ...
├── asthma/
├── copd/
├── pneumonia/
├── bronchitis/
├── tuberculosis/
└── long_covid/
```

### Audio Requirements

- Format: WAV or MP3
- Sample rate: 16000 Hz
- Duration: 1-10 seconds
- Mono channel

### Convert Audio (if needed)

```bash
python scripts/preprocess_audio.py
```

## Step 4: Upload Data to Edge Impulse

### Method 1: Using CLI

```bash
# Login
edge-impulse-uploader

# Upload training data
edge-impulse-uploader --category training data/normal/*.wav --label normal
edge-impulse-uploader --category training data/asthma/*.wav --label asthma
edge-impulse-uploader --category training data/copd/*.wav --label copd
edge-impulse-uploader --category training data/pneumonia/*.wav --label pneumonia
edge-impulse-uploader --category training data/bronchitis/*.wav --label bronchitis
edge-impulse-uploader --category training data/tuberculosis/*.wav --label tuberculosis
edge-impulse-uploader --category training data/long_covid/*.wav --label long_covid
```

### Method 2: Using Python Script

```bash
python scripts/upload_to_edge_impulse.py
```

### Method 3: Web Interface

1. Go to "Data acquisition"
2. Click "Upload data"
3. Select files
4. Choose category (training/testing)
5. Assign labels

## Step 5: Create Impulse

### 5.1 Configure Input

1. Go to "Create impulse"
2. Set time series data:
   - Window size: 3000 ms
   - Window increase: 500 ms
   - Frequency: 16000 Hz

### 5.2 Add Processing Block

1. Click "Add a processing block"
2. Select "Audio (MFCC)"
3. Click "Add"

### 5.3 Add Learning Block

1. Click "Add a learning block"
2. Select "Classification (Keras)"
3. Click "Add"
4. Click "Save Impulse"

## Step 6: Configure MFCC Parameters

1. Go to "MFCC" in left sidebar
2. Configure parameters:
   - Number of coefficients: 40
   - Frame length: 0.025
   - Frame stride: 0.010
   - Filter number: 128
   - FFT length: 2048
   - Low frequency: 20
   - High frequency: 8000
3. Click "Save parameters"
4. Click "Generate features"
5. Wait for feature generation to complete

## Step 7: Design Neural Network

### 7.1 Architecture

1. Go to "NN Classifier"
2. Use this architecture:

```
Input layer (auto)
↓
Conv2D (32 filters, 3x3, ReLU)
↓
BatchNormalization
↓
Conv2D (64 filters, 3x3, ReLU)
↓
BatchNormalization
↓
MaxPooling2D (2x2)
↓
Dropout (0.3)
↓
Reshape
↓
LSTM (128 units)
↓
BatchNormalization
↓
Dropout (0.5)
↓
Dense (256, ReLU)
↓
BatchNormalization
↓
Dropout (0.5)
↓
Dense (7, Softmax)
```

### 7.2 Training Settings

- Number of training cycles: 100
- Learning rate: 0.0005
- Validation set size: 15%
- Auto-balance dataset: Yes
- Data augmentation: Enable

### 7.3 Start Training

1. Click "Start training"
2. Wait for training to complete (10-30 minutes)
3. Review accuracy and confusion matrix

## Step 8: Test Model

1. Go to "Model testing"
2. Click "Classify all"
3. Review performance metrics:
   - Accuracy
   - Precision
   - Recall
   - F1 Score

Expected results:
- Accuracy: >85%
- Per-class accuracy: >80%

## Step 9: Deploy Model

### For Raspberry Pi

1. Go to "Deployment"
2. Select "Linux (ARMv7)"
3. Click "Build"
4. Download `.eim` file
5. Transfer to Raspberry Pi:
```bash
scp edge-impulse-linux-*.eim pi@raspberrypi.local:~/
```
6. Run on Pi:
```bash
edge-impulse-linux-runner
```

### For Android

1. Go to "Deployment"
2. Select "Android library"
3. Click "Build"
4. Download AAR file
5. Add to Android project:
   - Copy to `android/app/libs/`
   - Update `build.gradle`

### For ESP32

1. Go to "Deployment"
2. Select "Arduino library"
3. Click "Build"
4. Download ZIP
5. Install in Arduino IDE:
   - Sketch > Include Library > Add .ZIP Library

### For TensorFlow Lite

1. Go to "Deployment"
2. Select "TensorFlow Lite (int8)"
3. Enable "EON Compiler" for optimization
4. Click "Build"
5. Download model files

## Step 10: Optimize Model

### Quantization

- Already applied with INT8 selection
- Reduces model size by ~75%
- Minimal accuracy loss (<2%)

### EON Compiler

- Enable in deployment settings
- Further optimizes for edge devices
- Reduces inference time

## Troubleshooting

### Issue: Low Accuracy

**Solutions:**
- Collect more training data (aim for 500+ samples per class)
- Balance dataset (equal samples per class)
- Enable data augmentation
- Increase training cycles
- Adjust learning rate

### Issue: Overfitting

**Solutions:**
- Increase dropout rates
- Add more data augmentation
- Reduce model complexity
- Use early stopping

### Issue: Upload Fails

**Solutions:**
- Check audio format (WAV, 16kHz)
- Verify file size (<10MB per file)
- Check internet connection
- Try uploading in smaller batches

### Issue: Feature Generation Fails

**Solutions:**
- Verify audio duration (1-10 seconds)
- Check sample rate (16000 Hz)
- Ensure mono channel
- Re-upload problematic files

## Performance Benchmarks

Expected performance after training:

| Metric | Target | Typical |
|--------|--------|---------|
| Accuracy | >85% | 89-92% |
| Model Size | <1MB | 500-700KB |
| Inference (RPi) | <100ms | 30-50ms |
| Inference (ESP32) | <500ms | 200-300ms |

## Best Practices

1. **Data Quality**
   - Use high-quality recordings
   - Remove background noise
   - Consistent audio length

2. **Balanced Dataset**
   - Equal samples per class
   - Diverse recording conditions
   - Multiple speakers/sources

3. **Validation**
   - Use separate test set
   - Test on real-world data
   - Verify on target hardware

4. **Iteration**
   - Start with baseline model
   - Gradually increase complexity
   - Monitor validation accuracy

## Next Steps

After successful deployment:

1. Test on real audio samples
2. Collect feedback
3. Retrain with new data
4. Deploy updated model
5. Monitor performance

## Resources

- [Edge Impulse Documentation](https://docs.edgeimpulse.com/)
- [Audio Classification Tutorial](https://docs.edgeimpulse.com/docs/audio-classification)
- [Model Optimization Guide](https://docs.edgeimpulse.com/docs/edge-impulse-studio/model-optimization)
- [Deployment Options](https://docs.edgeimpulse.com/docs/deployment)

## Support

- Edge Impulse Forum: [forum.edgeimpulse.com](https://forum.edgeimpulse.com)
- Documentation: [docs.edgeimpulse.com](https://docs.edgeimpulse.com)
- GitHub Issues: For project-specific questions
