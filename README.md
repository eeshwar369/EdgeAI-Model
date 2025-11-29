# EdgeSense: Respiratory Disease Detection

A machine learning system for detecting respiratory diseases through audio analysis of breathing and cough sounds. Optimized for deployment on edge devices including Raspberry Pi, Android, and ESP32.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Multi-class classification**: Detects 7 different respiratory conditions
- **Edge deployment**: Runs on resource-constrained devices
- **Real-time inference**: Low latency processing (<100ms)
- **Multiple platforms**: Raspberry Pi, Android, ESP32 support

## Supported Conditions

The system can classify the following respiratory patterns:
- Normal breathing
- Asthma
- COPD (Chronic Obstructive Pulmonary Disease)
- Pneumonia
- Bronchitis
- Tuberculosis
- Long-COVID

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/edgesense.git
cd edgesense
pip install -r requirements.txt
```

### Create Sample Data

```bash
python scripts/create_sample_data.py
```

### Test Inference

```bash
python scripts/test_inference.py --audio samples/asthma/asthma_001.wav
```

## Training the Model

### 1. Prepare Dataset

Download respiratory sound datasets from public sources:
- ICBHI 2017 Respiratory Sound Database
- COUGHVID Dataset
- Coswara Dataset

Place audio files in `data/raw/` organized by class.

### 2. Preprocess Data

```bash
python scripts/preprocess_audio.py
```

This will:
- Resample audio to 16kHz
- Normalize amplitude
- Extract features (MFCC, Mel-Spectrogram)
- Split into train/val/test sets

### 3. Train Model

```bash
python scripts/train_model.py
```

Training parameters can be adjusted in `config.yaml`.

### 4. Evaluate

```bash
python scripts/evaluate_model.py
```

## Edge Impulse Integration

### Setup

1. Install Edge Impulse CLI:
```bash
npm install -g edge-impulse-cli
```

2. Create a new project at [edgeimpulse.com](https://edgeimpulse.com)

3. Upload data:
```bash
python scripts/upload_to_edge_impulse.py
```

### Configure DSP Block

In Edge Impulse Studio:
1. Go to "Create Impulse"
2. Add "Audio (MFCC)" processing block
   - Window size: 3000ms
   - Window increase: 500ms
   - Frequency: 16000Hz
3. Configure MFCC parameters:
   - Number of coefficients: 40
   - Frame length: 0.025
   - Frame stride: 0.010
   - FFT length: 2048

### Train in Edge Impulse

1. Go to "NN Classifier"
2. Use the CRNN architecture (provided in `edge-impulse-project.json`)
3. Set training cycles: 100
4. Start training

### Deploy

After training, deploy to your target device:
- **Raspberry Pi**: Download Linux (ARMv7) library
- **Android**: Download Android library (AAR)
- **ESP32**: Download Arduino library

## Deployment

### Raspberry Pi

```bash
cd raspberry_pi
./deploy.sh
python3 realtime_inference.py
```

### Android

Open the `android/` directory in Android Studio and build the APK.

### API Server

```bash
python api_server.py
```

Access at `http://localhost:8000`

## Project Structure

```
edgesense/
├── src/                    # Core source code
│   ├── data_loader.py
│   ├── feature_extractor.py
│   ├── model_builder.py
│   ├── inference_engine.py
│   └── anomaly_detector.py
├── scripts/                # Utility scripts
│   ├── create_sample_data.py
│   ├── preprocess_audio.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── test_inference.py
├── raspberry_pi/          # Raspberry Pi deployment
├── android/               # Android app
├── tests/                 # Unit tests
├── config.yaml           # Configuration
└── requirements.txt      # Dependencies
```

## Model Architecture

The system uses a CRNN (Convolutional Recurrent Neural Network) architecture:
- Conv2D layers for spatial feature extraction
- LSTM layer for temporal pattern recognition
- Dense layers for classification
- Optimized with INT8 quantization for edge deployment

## Performance

- **Accuracy**: 91.2% on test set
- **Model Size**: 567KB (quantized)
- **Inference Time**: 
  - Raspberry Pi 4: 34ms
  - Android (mid-range): 23ms
  - ESP32-S3: 245ms

## Datasets

This project uses publicly available respiratory sound datasets:

1. **ICBHI 2017 Respiratory Sound Database**
   - Source: [BHIC Challenge](https://bhichallenge.med.auth.gr/)
   - 920 recordings from 126 patients

2. **COUGHVID Dataset**
   - Source: [Zenodo](https://zenodo.org/record/4498364)
   - 25,000+ crowdsourced cough recordings
   - License: CC BY 4.0

3. **Coswara Dataset**
   - Source: [GitHub](https://github.com/iiscleap/Coswara-Data)
   - Breathing, cough, and voice samples
   - License: CC BY 4.0

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Benchmark performance:
```bash
python scripts/benchmark_edge.py
```

## Configuration

Edit `config.yaml` to customize:
- Audio processing parameters
- Feature extraction settings
- Model architecture
- Training hyperparameters
- Deployment options

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- ICBHI Challenge organizers
- EPFL team for COUGHVID dataset
- IISc Bangalore for Coswara dataset
- Edge Impulse platform

## Contact

For questions or issues, please open a GitHub issue.
