# Quick Start Guide

Get EdgeSense running in 10 minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
npm install -g edge-impulse-cli
```

## Step 2: Create Sample Data

```bash
python scripts/create_sample_data.py
```

This creates synthetic audio samples for testing.

## Step 3: Test Locally

```bash
python scripts/test_inference.py --audio samples/asthma/asthma_001.wav
```

## Step 4: Start Web Interface

```bash
python api_server.py
```

Visit `http://localhost:8000` to test the web interface.

## Step 5: Train with Edge Impulse (Optional)

See `EDGE_IMPULSE_GUIDE.md` for complete instructions.

Quick version:
1. Create project at edgeimpulse.com
2. Upload audio samples
3. Configure MFCC (40 coefficients, FFT 2048)
4. Train CRNN model (100 epochs)
5. Download TFLite model
6. Place in `models/quantized_model.tflite`

## Step 6: Deploy

See `DEPLOYMENT.md` for hosting options.

Quickest: Deploy to Render.com (free)
1. Connect GitHub repo
2. Deploy automatically
3. Share URL with judges

## Need Help?

- Edge Impulse setup: `EDGE_IMPULSE_GUIDE.md`
- Deployment options: `DEPLOYMENT.md`
- Hackathon prep: `HACKATHON_SETUP.md`
- Full documentation: `README.md`
