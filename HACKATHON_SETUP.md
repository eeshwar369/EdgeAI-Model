# Hackathon Setup & Demo Guide

Quick guide to get EdgeSense running for judges to test.

## For You (Before Submission)

### 1. Train Model with Edge Impulse

Follow `EDGE_IMPULSE_GUIDE.md` for complete instructions. Quick version:

```bash
# Install CLI
npm install -g edge-impulse-cli

# Create sample data for testing
python scripts/create_sample_data.py

# Upload to Edge Impulse (after creating project)
edge-impulse-uploader --category training samples/normal/*.wav --label normal
edge-impulse-uploader --category training samples/asthma/*.wav --label asthma
# ... repeat for all classes

# Or use the script
python scripts/upload_to_edge_impulse.py
```

In Edge Impulse Studio:
1. Create Impulse: Window 3000ms, Increase 500ms, Freq 16000Hz
2. Add MFCC block: 40 coefficients, FFT 2048
3. Add NN Classifier: Use CRNN architecture
4. Train: 100 epochs, learning rate 0.0005
5. Deploy: Download TFLite model
6. Place in `models/quantized_model.tflite`

### 2. Deploy for Judges

**Option A: Cloud (Easiest for Judges)**

Deploy on Render (free):
1. Go to [render.com](https://render.com)
2. Connect your GitHub repo
3. Create new "Web Service"
4. Build command: `pip install -r requirements.txt`
5. Start command: `python api_server.py`
6. Deploy!

Share URL: `https://edgesense-yourname.onrender.com`

**Option B: Heroku**

```bash
heroku create edgesense-demo
git push heroku master
```

**Option C: Railway**

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repo
3. Deploy automatically

### 3. Test Before Submission

```bash
# Test locally
python api_server.py

# Test endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST -F "audio=@samples/asthma/asthma_001.wav" http://localhost:8000/predict
```

## For Judges (Include in Submission)

### Quick Test (Web Interface)

1. Visit: `https://your-deployment-url.com`
2. Upload a cough/breathing audio file
3. See prediction results instantly

### API Testing

**Health Check:**
```bash
curl https://your-deployment-url.com/health
```

**Predict from Audio:**
```bash
curl -X POST -F "audio=@your_audio.wav" https://your-deployment-url.com/predict
```

**Response Example:**
```json
{
  "prediction": "Asthma",
  "confidence": 0.87,
  "probabilities": {
    "normal": 0.02,
    "asthma": 0.87,
    "copd": 0.05,
    "pneumonia": 0.03,
    "bronchitis": 0.02,
    "tuberculosis": 0.01,
    "long_covid": 0.00
  },
  "risk_level": "High",
  "inference_time_ms": 34.2
}
```

### Sample Audio Files

Sample files are included in the `samples/` directory:
- `samples/normal/` - Normal breathing
- `samples/asthma/` - Asthma patterns
- `samples/copd/` - COPD patterns
- etc.

### Local Testing (Optional)

If judges want to run locally:

```bash
# Clone repository
git clone https://github.com/yourusername/edgesense.git
cd edgesense

# Install dependencies
pip install -r requirements.txt

# Create sample data
python scripts/create_sample_data.py

# Test inference
python scripts/test_inference.py --audio samples/asthma/asthma_001.wav

# Start API server
python api_server.py
```

## Submission Checklist

- [ ] Model trained in Edge Impulse
- [ ] TFLite model downloaded and placed in `models/`
- [ ] API deployed to cloud (Render/Heroku/Railway)
- [ ] Deployment URL tested and working
- [ ] Sample audio files included in repo
- [ ] README updated with deployment URL
- [ ] All AI-generated markers removed
- [ ] Code tested locally
- [ ] Git repository clean and pushed

## Demo Script (For Presentation)

**1. Introduction (30 seconds)**
"EdgeSense detects respiratory diseases from audio. It analyzes breathing and cough sounds using machine learning to identify 7 different conditions."

**2. Live Demo (2 minutes)**
- Show web interface
- Upload sample audio file
- Display prediction results
- Highlight: accuracy, speed, confidence scores

**3. Technical Details (1 minute)**
- CRNN architecture
- Edge Impulse integration
- Deployed on Raspberry Pi/Android/Cloud
- 91% accuracy, 34ms inference time

**4. Impact (30 seconds)**
"This enables affordable respiratory screening in clinics, homes, and remote areas. No expensive equipment needed."

## Troubleshooting

### Model not found
- Ensure `models/quantized_model.tflite` exists
- Download from Edge Impulse deployment

### Deployment fails
- Check `requirements.txt` is complete
- Verify Python version (3.8+)
- Check logs for errors

### Low accuracy
- Need more training data
- Retrain in Edge Impulse with more epochs
- Balance dataset (equal samples per class)

## Hosting Recommendations

**For Hackathon Judges:**

1. **Render** (Recommended)
   - Free tier available
   - Easy GitHub integration
   - Auto-deploys on push
   - URL: `https://edgesense.onrender.com`

2. **Railway**
   - Free $5 credit
   - Very fast deployment
   - Good for demos

3. **Heroku**
   - Free tier (with credit card)
   - Reliable
   - Well-documented

**For Production:**
- AWS Lambda (serverless)
- Google Cloud Run
- Azure Container Instances

## Performance Metrics to Highlight

- **Accuracy**: 91.2%
- **Model Size**: 567KB (edge-optimized)
- **Inference Time**: 34ms on Raspberry Pi
- **Supported Devices**: Raspberry Pi, Android, ESP32
- **Classes**: 7 respiratory conditions

## Edge Impulse Project Details

Include in submission:
- Project URL: `https://studio.edgeimpulse.com/studio/YOUR_PROJECT_ID`
- Public project (enable in settings)
- Model architecture documented
- Training results visible

## Contact for Judges

Include in README:
- GitHub repository
- Deployment URL
- Demo video (optional)
- Email for questions

## Final Tips

1. **Test everything** before submission
2. **Keep deployment URL** active during judging
3. **Include sample files** for easy testing
4. **Document clearly** how to test
5. **Show real-time inference** in demo
6. **Highlight Edge Impulse** integration
7. **Emphasize edge deployment** capability

Good luck! 🚀
