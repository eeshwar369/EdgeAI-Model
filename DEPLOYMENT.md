# Deployment Guide

This guide covers deploying EdgeSense for judges and users to test.

## Option 1: Cloud Hosting (Recommended for Judges)

### Deploy on Heroku

1. Create `Procfile`:
```
web: python api_server.py
```

2. Deploy:
```bash
heroku create edgesense-demo
git push heroku main
```

3. Share URL: `https://edgesense-demo.herokuapp.com`

### Deploy on Railway

1. Connect GitHub repository to Railway
2. Set environment variables:
   - `PORT=8000`
   - `MODEL_PATH=models/quantized_model.tflite`
3. Deploy automatically on push

### Deploy on Render

1. Create new Web Service
2. Connect repository
3. Build command: `pip install -r requirements.txt`
4. Start command: `python api_server.py`

## Option 2: Docker Deployment

### Build Image

```bash
docker build -t edgesense .
```

### Run Container

```bash
docker run -p 8000:8000 edgesense
```

### Docker Compose

```bash
docker-compose up
```

## Option 3: Raspberry Pi (Edge Device)

### Setup

```bash
# SSH into Raspberry Pi
ssh pi@raspberrypi.local

# Clone repository
git clone https://github.com/yourusername/edgesense.git
cd edgesense

# Install dependencies
pip3 install -r requirements.txt

# Run deployment script
cd raspberry_pi
./deploy.sh
```

### Run Inference

```bash
python3 realtime_inference.py
```

## Option 4: Android APK

### Build APK

1. Open `android/` in Android Studio
2. Build > Build Bundle(s) / APK(s) > Build APK(s)
3. APK location: `android/app/build/outputs/apk/release/app-release.apk`

### Distribute

- Upload to Google Drive
- Share download link with judges
- Or use Firebase App Distribution

## Testing the Deployment

### API Endpoints

**Health Check:**
```bash
curl https://your-deployment-url.com/health
```

**Predict:**
```bash
curl -X POST -F "audio=@sample.wav" https://your-deployment-url.com/predict
```

### Web Interface

Visit `https://your-deployment-url.com` in browser for interactive testing.

## Monitoring

### Logs

**Heroku:**
```bash
heroku logs --tail
```

**Docker:**
```bash
docker logs -f container_id
```

**Raspberry Pi:**
```bash
tail -f logs/edgesense.log
```

## Troubleshooting

### Issue: Model not found
- Ensure `models/quantized_model.tflite` is included in deployment
- Check `MODEL_PATH` environment variable

### Issue: Out of memory
- Use quantized model (567KB)
- Reduce batch size in config

### Issue: Slow inference
- Enable hardware acceleration (NNAPI on Android)
- Use TFLite instead of full TensorFlow

## Security Considerations

- Add rate limiting for production
- Implement authentication if needed
- Validate audio file uploads
- Set maximum file size limits

## Cost Estimates

- **Heroku**: Free tier available
- **Railway**: $5/month
- **Render**: Free tier available
- **Raspberry Pi**: One-time $50 hardware cost
