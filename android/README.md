# EdgeSense Android App

Real-time respiratory disease detection on Android devices.

## Features

- 🎤 Real-time microphone capture
- 🧠 On-device TensorFlow Lite inference
- 📊 Visual probability graphs
- 🚨 Alert system for high-risk patterns
- 📈 Historical trend tracking

## Requirements

- Android Studio Arctic Fox or later
- Android SDK 24+ (Android 7.0+)
- Device with microphone

## Setup

1. Open this folder in Android Studio
2. Sync Gradle dependencies
3. Copy `quantized_model.tflite` to `app/src/main/assets/`
4. Build and run

## Project Structure

```
android/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/edgesense/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── AudioRecorder.kt
│   │   │   │   ├── FeatureExtractor.kt
│   │   │   │   └── ModelInference.kt
│   │   │   ├── res/
│   │   │   │   ├── layout/
│   │   │   │   │   └── activity_main.xml
│   │   │   │   └── values/
│   │   │   └── assets/
│   │   │       └── quantized_model.tflite
│   │   └── AndroidManifest.xml
│   └── build.gradle
└── build.gradle
```

## Permissions

The app requires microphone permission:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

## Usage

1. Launch app
2. Grant microphone permission
3. Tap "Start Detection"
4. Breathe or cough near microphone
5. View real-time predictions

## Building APK

```bash
./gradlew assembleRelease
```

APK location: `app/build/outputs/apk/release/app-release.apk`

## Deployment

### Via USB

```bash
adb install app/build/outputs/apk/release/app-release.apk
```

### Via Edge Impulse

Use Edge Impulse Android deployment for automatic integration.

## Performance

- Inference time: ~20-30ms on mid-range devices
- RAM usage: ~40MB
- Battery impact: Low (optimized for continuous monitoring)

## Troubleshooting

### Audio Recording Issues

Ensure microphone permission is granted in Settings > Apps > EdgeSense > Permissions

### Model Loading Errors

Verify `quantized_model.tflite` is in `app/src/main/assets/`

### Slow Inference

Try reducing audio buffer size or using a more powerful device

## License

MIT License - see LICENSE file
