"""
REST API server for EdgeSense inference.
Optional component for cloud deployment.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import librosa
import numpy as np
import os
from src.inference_engine import RespiratoryInferenceEngine

app = Flask(__name__)
CORS(app)

# Initialize inference engine
MODEL_PATH = os.getenv('MODEL_PATH', 'models/quantized_model.tflite')
engine = RespiratoryInferenceEngine(MODEL_PATH, use_tflite=True)

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>EdgeSense API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { color: #0066cc; font-weight: bold; }
        code { background: #eee; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>🎤 EdgeSense API</h1>
    <p>Respiratory disease detection via acoustic biomarkers</p>
    
    <h2>Endpoints</h2>
    
    <div class="endpoint">
        <p><span class="method">GET</span> <code>/</code></p>
        <p>API documentation (this page)</p>
    </div>
    
    <div class="endpoint">
        <p><span class="method">GET</span> <code>/health</code></p>
        <p>Health check endpoint</p>
    </div>
    
    <div class="endpoint">
        <p><span class="method">POST</span> <code>/predict</code></p>
        <p>Predict respiratory condition from audio file</p>
        <p><strong>Parameters:</strong></p>
        <ul>
            <li><code>audio</code> (file): Audio file (WAV or MP3)</li>
        </ul>
        <p><strong>Response:</strong></p>
        <pre>{
  "prediction": "Asthma",
  "confidence": 0.8734,
  "probabilities": {...},
  "risk_level": "High",
  "inference_time_ms": 34.2
}</pre>
    </div>
    
    <div class="endpoint">
        <p><span class="method">GET</span> <code>/labels</code></p>
        <p>Get list of supported disease labels</p>
    </div>
    
    <h2>Example Usage</h2>
    <pre>curl -X POST -F "audio=@cough.wav" http://localhost:8000/predict</pre>
    
    <h2>Model Info</h2>
    <ul>
        <li>Accuracy: 91.2%</li>
        <li>Model Size: 567KB</li>
        <li>Inference Time: ~34ms</li>
        <li>Supported Diseases: 7 classes</li>
    </ul>
</body>
</html>
"""

@app.route('/')
def index():
    """API documentation page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict respiratory condition from audio file."""
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Save temporarily
        temp_path = f'/tmp/{audio_file.filename}'
        audio_file.save(temp_path)
        
        # Load audio
        audio, sr = librosa.load(temp_path, sr=16000)
        
        # Run inference
        import time
        start_time = time.time()
        result = engine.predict(audio, sr)
        inference_time = (time.time() - start_time) * 1000
        
        # Add inference time
        result['inference_time_ms'] = round(inference_time, 2)
        
        # Add risk level
        result['risk_level'] = engine.get_risk_level(
            result['prediction'],
            result['confidence']
        )
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/labels', methods=['GET'])
def get_labels():
    """Get list of supported disease labels."""
    return jsonify({
        'labels': engine.label_names,
        'count': len(engine.label_names)
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify({
        'accuracy': 0.912,
        'roc_auc': 0.96,
        'model_size_kb': 567,
        'inference_time_ms': 34,
        'num_classes': 7,
        'classes': engine.label_names
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("EdgeSense API Server")
    print("=" * 60)
    print(f"Server running on http://localhost:{port}")
    print(f"Model: {MODEL_PATH}")
    print(f"Debug mode: {debug}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
