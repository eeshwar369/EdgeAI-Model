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
    <title>EdgeSense - Respiratory Disease Detection</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { 
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .upload-section {
            background: #f8f9fa;
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 30px 0;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-section:hover {
            background: #e9ecef;
            border-color: #764ba2;
        }
        .upload-section.dragover {
            background: #e9ecef;
            border-color: #764ba2;
            transform: scale(1.02);
        }
        input[type="file"] {
            display: none;
        }
        .upload-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 20px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .result.show {
            display: block;
            animation: slideIn 0.5s;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .prediction {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .confidence {
            font-size: 1.2em;
            color: #666;
        }
        .risk-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        .risk-low { background: #d4edda; color: #155724; }
        .risk-medium { background: #fff3cd; color: #856404; }
        .risk-high { background: #f8d7da; color: #721c24; }
        .probabilities {
            margin-top: 20px;
        }
        .prob-bar {
            margin: 10px 0;
        }
        .prob-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        .prob-fill {
            height: 25px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px;
            transition: width 0.5s;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .info-section {
            margin-top: 40px;
            padding-top: 40px;
            border-top: 1px solid #dee2e6;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .info-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .info-card h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        .error.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 EdgeSense</h1>
        <p class="subtitle">Respiratory Disease Detection using Machine Learning</p>
        
        <div class="upload-section" id="uploadSection" onclick="document.getElementById('audioFile').click()">
            <div class="upload-icon">📁</div>
            <h3>Upload Audio File</h3>
            <p>Click to select or drag and drop</p>
            <p style="color: #999; font-size: 0.9em; margin-top: 10px;">Supported: WAV, MP3 (Max 10MB)</p>
            <input type="file" id="audioFile" accept="audio/*" onchange="handleFileSelect(event)">
        </div>
        
        <div style="text-align: center;">
            <button class="btn" id="predictBtn" onclick="predictAudio()" disabled>Analyze Audio</button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px;">Analyzing audio...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result" id="result">
            <h2>Results</h2>
            <div class="prediction" id="prediction"></div>
            <div class="confidence" id="confidence"></div>
            <div id="riskBadge"></div>
            <div class="probabilities" id="probabilities"></div>
            <p style="color: #999; margin-top: 20px; font-size: 0.9em;">
                Inference time: <span id="inferenceTime"></span>ms
            </p>
        </div>
        
        <div class="info-section">
            <h2>About EdgeSense</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h3>91.2%</h3>
                    <p>Accuracy</p>
                </div>
                <div class="info-card">
                    <h3>567KB</h3>
                    <p>Model Size</p>
                </div>
                <div class="info-card">
                    <h3>~34ms</h3>
                    <p>Inference Time</p>
                </div>
                <div class="info-card">
                    <h3>7 Classes</h3>
                    <p>Conditions</p>
                </div>
            </div>
            
            <h3 style="margin-top: 30px;">Supported Conditions</h3>
            <ul style="margin-top: 10px; line-height: 2;">
                <li>Normal Breathing</li>
                <li>Asthma</li>
                <li>COPD (Chronic Obstructive Pulmonary Disease)</li>
                <li>Pneumonia</li>
                <li>Bronchitis</li>
                <li>Tuberculosis</li>
                <li>Long-COVID</li>
            </ul>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        // Drag and drop
        const uploadSection = document.getElementById('uploadSection');
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('audioFile').files = files;
                handleFileSelect({ target: { files: files } });
            }
        });
        
        function handleFileSelect(event) {
            selectedFile = event.target.files[0];
            if (selectedFile) {
                document.getElementById('predictBtn').disabled = false;
                uploadSection.querySelector('h3').textContent = 'File Selected: ' + selectedFile.name;
            }
        }
        
        async function predictAudio() {
            if (!selectedFile) return;
            
            // Hide previous results and errors
            document.getElementById('result').classList.remove('show');
            document.getElementById('error').classList.remove('show');
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('predictBtn').disabled = true;
            
            const formData = new FormData();
            formData.append('audio', selectedFile);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    showError(data.error || 'An error occurred');
                }
            } catch (error) {
                showError('Failed to connect to server: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('predictBtn').disabled = false;
            }
        }
        
        function displayResults(data) {
            document.getElementById('prediction').textContent = data.prediction;
            document.getElementById('confidence').textContent = 
                `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            document.getElementById('inferenceTime').textContent = 
                data.inference_time_ms.toFixed(1);
            
            // Risk badge
            const riskLevel = data.risk_level.toLowerCase();
            const riskBadge = document.getElementById('riskBadge');
            riskBadge.innerHTML = 
                `<span class="risk-badge risk-${riskLevel}">Risk: ${data.risk_level}</span>`;
            
            // Probabilities
            const probsDiv = document.getElementById('probabilities');
            probsDiv.innerHTML = '<h3>All Probabilities</h3>';
            
            for (const [label, prob] of Object.entries(data.probabilities)) {
                const percentage = (prob * 100).toFixed(1);
                probsDiv.innerHTML += `
                    <div class="prob-bar">
                        <div class="prob-label">
                            <span>${label}</span>
                            <span>${percentage}%</span>
                        </div>
                        <div style="background: #e9ecef; border-radius: 5px;">
                            <div class="prob-fill" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            }
            
            document.getElementById('result').classList.add('show');
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = 'Error: ' + message;
            errorDiv.classList.add('show');
        }
    </script>
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
