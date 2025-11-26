# EdgeSense Makefile
# Convenient commands for common tasks

.PHONY: help install setup download preprocess train evaluate test deploy clean

help:
	@echo "EdgeSense - Makefile Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make setup         - Complete setup (install + create directories)"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make download      - Show dataset download instructions"
	@echo "  make preprocess    - Preprocess audio data"
	@echo ""
	@echo "Model Training:"
	@echo "  make train         - Train the model"
	@echo "  make evaluate      - Evaluate trained model"
	@echo "  make test          - Run unit tests"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-rpi    - Deploy to Raspberry Pi"
	@echo "  make benchmark     - Benchmark model performance"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Clean generated files"
	@echo "  make notebook      - Start Jupyter notebook"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Installation complete!"

setup: install
	@echo "Creating directories..."
	mkdir -p data/raw data/processed data/features
	mkdir -p models results logs
	mkdir -p samples/normal samples/asthma samples/copd
	mkdir -p samples/pneumonia samples/bronchitis samples/tuberculosis samples/long_covid
	@echo "Setup complete!"

download:
	@echo "Running dataset download helper..."
	python scripts/download_datasets.py

preprocess:
	@echo "Preprocessing audio data..."
	python scripts/preprocess_audio.py

train:
	@echo "Training model..."
	python scripts/train_model.py

evaluate:
	@echo "Evaluating model..."
	python scripts/evaluate_model.py

test:
	@echo "Running unit tests..."
	pytest tests/ -v

test-inference:
	@echo "Testing inference on sample audio..."
	python scripts/test_inference.py --audio samples/normal/normal_001.wav

benchmark:
	@echo "Benchmarking model performance..."
	python scripts/benchmark_edge.py --model models/quantized_model.tflite

deploy-rpi:
	@echo "Deploying to Raspberry Pi..."
	@echo "Make sure Raspberry Pi is connected and accessible"
	scp -r raspberry_pi/ pi@raspberrypi.local:~/edgesense/
	scp models/quantized_model.tflite pi@raspberrypi.local:~/edgesense/models/
	@echo "Deployment complete! SSH into Raspberry Pi and run:"
	@echo "  cd ~/edgesense/raspberry_pi"
	@echo "  ./deploy.sh"

upload-ei:
	@echo "Uploading data to Edge Impulse..."
	python scripts/upload_to_edge_impulse.py

notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ src/__pycache__ scripts/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf build dist
	rm -rf logs/*.log
	@echo "Clean complete!"

clean-all: clean
	@echo "Removing processed data and models..."
	rm -rf data/processed/*
	rm -rf models/*.h5 models/*.tflite
	rm -rf results/*
	@echo "Deep clean complete!"

# Development commands
lint:
	@echo "Running linter..."
	flake8 src/ scripts/ --max-line-length=100

format:
	@echo "Formatting code..."
	black src/ scripts/

# Quick pipeline
pipeline: preprocess train evaluate
	@echo "Complete pipeline finished!"

# All-in-one command
all: setup download preprocess train evaluate
	@echo "Complete workflow finished!"
