#!/bin/bash

# EdgeSense Raspberry Pi Deployment Script

echo "======================================"
echo "EdgeSense Raspberry Pi Setup"
echo "======================================"

# Update system
echo "Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
echo "Installing dependencies..."
sudo apt-get install -y python3-pip python3-dev portaudio19-dev
sudo apt-get install -y libatlas-base-dev libhdf5-dev

# Install Python packages
echo "Installing Python packages..."
pip3 install numpy scipy
pip3 install tensorflow-lite-runtime
pip3 install librosa soundfile
pip3 install pyaudio

# Install Edge Impulse CLI
echo "Installing Edge Impulse CLI..."
curl -sL https://deb.nodesource.com/setup_14.x | sudo bash -
sudo apt-get install -y nodejs
sudo npm install -g edge-impulse-linux

# Copy model files
echo "Copying model files..."
mkdir -p ~/edgesense/models
cp ../models/quantized_model.tflite ~/edgesense/models/

# Copy inference script
cp realtime_inference.py ~/edgesense/

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To run real-time inference:"
echo "  cd ~/edgesense"
echo "  python3 realtime_inference.py"
echo ""
echo "To use Edge Impulse runner:"
echo "  edge-impulse-linux-runner"
echo ""
