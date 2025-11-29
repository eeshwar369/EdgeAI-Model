FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs samples

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=models/quantized_model.tflite
ENV PORT=8000

# Run API server
CMD ["python", "api_server.py"]
