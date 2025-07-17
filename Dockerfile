# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
  apt-get install -y ffmpeg git && \
  rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Set HuggingFace cache directory and copy pre-downloaded models
ENV TRANSFORMERS_CACHE=/app/hf_models

# Create the hf_models directory (models will be downloaded at runtime)
RUN mkdir -p /app/hf_models

# Copy the rest of the application code
COPY . .

# Expose port (Flask default is 5000, but Azure uses 8000 by default)
EXPOSE 8000

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Use Gunicorn as the production server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "300", "app:app"]