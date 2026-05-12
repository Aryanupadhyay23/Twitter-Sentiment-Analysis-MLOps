FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/local/nltk_data

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Download required NLTK resources
RUN python -m nltk.downloader -d /usr/local/nltk_data \
    stopwords \
    wordnet \
    omw-1.4

# Copy Flask application
COPY flask_app/ /app/

# Create artifact directory
RUN mkdir -p /app/models/artifacts

# Copy model artifacts
COPY models/artifacts/vectorizer.pkl /app/models/artifacts/
COPY models/artifacts/label_encoder.pkl /app/models/artifacts/

# Expose application port
EXPOSE 5000

# Run production server using Gunicorn
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]