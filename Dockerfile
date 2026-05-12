FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/local/nltk_data

# Set working directory
WORKDIR /app

# Install only required system dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Download required NLTK resources
RUN python -m nltk.downloader -d /usr/local/nltk_data \
    stopwords \
    wordnet \
    omw-1.4

# Copy Flask app
COPY flask_app/ /app/

# Create artifact directory
RUN mkdir -p /app/models/artifacts

# Copy model artifacts
COPY models/artifacts/vectorizer.pkl /app/models/artifacts/
COPY models/artifacts/label_encoder.pkl /app/models/artifacts/

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]