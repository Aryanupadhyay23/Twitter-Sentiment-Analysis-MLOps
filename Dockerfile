FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Download required NLTK resources
RUN python -m nltk.downloader \
    stopwords \
    wordnet \
    omw-1.4

# Copy Flask app
COPY flask_app/ /app/

# Create artifact directory
RUN mkdir -p /app/models/artifacts

# Copy artifacts with SAME paths used in code
COPY models/artifacts/vectorizer.pkl /app/models/artifacts/vectorizer.pkl
COPY models/artifacts/label_encoder.pkl /app/models/artifacts/label_encoder.pkl

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]