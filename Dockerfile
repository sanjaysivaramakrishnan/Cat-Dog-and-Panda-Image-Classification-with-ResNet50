# Use Python 3.9 slim image for better compatibility
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose port 7860 (Hugging Face Spaces standard)
EXPOSE 7860

# Health check for container monitoring
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run the Streamlit application with secure options for Spaces iframe
CMD [
  "streamlit", "run", "app.py",
  "--server.port=7860",
  "--server.address=0.0.0.0",
  "--server.enableXsrfProtection=false",
  "--server.enableCORS=false",
  "--server.maxUploadSize=50"
]