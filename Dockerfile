FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0","--server.enableXsrfProtection=false","--server.enableCORS=false","--server.maxUploadSize=50"]