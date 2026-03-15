# AI People Reader - Backend (for Render with FFmpeg)
FROM python:3.11-slim

# Install FFmpeg for video conversion (MKV, MOV, AVI -> MP4)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT; default 8000 for local
ENV PORT=8000
EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
