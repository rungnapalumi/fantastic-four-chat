# Video Pose Overlay - Backend

FastAPI server that accepts video uploads, normalizes them to web MP4, and extracts pose landmarks via MediaPipe.

## Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and on your PATH

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
python main.py
```

## API

- `POST /upload` - Upload a video (MP4, MKV, MOV, AVI). Returns JSON with `video_url`, `frames` (landmarks per frame), and `job_id`.
- `GET /health` - Health check.
