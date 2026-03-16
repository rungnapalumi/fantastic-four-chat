"""
Video Pose Overlay - FastAPI Backend
Accepts video uploads, normalizes to web MP4, extracts pose landmarks via MediaPipe.
"""

import hashlib
import json
import os
import shutil
import subprocess
import uuid
import tempfile
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import pose_landmarker

from analysis import run_analysis

# MediaPipe pose landmark names (index 0–32)
POSE_LANDMARK_NAMES = [e.name for e in pose_landmarker.PoseLandmark]

app = FastAPI(title="Video Pose Overlay API")

# CORS: localhost for dev; set ALLOWED_ORIGINS env for production (e.g. https://your-site.onrender.com)
_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for processed videos
UPLOAD_DIR = Path(tempfile.gettempdir()) / "pose_overlay_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Debug cache: skip MediaPipe when skeleton data exists (for chart debugging)
SKELETON_CACHE_DIR = Path(__file__).resolve().parent / "debug_skeleton_cache"
SKELETON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}


def _find_ffmpeg() -> str:
    """Find ffmpeg executable: PATH, FFMPEG_PATH env, or common Windows locations."""
    path = shutil.which("ffmpeg")
    if path:
        return path

    if os.environ.get("FFMPEG_PATH"):
        p = Path(os.environ["FFMPEG_PATH"])
        if p.is_file():
            return str(p)
        exe = p / "ffmpeg.exe" if p.is_dir() else p.parent / "ffmpeg.exe"
        if exe.exists():
            return str(exe)

    # WinGet / winget install location
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    for p in local.glob("Microsoft/WinGet/Packages/*/ffmpeg*/bin/ffmpeg.exe"):
        if p.exists():
            return str(p)

    raise HTTPException(
        status_code=503,
        detail="FFmpeg not found. Install via: winget install ffmpeg. Then add its bin folder to PATH, or set FFMPEG_PATH.",
    )


def normalize_video(input_path: Path, output_path: Path) -> Path:
    """Convert video to web-compatible H.264/AAC MP4 using ffmpeg."""
    ffmpeg_exe = _find_ffmpeg()
    cmd = [
        ffmpeg_exe,
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-y",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr or result.stdout}")
        return output_path
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="FFmpeg timed out")
    except (FileNotFoundError, OSError) as e:
        raise HTTPException(status_code=500, detail=str(e))


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def _get_model_path() -> Path:
    """Download and cache the pose landmarker model."""
    cache_dir = Path(tempfile.gettempdir()) / "pose_overlay_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "pose_landmarker_lite.task"
    if not model_path.exists():
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def _video_hash(contents: bytes) -> str:
    """SHA256 hash of video content for cache key."""
    return hashlib.sha256(contents).hexdigest()[:16]


def _load_skeleton_cache(cache_key: str) -> tuple[list[dict], dict] | None:
    """Load skeleton frames from cache, compute analysis on load. Returns (frames, analysis) or None."""
    cache_path = SKELETON_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        if not frames:
            return None
        analysis = run_analysis(frames) if frames else {}
        return (frames, analysis)
    except (json.JSONDecodeError, OSError):
        return None


def _save_skeleton_cache(cache_key: str, frames: list[dict]) -> None:
    """Save only skeleton coords time series. Analysis is computed on load (fast)."""
    cache_path = SKELETON_CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"frames": frames}, f, indent=0)
    except OSError:
        pass


def extract_pose_landmarks(video_path: Path) -> list[dict]:
    """Process 5 frames per second with MediaPipe PoseLandmarker, return landmarks per frame."""
    model_path = _get_model_path()
    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(model_path),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,  # Avoid NSOpenGLPixelFormat error on macOS
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval_ms = int(1000 / fps) if fps > 0 else 33

    # Sample 30 frames per second for smooth skeleton overlay
    step = max(1, int(fps / 30))
    frame_indices = list(range(0, total_frames, step))
    frames = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = frame_idx * frame_interval_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks_list = []
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                for i, lm in enumerate(result.pose_landmarks[0]):
                    name = POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"LANDMARK_{i}"
                    landmarks_list.append({
                        "name": name,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                    })
            while len(landmarks_list) < 33:
                i = len(landmarks_list)
                name = POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"LANDMARK_{i}"
                landmarks_list.append({"name": name, "x": 0, "y": 0, "z": 0})

            frames.append({
                "frame_index": frame_idx,
                "timestamp": frame_idx / fps,
                "landmarks": landmarks_list,
            })

    cap.release()
    return frames


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    force_reprocess: bool = Query(False, description="Bypass skeleton cache and reprocess with MediaPipe"),
):
    """Accept video file, normalize to MP4, extract pose landmarks, return JSON with frames and video URL."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    raw_path = job_dir / f"raw{ext}"
    mp4_path = job_dir / "video.mp4"

    try:
        contents = await file.read()
        with open(raw_path, "wb") as f:
            f.write(contents)

        # For MP4, copy directly (already web-compatible). For others, normalize with FFmpeg.
        if ext == ".mp4":
            shutil.copy2(raw_path, mp4_path)
        else:
            normalize_video(raw_path, mp4_path)

        if not mp4_path.exists() or mp4_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Video conversion produced empty file")

        cache_key = _video_hash(contents)
        cached = _load_skeleton_cache(cache_key) if not force_reprocess else None

        if cached is not None:
            frames, analysis = cached
            print(f"[cache] HIT: {cache_key}.json", flush=True)
        else:
            print(f"[cache] MISS: {cache_key}.json (no matching cache file)", flush=True)
            # Extract pose landmarks
            frames = extract_pose_landmarks(mp4_path)
            # Run pose analysis
            analysis = run_analysis(frames) if frames else {}
            # Save to cache for next time
            _save_skeleton_cache(cache_key, frames)

        # Clean up raw file
        raw_path.unlink(missing_ok=True)

        video_url = f"/video/{job_id}"
        return {
            "video_url": video_url,
            "frames": frames,
            "analysis": analysis,
            "job_id": job_id,
            "from_cache": cached is not None,
            "cache_key": cache_key,
        }
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/video/{job_id}")
def serve_video(job_id: str):
    """Stream the processed video with proper headers for browser playback."""
    mp4_path = UPLOAD_DIR / job_id / "video.mp4"
    if not mp4_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        mp4_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve React frontend when deployed together (backend/static from build)
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    _static_assets = _STATIC_DIR / "static"
    if _static_assets.exists():
        app.mount("/static", StaticFiles(directory=_static_assets), name="static")

    @app.get("/")
    def serve_app():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/{path:path}")
    def serve_app_path(path: str):
        """Catch-all for React client-side routing."""
        full_path = _STATIC_DIR / path
        if full_path.exists() and full_path.is_file():
            return FileResponse(full_path)
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
