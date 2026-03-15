# AI People Reader

A web application for Khun Rungnapa that analyzes video to detect body language and motion types. It shows a skeleton overlay on the video and a motion confidence chart over time, based on Laban movement concepts (Advancing, Retreating, Enclosing, Spreading, Directing, Indirecting).

---

## How It Works (Overview)

1. **You upload a video** (MP4, MKV, MOV, or AVI).
2. **The system extracts the skeleton** from each frame using Google's MediaPipe (pose detection).
3. **Motion is analyzed** every second: the algorithm detects which motion types appear and how strongly.
4. **You see** the skeleton drawn on the video and a chart showing motion confidence over time.

---

## How the Skeleton Is Obtained

- **Technology:** Google MediaPipe Pose Landmarker
- **Process:** The video is processed frame by frame (30 frames per second). For each frame, MediaPipe finds the person and outputs 33 body landmarks (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.) with x, y, z coordinates.
- **Output:** A list of frames, each with a timestamp and landmark positions. This is the "skeleton" data.
- **Storage:** Skeleton data is saved in a cache so the same video does not need to be processed again. Only the skeleton coordinates are stored—not the analysis results.

---

## How Motion Is Detected

The algorithm looks at how body parts move between consecutive frames and classifies motion into six types:

| Motion Type | What It Detects |
|-------------|-----------------|
| **Advancing** | Moving forward: hands, feet, hips, or head moving toward the camera |
| **Retreating** | Moving backward: hands moving away from the camera |
| **Enclosing** | Arms folding inward: wrists close together or contracting |
| **Spreading** | Arms opening outward: wrists far apart or expanding |
| **Directing** | Pointing/reaching: hands forward, sustained movement toward a target |
| **Indirecting** | Curved or lateral movement: hands to the sides or moving sideways |

**Body parts used:**
- **Arms:** Wrists, shoulders (for expansion, velocity, direction)
- **Body:** Hips, ankles, nose (for stepping and body movement)

**Per second:** The algorithm counts how many frame-pairs in each 1-second window show each motion type. Confidence is the percentage of frame-pairs in that second where the motion appeared (multiple motions can occur at once—Integrated Movement).

---

## How the Motion Plot Is Made

- **Data source:** `motion_per_second` from the backend—one row per second with motion types and confidence values.
- **Chart:** A line chart (Recharts) with time (seconds) on the X-axis and confidence (0–100) on the Y-axis. Each motion type has its own colored line.
- **Playhead:** A red line moves with the video so you can see which second you are watching.
- **Confidence:** Each motion type has its own confidence score per second. Values do not add to 100—multiple motions can be present at the same time.

---

## Skeleton Cache (For Debugging)

- **Location:** `backend/debug_skeleton_cache/`
- **What is stored:** Only skeleton coordinates (frames with landmarks). No analysis.
- **Cache key:** A hash of the video file content. Same video → same cache file.
- **When used:** If you upload a video that was processed before, the system loads the cached skeleton and skips MediaPipe. Analysis (motion confidence) is always recomputed from the skeleton.
- **Force reprocess:** Add `?force_reprocess=true` to the upload URL to bypass the cache and run MediaPipe again.
- **Purpose:** Speeds up testing when you change the motion algorithm. The cache is for development; you can delete it or leave it. It is in `.gitignore` so it is not pushed to the repo.

---

## Running Locally

### Prerequisites

- **Node.js** (for the frontend)
- **Python 3.10, 3.11, or 3.12** (for the backend). **Do not use Python 3.13** — MediaPipe crashes on macOS ARM due to protobuf/PyArrow conflicts. Use `pyenv install 3.11` or install Python 3.11 from [python.org](https://www.python.org/downloads/).
- **FFmpeg** (for video conversion). Install: `winget install ffmpeg` (Windows) or `brew install ffmpeg` (Mac)

### Backend

Use **Python 3.10, 3.11, or 3.12** (not 3.13). Create a virtual environment, then run:

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000). Upload a video to test.

### Backend crashes on startup (macOS)

If you see `mutex lock failed: Invalid argument` or `libc++abi: terminating`, you are likely using **Python 3.13**. MediaPipe does not support Python 3.13 and conflicts with protobuf/PyArrow on macOS ARM.

**Fix:** Use Python 3.11 (or 3.10/3.12):

```bash
# With pyenv:
pyenv install 3.11
pyenv local 3.11
cd backend
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Hosting on Render

Render has two parts: a **Web Service** (backend) and a **Static Site** (frontend). Deploy them separately and connect them.

### Step 1: Deploy the Backend (Web Service)

**Option A: Docker (recommended, includes FFmpeg)**

1. Go to [Render Dashboard](https://dashboard.render.com/) → **New** → **Web Service**.
2. Connect your GitHub repo (the one with this project). Use a **private** repo if you prefer—only you need access.
3. Configure:
   - **Name:** `ai-people-reader-api` (or any name)
   - **Region:** Choose closest to you (e.g. Singapore)
   - **Root Directory:** `backend`
   - **Runtime:** **Docker** (required for FFmpeg)
   - **Build:** Render uses the `Dockerfile` in `backend/`. No extra build command.
   - **Start:** Handled by the Dockerfile.

**Option B: Python + React (project root, single URL)**

If using the project root (no Root Directory set), serve both frontend and backend from one URL:
- **Build Command:** `bash build_for_render.sh`
- **Start Command:** `python run_backend.py`

This builds the React app and serves it from the backend at `/`.
4. **Environment Variables** (optional for now):
   - `ALLOWED_ORIGINS` = `https://your-frontend.onrender.com` (add this after you create the static site)
5. Click **Create Web Service**. Wait for the first deploy.
6. Copy the service URL, e.g. `https://ai-people-reader-api.onrender.com`.

### Step 2: Deploy the Frontend (Static Site)

1. In Render Dashboard → **New** → **Static Site**.
2. Connect the same GitHub repo.
3. Configure:
   - **Name:** `ai-people-reader` (or any name)
   - **Root Directory:** leave blank (project root)
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `build`
4. **Environment Variables:**
   - `REACT_APP_API_URL` = `https://ai-people-reader-api.onrender.com` (your backend URL from Step 1)
5. Click **Create Static Site**. Wait for the first deploy.
6. Copy the site URL, e.g. `https://ai-people-reader.onrender.com`.

### Step 3: Connect Backend and Frontend

1. Go back to your **Web Service** (backend) → **Environment**.
2. Add: `ALLOWED_ORIGINS` = `https://ai-people-reader.onrender.com` (your static site URL).
3. Save. Render will redeploy the backend.

Now the frontend can talk to the backend. Open your static site URL and upload a video.

---

## Tips for Render

### Free Tier Limits

- **Web Service:** Spins down after ~15 minutes of no traffic. First request after that may take 30–60 seconds to wake up.
- **Static Site:** Stays up; no spin-down.
- **Bandwidth:** Check your plan. Video uploads use bandwidth.

### Making It Work Better

1. **Use MP4 videos** when possible. MP4 does not need FFmpeg conversion; other formats do. This reduces backend load.
2. **Shorter videos** process faster. Very long videos may hit timeouts.
3. **Upgrade plan:** If you need faster processing or no spin-down, consider a paid plan.
4. **GPU:** Render’s standard plans do not include GPU. MediaPipe runs on CPU. For this app, CPU is usually enough. GPU would require a different host (e.g. AWS, GCP).

### If Something Fails

- **Backend fails to start:** Check the Render logs. Common issues: missing `ALLOWED_ORIGINS`, wrong root directory.
- **Frontend cannot reach backend:** Ensure `REACT_APP_API_URL` is correct and `ALLOWED_ORIGINS` includes your frontend URL.
- **Video upload fails:** Ensure the backend is awake (send a request first). Check file size limits.
- **CORS errors:** Add your frontend URL to `ALLOWED_ORIGINS` (comma-separated if you have more than one).

---

## Project Structure

```
ai-people-reader-react/
├── backend/
│   ├── main.py          # FastAPI server, upload, skeleton extraction
│   ├── analysis.py     # Motion detection algorithm
│   ├── requirements.txt
│   ├── Dockerfile      # For Render (includes FFmpeg)
│   └── debug_skeleton_cache/   # Cached skeletons (local only)
├── src/
│   └── App.js          # React app, video player, chart
├── public/
└── README.md
```

---

## For Khun Rungnapa

This app is for your research and analysis. The motion types (Advancing, Retreating, Enclosing, Spreading, Directing, Indirecting) follow the Motion Contrast Chart and Laban concepts. The algorithm was tuned using your Authority Coding Sheets as ground truth.

- **Private repo:** Only you (and people you invite) can see the code.
- **Render:** Free tier is enough to try it. You can upgrade later if needed.
- **Support:** If you need changes or have questions, share them with your developer.
