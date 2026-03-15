import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Upload, Video, Loader2, Heart } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import './App.css';

// Legacy motion types (original graph)
const LEGACY_MOTION_COLORS = {
  Advancing: '#22c55e',
  Retreating: '#ef4444',
  Enclosing: '#a855f7',
  Spreading: '#3b82f6',
  Directing: '#f59e0b',
  Indirecting: '#06b6d4',
  Neutral: '#64748b',
};

// Second graph: 6 types from Authority Coding Sheet (professor's coding)
const MOTION_TYPES_AUTHORITY = ['Pressing', 'Floating', 'Dabbing', 'Punching', 'Slashing', 'Gliding'];
const MOTION_COLORS = {
  Pressing: '#8b5cf6',
  Floating: '#3b82f6',
  Dabbing: '#f59e0b',
  Punching: '#ef4444',
  Slashing: '#a855f7',
  Gliding: '#22c55e',
};

// Laban Effort factors (derived from motion types)
const EFFORT_COLORS = {
  Light: '#86efac',
  Strong: '#f87171',
  Sustained: '#93c5fd',
  Sudden: '#fbbf24',
  Direct: '#34d399',
  Indirect: '#a78bfa',
  Free: '#67e8f9',
  Bound: '#f472b6',
};

// Empty = same origin (when frontend served from backend)
const API_BASE = process.env.REACT_APP_API_URL ?? 'http://localhost:8000';

// MediaPipe Pose landmark connections (pairs of indices to draw lines between)
const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
];

function App() {
  const [status, setStatus] = useState('idle'); // idle | preview | uploading | ready | error
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [frames, setFrames] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const overlayVideoRef = useRef(null);
  const chartContainerRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const rafIdRef = useRef(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const progressIntervalRef = useRef(null);
  const [videoCurrentTime, setVideoCurrentTime] = useState(0);

  const drawSkeleton = useCallback((ctx, landmarks, width, height) => {
    if (!landmarks || landmarks.length === 0) return;

    const jointRadius = 6;
    const lineWidth = 3.5;

    // Draw lines with glow
    for (const [i, j] of POSE_CONNECTIONS) {
      const a = landmarks[i];
      const b = landmarks[j];
      if (!a || !b || (a.x === 0 && a.y === 0) || (b.x === 0 && b.y === 0)) continue;

      const x1 = a.x * width;
      const y1 = a.y * height;
      const x2 = b.x * width;
      const y2 = b.y * height;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);

      // Soft glow
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
      ctx.lineWidth = lineWidth + 8;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();

      // Main line with gradient
      const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
      gradient.addColorStop(0, '#38bdf8');
      gradient.addColorStop(0.5, '#22d3ee');
      gradient.addColorStop(1, '#0ea5e9');
      ctx.strokeStyle = gradient;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }

    // Draw joints with glow + fill
    for (const lm of landmarks) {
      if (lm.x === 0 && lm.y === 0) continue;
      const x = lm.x * width;
      const y = lm.y * height;

      // Outer glow
      const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, jointRadius + 6);
      glowGradient.addColorStop(0, 'rgba(56, 189, 248, 0.5)');
      glowGradient.addColorStop(0.6, 'rgba(56, 189, 248, 0.15)');
      glowGradient.addColorStop(1, 'transparent');
      ctx.fillStyle = glowGradient;
      ctx.beginPath();
      ctx.arc(x, y, jointRadius + 6, 0, Math.PI * 2);
      ctx.fill();

      // Joint fill
      const jointGradient = ctx.createRadialGradient(x - 2, y - 2, 0, x, y, jointRadius);
      jointGradient.addColorStop(0, '#7dd3fc');
      jointGradient.addColorStop(0.7, '#38bdf8');
      jointGradient.addColorStop(1, '#0284c7');
      ctx.fillStyle = jointGradient;
      ctx.beginPath();
      ctx.arc(x, y, jointRadius, 0, Math.PI * 2);
      ctx.fill();

      // Subtle highlight
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.beginPath();
      ctx.arc(x - 1.5, y - 1.5, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }, []);

  const findFrameForTime = useCallback((time) => {
    if (!frames.length) return null;
    let best = frames[0];
    for (const f of frames) {
      if (f.timestamp <= time) best = f;
      else break;
    }
    return best;
  }, [frames]);

  // Transform legacy_motion_per_second to chart data (Advancing, Retreating, Enclosing, etc.)
  const legacyChartData = useMemo(() => {
    const lmps = analysis?.legacy_motion_per_second;
    if (!lmps?.length) return [];
    const types = ['Advancing', 'Retreating', 'Enclosing', 'Spreading', 'Directing', 'Indirecting', 'Neutral'];
    return lmps.map((row) => {
      const point = { second: row.second };
      types.forEach((t) => { point[t] = 0; });
      row.motions?.forEach((m) => {
        if (types.includes(m.motion_type)) point[m.motion_type] = m.confidence;
      });
      return point;
    });
  }, [analysis?.legacy_motion_per_second]);

  // Transform motion_per_second to chart data: only 6 types from Authority Coding Sheet
  const motionChartData = useMemo(() => {
    const mps = analysis?.motion_per_second;
    if (!mps?.length) return [];
    const types = MOTION_TYPES_AUTHORITY;
    return mps.map((row) => {
      const point = { second: row.second };
      types.forEach((t) => { point[t] = 0; });
      row.motions?.forEach((m) => {
        if (types.includes(m.motion_type)) point[m.motion_type] = m.confidence;
      });
      return point;
    });
  }, [analysis?.motion_per_second]);

  // Transform effort_per_second to chart data: { second, Light, Strong, ... } with 0 for missing
  const effortChartData = useMemo(() => {
    const eps = analysis?.effort_per_second;
    if (!eps?.length) return [];
    const types = ['Light', 'Strong', 'Sustained', 'Sudden', 'Direct', 'Indirect', 'Free', 'Bound'];
    return eps.map((row) => {
      const point = { second: row.second };
      types.forEach((t) => { point[t] = 0; });
      row.efforts?.forEach((e) => {
        if (types.includes(e.effort_type)) point[e.effort_type] = e.confidence;
      });
      return point;
    });
  }, [analysis?.effort_per_second]);

  // Playhead sync: ~30fps RAF loop, syncs from whichever video is playing
  useEffect(() => {
    if (status !== 'ready' || (!legacyChartData.length && !motionChartData.length && !effortChartData.length)) return;
    const videos = [overlayVideoRef.current].filter(Boolean);
    if (videos.length === 0) return;

    let rafId;
    const tick = () => {
      const playing = videos.find((v) => !v.paused && !v.ended);
      if (playing) setVideoCurrentTime(playing.currentTime);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [status, legacyChartData.length, motionChartData.length, effortChartData.length]);

  useEffect(() => {
    const video = overlayVideoRef.current;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!video || !canvas || !container || status !== 'ready' || !frames.length) return;

    const ctx = canvas.getContext('2d');
    let lastDrawnFrame = -1;

    const resizeCanvas = () => {
      const rect = container.getBoundingClientRect();
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw === 0 || vh === 0) return;

      const scale = Math.min(rect.width / vw, rect.height / vh);
      const dw = vw * scale;
      const dh = vh * scale;

      canvas.width = vw;
      canvas.height = vh;
      canvas.style.width = `${dw}px`;
      canvas.style.height = `${dh}px`;
      canvas.style.left = `${(rect.width - dw) / 2}px`;
      canvas.style.top = `${(rect.height - dh) / 2}px`;
    };

    const tick = () => {
      if (!video || video.paused || video.ended) {
        rafIdRef.current = requestAnimationFrame(tick);
        return;
      }

      const time = video.currentTime;
      const frame = findFrameForTime(time);
      if (frame && frame.frame_index !== lastDrawnFrame) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawSkeleton(ctx, frame.landmarks, canvas.width, canvas.height);
        lastDrawnFrame = frame.frame_index;
      }

      rafIdRef.current = requestAnimationFrame(tick);
    };

    const onSeeked = () => {
      lastDrawnFrame = -1;
      const frame = findFrameForTime(video.currentTime);
      if (frame) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawSkeleton(ctx, frame.landmarks, canvas.width, canvas.height);
        lastDrawnFrame = frame.frame_index;
      }
    };

    const onLoadedData = () => {
      lastDrawnFrame = -1;
      tick();
    };

    resizeCanvas();
    const ro = new ResizeObserver(resizeCanvas);
    ro.observe(container);

    video.addEventListener('loadedmetadata', resizeCanvas);
    video.addEventListener('loadeddata', onLoadedData);
    video.addEventListener('play', tick);
    video.addEventListener('seeked', onSeeked);

    tick();

    return () => {
      ro.disconnect();
      video.removeEventListener('loadedmetadata', resizeCanvas);
      video.removeEventListener('loadeddata', onLoadedData);
      video.removeEventListener('play', tick);
      video.removeEventListener('seeked', onSeeked);
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [status, frames, findFrameForTime, drawSkeleton]);

  const handleFileSelect = useCallback((files) => {
    const file = files?.[0];
    if (!file) return;

    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    const allowed = ['.mp4', '.mkv', '.mov', '.avi'];
    if (!allowed.includes(ext)) {
      setError('Unsupported format. Use MP4, MKV, MOV, or AVI.');
      setStatus('error');
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setError(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setFrames([]);
    setStatus('preview');
  }, [previewUrl]);

  const createPose = useCallback(async () => {
    if (!selectedFile) return;

    setError(null);
    setStatus('uploading');
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', selectedFile);

    const startTime = Date.now();
    // Progress reaches 90% over ~5 min so it doesn't stall while backend processes (MediaPipe is slow)
    progressIntervalRef.current = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      setUploadProgress(Math.min(90, (elapsed / 300) * 90));
    }, 150);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
      setUploadProgress(100);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }

      const data = await res.json();
      setFrames(data.frames || []);
      setAnalysis(data.analysis || null);
      setStatus('ready');
    } catch (e) {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      setUploadProgress(0);
      setError(e.message || 'Processing failed');
      setStatus('preview');
    }
  }, [selectedFile]);

  const reset = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setFrames([]);
    setAnalysis(null);
    setStatus('idle');
    setError(null);
    setUploadProgress(0);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer?.files);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onFileInput = (e) => {
    handleFileSelect(e.target?.files);
    e.target.value = '';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 text-slate-100 flex flex-col items-center p-6 relative overflow-hidden">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 -left-40 w-72 h-72 bg-blue-400/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 right-1/3 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />
      </div>

      <header className="mb-10 relative z-10">
        <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-blue-200 via-blue-100 to-indigo-200 bg-clip-text text-transparent drop-shadow-sm">
          AI People Reader
        </h1>
        <p className="text-slate-400 text-sm mt-2">
          Upload a video, preview it, then create the pose overlay
        </p>
      </header>

      {status === 'idle' && (
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`
            relative z-10 w-full max-w-xl rounded-2xl p-14
            flex flex-col items-center justify-center gap-5
            transition-all duration-300 cursor-pointer
            shadow-xl
            ${isDragging
              ? 'glass-blue scale-[1.02] shadow-blue-500/20'
              : 'glass hover:bg-blue-500/5 hover:border-blue-400/30'
            }
          `}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <input
            id="file-input"
            type="file"
            accept=".mp4,.mkv,.mov,.avi"
            onChange={onFileInput}
            className="hidden"
          />
          <div className={`p-4 rounded-2xl transition-colors ${isDragging ? 'bg-blue-500/20' : 'bg-blue-500/10'}`}>
            <Upload className="w-12 h-12 text-blue-300" />
          </div>
          <p className="text-slate-300 text-center text-base">
            Drag and drop a video here, or click to browse
          </p>
          <p className="text-slate-500 text-sm">
            MP4, MKV, MOV, AVI
          </p>
        </div>
      )}

      {(status === 'preview' || status === 'uploading' || status === 'ready') && previewUrl && (
        <div className="relative z-10 w-full max-w-4xl space-y-6">
          {/* Single video: preview/upload shows plain video; ready shows video + skeleton overlay */}
          <div className="space-y-2">
            <h2 className="text-sm font-medium text-slate-400">
              {status === 'ready' ? 'Pose overlay' : 'Video'}
            </h2>
            <div
              ref={containerRef}
              className="relative rounded-2xl overflow-hidden aspect-video max-h-[50vh] flex items-center justify-center w-full glass shadow-xl bg-black"
            >
              <video
                ref={overlayVideoRef}
                src={previewUrl}
                controls
                playsInline
                className="max-w-full max-h-full object-contain"
              />
              {status === 'ready' && frames.length > 0 && (
                <canvas
                  ref={canvasRef}
                  className="absolute pointer-events-none"
                />
              )}
            </div>
            <div className="space-y-3">
              {status !== 'ready' && (
                <div className="flex items-center gap-3">
                  <button
                    onClick={createPose}
                    disabled={status === 'uploading'}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl glass-blue hover:bg-blue-500/20 text-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {status === 'uploading' ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Processing…
                      </>
                    ) : (
                      <>
                        <Heart className="w-4 h-4" />
                        Create Pose
                      </>
                    )}
                  </button>
                  <button
                    onClick={reset}
                    disabled={status === 'uploading'}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl glass hover:bg-white/5 text-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Video className="w-4 h-4" />
                    New video
                  </button>
                </div>
              )}
              {status === 'ready' && (
                <div className="flex items-center gap-3">
                  <button
                    onClick={reset}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl glass hover:bg-white/5 text-slate-300"
                  >
                    <Video className="w-4 h-4" />
                    New video
                  </button>
                </div>
              )}
              {status === 'uploading' && (
                <div className="w-full space-y-1.5">
                  <div className="h-2 rounded-full bg-slate-700/80 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-300 ease-out rounded-full"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-slate-400 text-sm">
                    Extracting pose & analyzing… {Math.round(uploadProgress)}%
                  </p>
                </div>
              )}
            </div>
            {status === 'ready' && frames.length > 0 && (
              <p className="text-slate-500 text-sm">{frames.length} frames • Skeleton overlay</p>
            )}
          </div>

          {/* Charts - shown after processing */}
          {status === 'ready' && frames.length > 0 && (
            <div className="space-y-6">

              {/* Top: Legacy motion (Advancing, Retreating, Enclosing, Spreading, Directing, Indirecting) */}
              {legacyChartData.length > 0 && (
                <div className="space-y-2">
                  <h2 className="text-sm font-medium text-slate-400">Shape & direction confidence over time</h2>
                  <div className="relative rounded-2xl p-4 h-64 bg-black">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={legacyChartData} margin={{ top: 5, right: 30, left: 55, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                        <XAxis
                          dataKey="second"
                          tickFormatter={(v) => `${v}s`}
                          stroke="#94a3b8"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tickFormatter={(v) => `${v}%`}
                          stroke="#94a3b8"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                          labelFormatter={(v) => `${v}s`}
                          formatter={(value, name) => [`${value}%`, name]}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {Object.keys(LEGACY_MOTION_COLORS).map((key) => (
                          <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            stroke={LEGACY_MOTION_COLORS[key]}
                            strokeWidth={2}
                            dot={false}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                    {/* Sliding playhead */}
                    {(() => {
                      const maxSec = legacyChartData.length ? legacyChartData[legacyChartData.length - 1].second : 1;
                      const pct = maxSec > 0 ? Math.min(1, Math.max(0, videoCurrentTime / maxSec)) : 0;
                      return (
                        <div className="absolute top-4 bottom-4 left-4 right-4 pointer-events-none">
                          <div
                            className="absolute top-0 bottom-0 -ml-px"
                            style={{
                              left: `calc(115px + (100% - 130px) * ${pct})`,
                              width: 5,
                              backgroundColor: '#ef4444',
                              borderLeft: 'none',
                              boxShadow: '0 0 8px #ef4444',
                            }}
                          />
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}

              {/* Middle: Motion confidence over time (CSV types) */}
              {motionChartData.length > 0 && (
                <div className="space-y-2">
                  <h2 className="text-sm font-medium text-slate-400">Motion confidence over time</h2>
                  <div ref={chartContainerRef} className="relative rounded-2xl p-4 h-64 bg-black">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={motionChartData} margin={{ top: 5, right: 30, left: 55, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                        <XAxis
                          dataKey="second"
                          tickFormatter={(v) => `${v}s`}
                          stroke="#94a3b8"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tickFormatter={(v) => `${v}%`}
                          stroke="#94a3b8"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                          labelFormatter={(v) => `${v}s`}
                          formatter={(value, name) => [`${value}%`, name]}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {Object.keys(MOTION_COLORS).map((key) => (
                          <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            stroke={MOTION_COLORS[key]}
                            strokeWidth={2}
                            dot={false}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                    {/* Sliding playhead */}
                    {(() => {
                      const maxSec = motionChartData.length ? motionChartData[motionChartData.length - 1].second : 1;
                      const pct = maxSec > 0 ? Math.min(1, Math.max(0, videoCurrentTime / maxSec)) : 0;
                      return (
                        <div className="absolute top-4 bottom-4 left-4 right-4 pointer-events-none">
                          <div
                            className="absolute top-0 bottom-0 -ml-px"
                            style={{
                              left: `calc(115px + (100% - 130px) * ${pct})`,
                              width: 5,
                              backgroundColor: '#ef4444',
                              borderLeft: 'none',
                              boxShadow: '0 0 8px #ef4444',
                            }}
                          />
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}

              {/* Lower: Effort confidence over time */}
              {effortChartData.length > 0 && (
                <div className="space-y-2">
                  <h2 className="text-sm font-medium text-slate-400">Effort confidence over time</h2>
                  <div className="relative rounded-2xl p-4 h-64 bg-black">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={effortChartData} margin={{ top: 5, right: 30, left: 55, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                        <XAxis
                          dataKey="second"
                          tickFormatter={(v) => `${v}s`}
                          stroke="#94a3b8"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tickFormatter={(v) => `${v}%`}
                          stroke="#94a3b8"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                          labelFormatter={(v) => `${v}s`}
                          formatter={(value, name) => [`${value}%`, name]}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {Object.keys(EFFORT_COLORS).map((key) => (
                          <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            stroke={EFFORT_COLORS[key]}
                            strokeWidth={2}
                            dot={false}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                    {/* Sliding playhead */}
                    {(() => {
                      const maxSec = effortChartData.length ? effortChartData[effortChartData.length - 1].second : 1;
                      const pct = maxSec > 0 ? Math.min(1, Math.max(0, videoCurrentTime / maxSec)) : 0;
                      return (
                        <div className="absolute top-4 bottom-4 left-4 right-4 pointer-events-none">
                          <div
                            className="absolute top-0 bottom-0 -ml-px"
                            style={{
                              left: `calc(115px + (100% - 130px) * ${pct})`,
                              width: 5,
                              backgroundColor: '#ef4444',
                              borderLeft: 'none',
                              boxShadow: '0 0 8px #ef4444',
                            }}
                          />
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}

            </div>
          )}

          {error && (
            <div className="rounded-xl glass border-red-500/30 px-4 py-3 text-red-300 text-sm">
              {error}
            </div>
          )}
        </div>
      )}

      {status === 'error' && (
        <div className="relative z-10 w-full max-w-xl rounded-2xl glass p-6 border-red-500/30">
          <p className="text-red-300">{error}</p>
          <button
            onClick={reset}
            className="mt-4 px-5 py-2.5 rounded-xl glass-blue hover:bg-blue-500/20 text-blue-200 transition-colors"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
