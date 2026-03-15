"""
Tune motion detection to match Authority Coding Sheet.csv (professor's coding).
Uses the 6 motion types: Pressing, Floating, Dabbing, Punching, Slashing, Gliding.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTHORITY_CSV = PROJECT_ROOT / "Authority Coding Sheet.csv"
SKELETON_CACHE_DIR = Path(__file__).resolve().parent / "debug_skeleton_cache"

# 6 types from Authority Coding Sheet (second graph)
AUTHORITY_TYPES = ["Pressing", "Floating", "Dabbing", "Punching", "Slashing", "Gliding"]


def parse_time_mm_ss(s: str) -> float:
    """Parse MM:SS to seconds. E.g. 00:02 -> 2, 01:30 -> 90."""
    s = str(s).strip()
    if not s:
        return -1
    parts = s.split(":")
    if len(parts) == 2:
        try:
            m, sec = int(parts[0]), int(parts[1])
            return m * 60 + sec
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return -1


def load_authority_csv(path: Path) -> list:
    """Load Authority Coding Sheet.csv. Returns [(time_sec, {type: 1 or 0})]."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            time_str = (row.get("Time") or "").strip()
            if not time_str:
                continue
            time_sec = parse_time_mm_ss(time_str)
            if time_sec < 0:
                continue
            shapes = {}
            for col in AUTHORITY_TYPES:
                if col in headers:
                    val = (row.get(col) or "").strip()
                    shapes[col] = 1 if val in ("1", "x", "X") else 0
            if any(shapes.values()):
                rows.append((time_sec, shapes))
    return rows


def align_gt_to_seconds(gt_rows: list) -> dict:
    """Group ground truth by second. Each second gets union of shapes present."""
    by_second = defaultdict(lambda: {t: 0 for t in AUTHORITY_TYPES})
    for time_sec, shapes in gt_rows:
        sec = int(time_sec)
        for name, val in shapes.items():
            if val:
                by_second[sec][name] = 1
    return dict(by_second)


def load_skeleton(cache_path: Path):
    """Load frames from skeleton cache JSON."""
    with open(cache_path) as f:
        data = json.load(f)
    return data.get("frames", [])


def run_analysis(frames, params=None):
    """Run motion analysis with optional param overrides."""
    import analysis

    if params:
        analysis.TUNE_PARAMS.update(params)
    else:
        analysis.TUNE_PARAMS.clear()
    result = analysis.analyze_motion_per_second(frames)
    analysis.TUNE_PARAMS.clear()
    return result


def evaluate(pred_per_second: dict, gt_by_second: dict) -> dict:
    """Compare predicted to ground truth for the 6 Authority types."""
    tp = fp = tn = fn = 0
    all_seconds = set(pred_per_second.keys()) | set(gt_by_second.keys())

    for sec in sorted(all_seconds):
        pred_motions = pred_per_second.get(sec, [])
        pred_shapes = {m["motion_type"] for m in pred_motions if m["motion_type"] in AUTHORITY_TYPES}
        gt = gt_by_second.get(sec, {})

        for name in AUTHORITY_TYPES:
            pred_pos = name in pred_shapes
            gt_pos = gt.get(name, 0) == 1
            if gt_pos and pred_pos:
                tp += 1
            elif gt_pos and not pred_pos:
                fn += 1
            elif not gt_pos and pred_pos:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main():
    if not AUTHORITY_CSV.exists():
        print(f"Authority Coding Sheet not found: {AUTHORITY_CSV}")
        return

    gt_rows = load_authority_csv(AUTHORITY_CSV)
    print(f"Loaded {len(gt_rows)} ground truth rows from {AUTHORITY_CSV.name}")
    if gt_rows:
        print(f"  Time range: {gt_rows[0][0]:.0f}s - {gt_rows[-1][0]:.0f}s")
        print(f"  Types: {AUTHORITY_TYPES}")

    gt_by_second = align_gt_to_seconds(gt_rows)
    gt_seconds = sorted(gt_by_second.keys())
    print(f"  Ground truth seconds: {gt_seconds[:10]}... ({len(gt_seconds)} total)")

    # Find skeleton cache (use first available)
    cache_files = list(SKELETON_CACHE_DIR.glob("*.json")) if SKELETON_CACHE_DIR.exists() else []
    if not cache_files:
        print("No skeleton cache found. Upload a video first to generate cache.")
        return

    cache_path = cache_files[0]
    frames = load_skeleton(cache_path)
    print(f"\nLoaded {len(frames)} frames from {cache_path.name}")

    # Param sets to try (tuned for professor's coding)
    param_sets = [
        {},
        # Relaxed for better recall (professor may code more liberally)
        {
            "enclosing_max_expansion": 0.9,
            "enclosing_min_velocity": 0.02,
            "spreading_expansion_threshold": 1.2,
            "spreading_min_velocity": 0.02,
            "sustained_vel_min": 0.015,
            "sustained_vel_max": 0.10,
            "forward_z_threshold": -0.025,
            "backward_z_threshold": 0.02,
        },
        # More sensitive for Pressing, Gliding (sustained)
        {
            "enclosing_max_expansion": 0.95,
            "enclosing_min_velocity": 0.015,
            "spreading_expansion_threshold": 1.15,
            "spreading_min_velocity": 0.015,
            "sustained_vel_min": 0.01,
            "sustained_vel_max": 0.12,
            "forward_z_threshold": -0.02,
            "backward_z_threshold": 0.018,
        },
    ]

    best_f1 = -1
    best_params = None
    best_pred = None

    for i, params in enumerate(param_sets):
        pred = run_analysis(frames, params)
        pred_by_second = {r["second"]: r["motions"] for r in pred}
        metrics = evaluate(pred_by_second, gt_by_second)
        label = "default" if not params else f"tuned_{i}"
        print(f"\n--- {label} --- F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_params = params
            best_pred = pred_by_second

    print("\n" + "=" * 50)
    print(f"BEST F1={best_f1:.3f}")
    if best_params:
        print("Params:", best_params)

    # Per-second comparison for best
    pred_by_second = best_pred or {r["second"]: r["motions"] for r in run_analysis(frames, best_params)}
    print("\n--- Per-second comparison (best) ---")
    for sec in gt_seconds[:20]:  # First 20 seconds
        gt_types = sorted([k for k, v in gt_by_second[sec].items() if v == 1])
        pred_types = sorted(
            [m["motion_type"] for m in pred_by_second.get(sec, []) if m["motion_type"] in AUTHORITY_TYPES]
        )
        match = "OK" if set(gt_types) == set(pred_types) else "MISMATCH"
        print(f"  {sec}s: GT={gt_types} | PRED={pred_types} [{match}]")


if __name__ == "__main__":
    main()
