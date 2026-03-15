"""
Tune motion detection algorithm to match Excel ground truth.
Excel time is in MINUTES. Shape columns: Enclosing(7), Spreading(8), Directing(9), Indirecting(10), Advancing(11), Retreating(12)
"""

import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

EXCEL_PATH = Path(__file__).parent.parent / "public" / "Authority Coding Sheets (1).xlsx"
SKELETON_PATH = Path(__file__).parent / "debug_skeleton_cache" / "f26c6a522f4c62e8.json"

# Shape columns in Excel (0-indexed): Enclosing, Spreading, Directing, Indirecting, Advancing, Retreating
SHAPE_COLS = {
    7: "Enclosing",
    8: "Spreading",
    9: "Directing",
    10: "Indirecting",
    11: "Advancing",
    12: "Retreating",
}

# Tunable parameters for _detect_motion_counts_for_frames
DEFAULT_PARAMS = {
    "enclosing_max_expansion": 0.8,
    "enclosing_min_velocity": 0.03,
    "spreading_expansion_threshold": 1.3,
    "spreading_min_velocity": 0.03,
    "forward_z_threshold": -0.03,
    "backward_z_threshold": 0.05,
    "advancing_min_velocity": 0.06,
    "retreating_min_velocity": 0.07,
    "indirecting_min_velocity": 0.04,
    "indirecting_min_expansion": 1.0,
    "indirecting_lateral_threshold": 0.001,
    "sustained_vel_min": 0.02,
    "sustained_vel_max": 0.08,
}


def load_excel_ground_truth():
    """Load Excel and return list of (time_sec, {shape: 1 or 0})"""
    shared_strings = []
    with zipfile.ZipFile(EXCEL_PATH, "r") as z:
        try:
            with z.open("xl/sharedStrings.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()
                ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in root.findall(".//main:si", ns):
                    texts = si.findall(".//main:t", ns)
                    shared_strings.append("".join(t.text or "" for t in texts))
        except Exception:
            pass

        cells = {}
        with z.open("xl/worksheets/sheet1.xml") as f:
            tree = ET.parse(f)
            root = tree.getroot()
            ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for c in root.findall(".//main:c", ns):
                ref = c.get("r", "")
                col = 0
                row = 0
                for i, ch in enumerate(ref):
                    if ch.isdigit():
                        row = int(ref[i:]) - 1
                        break
                    col = col * 26 + (ord(ch.upper()) - ord("A") + 1)
                col -= 1
                t = c.get("t")
                v = c.find("main:v", ns)
                val = v.text if v is not None else ""
                if t == "s" and val.isdigit():
                    idx = int(val)
                    val = shared_strings[idx] if idx < len(shared_strings) else val
                cells[(row, col)] = val

    # Extract data rows (skip header rows 0-5)
    rows = []
    for r in range(6, 10000):
        time_val = cells.get((r, 0), "")
        if not time_val:
            continue
        try:
            time_min = float(time_val)
        except ValueError:
            continue
        time_sec = time_min * 60  # Excel time is in minutes
        shapes = {}
        for col, name in SHAPE_COLS.items():
            val = cells.get((r, col), "")
            shapes[name] = 1 if str(val).strip() == "1" else 0
        if any(shapes.values()):
            rows.append((time_sec, shapes))
    return rows


def load_skeleton():
    with open(SKELETON_PATH) as f:
        data = json.load(f)
    return data["frames"]


def run_analysis(frames, params=None):
    """Import and run the analysis - returns motion_per_second"""
    import analysis
    if params:
        analysis.TUNE_PARAMS.update(params)
    else:
        analysis.TUNE_PARAMS.clear()
    from analysis import analyze_motion_per_second
    result = analyze_motion_per_second(frames)
    analysis.TUNE_PARAMS.clear()
    return result


def align_ground_truth_to_seconds(gt_rows):
    """Group ground truth by second (floor). Each second gets union of all shapes present."""
    by_second = defaultdict(lambda: defaultdict(int))
    for time_sec, shapes in gt_rows:
        sec = int(time_sec)
        for name, val in shapes.items():
            if val:
                by_second[sec][name] = 1
    return dict(by_second)


def evaluate(pred_per_second, gt_by_second):
    """Compare predicted motion_per_second to ground truth. Return accuracy metrics."""
    shape_names = list(SHAPE_COLS.values())
    tp = fp = tn = fn = 0
    matched_seconds = 0
    total_seconds = 0

    all_seconds = set(pred_per_second.keys()) | set(gt_by_second.keys())
    for sec in sorted(all_seconds):
        pred = pred_per_second.get(sec, [])
        gt = gt_by_second.get(sec, {})

        pred_shapes = set()
        for m in pred:
            if m["motion_type"] in shape_names:
                pred_shapes.add(m["motion_type"])

        for name in shape_names:
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

        if sec in gt_by_second and any(gt_by_second[sec].values()):
            total_seconds += 1
            pred_set = set(m["motion_type"] for m in pred if m["motion_type"] in shape_names)
            gt_set = set(k for k, v in gt.items() if v == 1)
            if pred_set == gt_set:
                matched_seconds += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_seconds": matched_seconds,
        "total_gt_seconds": total_seconds,
    }


def main():
    gt_rows = load_excel_ground_truth()
    print(f"Loaded {len(gt_rows)} ground truth rows from Excel")
    print(f"Time range: {gt_rows[0][0]:.3f}s - {gt_rows[-1][0]:.3f}s" if gt_rows else "empty")

    gt_by_second = align_ground_truth_to_seconds(gt_rows)
    print(f"Ground truth seconds: {sorted(gt_by_second.keys())}")

    frames = load_skeleton()
    print(f"Loaded {len(frames)} skeleton frames")

    # Try progressively relaxed parameters
    param_sets = [
        {},
        # Target: Retreating (lower bwd_z, ret_vel), Indirecting (lower lateral, ind_vel)
        {
            "backward_z_threshold": 0.02,
            "retreating_min_velocity": 0.025,
            "indirecting_lateral_threshold": 0.001,
            "indirecting_min_velocity": 0.01,
        },
        {
            "enclosing_max_expansion": 0.95,
            "enclosing_min_velocity": 0.015,
            "spreading_expansion_threshold": 1.15,
            "spreading_min_velocity": 0.015,
            "forward_z_threshold": -0.02,
            "backward_z_threshold": 0.02,
            "advancing_min_velocity": 0.05,
            "retreating_min_velocity": 0.025,
            "indirecting_min_velocity": 0.01,
            "indirecting_min_expansion": 0.9,
            "indirecting_lateral_threshold": 0.001,
            "sustained_vel_min": 0.015,
            "sustained_vel_max": 0.10,
        },
        {
            "enclosing_max_expansion": 1.0,
            "enclosing_min_velocity": 0.01,
            "spreading_expansion_threshold": 1.1,
            "spreading_min_velocity": 0.01,
            "forward_z_threshold": -0.015,
            "backward_z_threshold": 0.02,
            "advancing_min_velocity": 0.04,
            "retreating_min_velocity": 0.02,
            "indirecting_min_velocity": 0.008,
            "indirecting_min_expansion": 0.85,
            "indirecting_lateral_threshold": 0.0008,
            "sustained_vel_min": 0.01,
            "sustained_vel_max": 0.12,
        },
    ]

    best_f1 = -1
    best_params = None
    best_pred = None

    for i, params in enumerate(param_sets):
        pred = run_analysis(frames, params)
        pred_by_second = {r["second"]: r["motions"] for r in pred}
        metrics = evaluate(pred_by_second, gt_by_second)
        label = "default" if not params else f"relaxed_{i}"
        print(f"\n--- {label} --- F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_params = params
            best_pred = pred_by_second

    print("\n" + "=" * 50)
    print("BEST RESULT (F1={:.3f})".format(best_f1))
    if best_params:
        print("Params:", best_params)

    pred_by_second = best_pred or {r["second"]: r["motions"] for r in run_analysis(frames, best_params)}
    print("\n--- Per-second comparison (BEST) ---")
    for sec in sorted(gt_by_second.keys()):
        gt_shapes = sorted([k for k, v in gt_by_second[sec].items() if v == 1])
        pred_shapes = sorted([m["motion_type"] for m in pred_by_second.get(sec, []) if m["motion_type"] in SHAPE_COLS.values()])
        match = "OK" if set(gt_shapes) == set(pred_shapes) else "MISMATCH"
        print(f"  {sec}s: GT={gt_shapes}")
        print(f"       PRED={pred_shapes} [{match}]")
        print()



if __name__ == "__main__":
    main()
