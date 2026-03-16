"""
Pose analysis matching report_core.py exactly.
Eye contact, uprightness, stance, engagement, confidence, authority.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Project root (parent of backend/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MOVEMENT_COMBO_CSV = _PROJECT_ROOT / "Movement Combination Summary.csv"

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

ENCLOSING_MAX_EXPANSION = 0.8
ENCLOSING_MIN_VELOCITY = 0.03
SPREADING_BODY_EXPANSION_THRESHOLD = 1.3
SPREADING_MIN_VELOCITY = 0.03

# Optional overrides for tuning (set by tune_algorithm.py)
TUNE_PARAMS: Dict[str, float] = {}


def _visible(lm: dict) -> bool:
    if not lm:
        return False
    x, y, z = lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)
    return abs(x) > 1e-6 or abs(y) > 1e-6 or abs(z) > 1e-6


def _pt(lm: dict) -> tuple:
    return (float(lm.get("x", 0)), float(lm.get("y", 0)))


def _normalize_horizontal_angle(deg: float) -> float:
    while deg > 180.0:
        deg -= 360.0
    while deg <= -180.0:
        deg += 360.0
    if deg > 90.0:
        deg -= 180.0
    if deg < -90.0:
        deg += 180.0
    return deg


def _apply_default_retreating_share(
    detection: Dict[str, float],
    retreat_key: str = "Retreating",
    retreat_default_pct: float = 1.0,
) -> Dict[str, float]:
    if not isinstance(detection, dict) or retreat_key not in detection:
        return detection
    fixed = max(0.0, min(100.0, float(retreat_default_pct)))
    other_keys = [k for k in detection.keys() if k != retreat_key]
    target_other_total = max(0.0, 100.0 - fixed)
    current_other_total = sum(max(0.0, float(detection.get(k, 0.0))) for k in other_keys)
    out: Dict[str, float] = {}
    if other_keys:
        if current_other_total <= 0.0:
            even_share = target_other_total / float(len(other_keys))
            for k in other_keys:
                out[k] = round(even_share, 1)
        else:
            for k in other_keys:
                raw = max(0.0, float(detection.get(k, 0.0)))
                out[k] = round((raw / current_other_total) * target_other_total, 1)
    out[retreat_key] = round(fixed, 1)
    return out


def analyze_first_impression(frames: List[dict]) -> Dict[str, float]:
    """Compute eye contact, uprightness, stance — matches report_core.analyze_first_impression_from_video."""
    samples = []
    for f in frames:
        lms = f.get("landmarks", [])
        if len(lms) < 29:
            continue
        nose = lms[NOSE]
        leye = lms[LEFT_EYE]
        reye = lms[RIGHT_EYE]
        lsh = lms[LEFT_SHOULDER]
        rsh = lms[RIGHT_SHOULDER]
        lhip = lms[LEFT_HIP]
        rhip = lms[RIGHT_HIP]
        lank = lms[LEFT_ANKLE]
        rank = lms[RIGHT_ANKLE]
        if not all(_visible(p) for p in [nose, leye, reye, lsh, rsh, lhip, rhip]):
            continue
        samples.append({
            "nose": _pt(nose),
            "leye": _pt(leye),
            "reye": _pt(reye),
            "lsh": _pt(lsh),
            "rsh": _pt(rsh),
            "lhip": _pt(lhip),
            "rhip": _pt(rhip),
            "lank": _pt(lank),
            "rank": _pt(rank),
            "leye_vis": 1.0,
            "reye_vis": 1.0,
            "lank_vis": 1.0,
            "rank_vis": 1.0,
        })

    if len(samples) < 3:
        return {"eye_contact": 0.0, "uprightness": 0.0, "stance": 0.0}

    roll_candidates = []
    for s in samples:
        lsh, rsh = s["lsh"], s["rsh"]
        lhip, rhip = s["lhip"], s["rhip"]
        sh_ang = math.degrees(math.atan2(rsh[1] - lsh[1], rsh[0] - lsh[0]))
        hip_ang = math.degrees(math.atan2(rhip[1] - lhip[1], rhip[0] - lhip[0]))
        roll_candidates.extend([_normalize_horizontal_angle(sh_ang), _normalize_horizontal_angle(hip_ang)])
    camera_roll = max(-20.0, min(20.0, float(np.median(roll_candidates))))
    theta = -math.radians(camera_roll)

    def rotate(pt: tuple, cx: float = 0.5, cy: float = 0.5) -> tuple:
        x, y = pt
        dx, dy = x - cx, y - cy
        xr = dx * math.cos(theta) - dy * math.sin(theta) + cx
        yr = dx * math.sin(theta) + dy * math.cos(theta) + cy
        return (xr, yr)

    eye_scores = []
    upright_scores = []
    ankle_dists = []
    ankle_centers = []
    stance_ratios = []
    prev_nose_offset = None
    prev_torso_angle = None

    for s in samples:
        nose = rotate(s["nose"])
        leye = rotate(s["leye"])
        reye = rotate(s["reye"])
        lsh = rotate(s["lsh"])
        rsh = rotate(s["rsh"])
        lhip = rotate(s["lhip"])
        rhip = rotate(s["rhip"])
        lank = rotate(s["lank"])
        rank = rotate(s["rank"])

        eye_dist = abs(leye[0] - reye[0])
        if eye_dist > 1e-4:
            mid_eye_x = (leye[0] + reye[0]) / 2.0
            mid_eye_y = (leye[1] + reye[1]) / 2.0
            nose_offset = abs(nose[0] - mid_eye_x) / eye_dist
            sym = max(0.0, 1.0 - (nose_offset / 0.75))
            stab = 1.0
            if prev_nose_offset is not None:
                jitter = abs(nose_offset - prev_nose_offset)
                stab = max(0.70, 1.0 - (jitter / 0.35))
            prev_nose_offset = nose_offset
            base = 100.0 * sym * stab
            nose_eye_ratio = (nose[1] - mid_eye_y) / eye_dist
            look_down = max(0.0, min(1.0, (nose_eye_ratio - 0.55) / 0.50))
            mid_sh_y = (lsh[1] + rsh[1]) / 2.0
            mid_sh = np.array([(lsh[0] + rsh[0]) / 2.0, (lsh[1] + rsh[1]) / 2.0])
            mid_hip = np.array([(lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0])
            torso_len = float(np.linalg.norm(mid_sh - mid_hip)) + 1e-9
            head_raise = (mid_sh_y - nose[1]) / torso_len
            head_down = max(0.0, min(1.0, (0.28 - head_raise) / 0.18))
            eye_vis = min(float(s["leye_vis"]), float(s["reye_vis"]))
            eye_closed = max(0.0, min(1.0, (0.85 - eye_vis) / 0.35))
            penalty = 0.28 * look_down + 0.22 * head_down + 0.18 * eye_closed
            eye_scores.append(base * max(0.55, 1.0 - penalty))

        mid_sh = np.array([(lsh[0] + rsh[0]) / 2.0, (lsh[1] + rsh[1]) / 2.0])
        mid_hip = np.array([(lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0])
        v = mid_sh - mid_hip
        vert = np.array([0.0, -1.0])
        v_norm = np.linalg.norm(v) + 1e-9
        cosang = float(np.dot(v / v_norm, vert))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        shoulder_tilt = abs(lsh[1] - rsh[1]) / v_norm
        angle_score = max(0.0, 1.0 - (ang / 28.0))
        shoulder_score = max(0.0, 1.0 - (shoulder_tilt / 0.20))
        temporal = 1.0
        if prev_torso_angle is not None:
            jump = abs(ang - prev_torso_angle)
            temporal = max(0.75, 1.0 - (jump / 45.0))
        prev_torso_angle = ang
        upright_scores.append(100.0 * (0.75 * angle_score + 0.25 * shoulder_score) * temporal)

        if min(float(s["lank_vis"]), float(s["rank_vis"])) >= 0.5:
            dx = lank[0] - rank[0]
            dy = lank[1] - rank[1]
            dist = math.sqrt(dx * dx + dy * dy)
            ankle_dists.append(dist)
            ankle_centers.append((lank[0] + rank[0]) / 2.0)
            sw = abs(lsh[0] - rsh[0])
            if sw > 1e-4:
                stance_ratios.append(dist / sw)

    eye_pct = float(np.mean(eye_scores)) if eye_scores else 0.0
    upright_pct = float(np.mean(upright_scores)) if upright_scores else 0.0

    stance = 0.0
    if len(ankle_dists) >= 10:
        dist_arr = np.array(ankle_dists)
        rel_std = np.std(dist_arr) / (np.mean(dist_arr) + 1e-9)
        base = max(0.0, min(100.0, 100.0 * (1.0 - (rel_std / 0.35))))
        sway = 0.0
        if len(ankle_centers) >= 10:
            sway = max(0.0, min(100.0, 100.0 * (1.0 - (np.std(ankle_centers) / 0.06))))
        width = 75.0
        if stance_ratios:
            r = np.array(stance_ratios)
            pref = max(0.70, 1.0 - (abs(np.mean(r) - 1.10) / 1.80))
            wstab = max(0.0, 1.0 - (np.std(r) / 0.35))
            width = 100.0 * (0.65 * pref + 0.35 * wstab)
        stance = max(0.0, min(100.0, 0.62 * base + 0.30 * sway + 0.08 * width))

    return {"eye_contact": round(eye_pct, 1), "uprightness": round(upright_pct, 1), "stance": round(stance, 1)}


def _detect_motion_counts_for_frames(frames: List[dict]) -> Dict[str, int]:
    """Count motion type occurrences across consecutive frame pairs. Used for per-second analysis."""
    p = TUNE_PARAMS  # Optional tuning overrides
    enc_max = p.get("enclosing_max_expansion", ENCLOSING_MAX_EXPANSION)
    enc_min_vel = p.get("enclosing_min_velocity", ENCLOSING_MIN_VELOCITY)
    spread_exp = p.get("spreading_expansion_threshold", SPREADING_BODY_EXPANSION_THRESHOLD)
    spread_min_vel = p.get("spreading_min_velocity", SPREADING_MIN_VELOCITY)
    fwd_z = p.get("forward_z_threshold", -0.03)
    bwd_z = p.get("backward_z_threshold", 0.02)  # Tuned: 0.05 missed Retreating in sec 0
    adv_vel = p.get("advancing_min_velocity", 0.05)  # Tuned: 0.06 missed Advancing in sec 1
    ret_vel = p.get("retreating_min_velocity", 0.025)  # Tuned: 0.07 missed Retreating
    ind_vel = p.get("indirecting_min_velocity", 0.04)
    ind_exp = p.get("indirecting_min_expansion", 1.0)
    ind_lateral = p.get("indirecting_lateral_threshold", 0.001)
    sus_min = p.get("sustained_vel_min", 0.02)
    sus_max = p.get("sustained_vel_max", 0.07)  # Keep below is_sudden (0.08) for clean Pressing vs Punching

    # Complete_Effort_Action_Motion_Descriptions.csv: Gliding, Floating, Punching, Dabbing, Flicking, Slashing, Wringing, Pressing
    counts = {
        "Gliding": 0, "Floating": 0, "Punching": 0, "Dabbing": 0,
        "Flicking": 0, "Slashing": 0, "Wringing": 0, "Pressing": 0,
    }
    for i in range(1, len(frames)):
        curr = frames[i].get("landmarks", [])
        prev = frames[i - 1].get("landmarks", [])
        if len(curr) < 29 or len(prev) < 29:  # need ankles (27,28) for stepping
            continue

        lw, rw = curr[LEFT_WRIST], curr[RIGHT_WRIST]
        plw, prw = prev[LEFT_WRIST], prev[RIGHT_WRIST]
        ls, rs = curr[LEFT_SHOULDER], curr[RIGHT_SHOULDER]

        if not all(_visible(p) for p in [lw, rw, plw, prw, ls, rs]):
            continue

        # Body movement for stepping (Advancing) - arms may be still when stepping forward
        lh, rh = curr[LEFT_HIP], curr[RIGHT_HIP]
        plh, prh = prev[LEFT_HIP], prev[RIGHT_HIP]
        la, ra = curr[LEFT_ANKLE], curr[RIGHT_ANKLE]
        pla, pra = prev[LEFT_ANKLE], prev[RIGHT_ANKLE]
        nose, pnose = curr[NOSE], prev[NOSE]
        hip_z_d = (lh.get("z", 0) + rh.get("z", 0)) / 2 - (plh.get("z", 0) + prh.get("z", 0)) / 2
        ankle_z_d = (la.get("z", 0) + ra.get("z", 0)) / 2 - (pla.get("z", 0) + pra.get("z", 0)) / 2
        nose_z_d = nose.get("z", 0) - pnose.get("z", 0)
        body_forward = hip_z_d < fwd_z or nose_z_d < fwd_z or ankle_z_d < fwd_z
        hip_vel = math.sqrt(
            ((lh["x"] + rh["x"]) / 2 - (plh["x"] + prh["x"]) / 2) ** 2
            + ((lh["y"] + rh["y"]) / 2 - (plh["y"] + prh["y"]) / 2) ** 2
            + (hip_z_d ** 2)
        )
        ankle_vel = math.sqrt(
            ((la["x"] + ra["x"]) / 2 - (pla["x"] + pra["x"]) / 2) ** 2
            + ((la["y"] + ra["y"]) / 2 - (pla["y"] + pra["y"]) / 2) ** 2
            + (ankle_z_d ** 2)
        )
        nose_vel = math.sqrt(
            (nose["x"] - pnose["x"]) ** 2 + (nose["y"] - pnose["y"]) ** 2 + (nose_z_d ** 2)
        )

        lw_vel = math.sqrt((lw["x"] - plw["x"]) ** 2 + (lw["y"] - plw["y"]) ** 2 + (lw.get("z", 0) - plw.get("z", 0)) ** 2)
        rw_vel = math.sqrt((rw["x"] - prw["x"]) ** 2 + (rw["y"] - prw["y"]) ** 2 + (rw.get("z", 0) - prw.get("z", 0)) ** 2)
        avg_vel = (lw_vel + rw_vel) / 2

        wrist_dist = abs(lw["x"] - rw["x"])
        shoulder_width = max(abs(ls["x"] - rs["x"]), 0.1)
        body_expansion = wrist_dist / shoulder_width

        # Expansion delta: contracting (negative) = Enclosing cue; expanding (positive) = Spreading cue
        pwrist = abs(plw["x"] - prw["x"])
        pshoulder = max(abs(prev[LEFT_SHOULDER]["x"] - prev[RIGHT_SHOULDER]["x"]), 0.1)
        prev_expansion = pwrist / pshoulder
        expansion_delta = body_expansion - prev_expansion

        avg_hand_y = (lw["y"] + rw["y"]) / 2
        avg_sh_y = (ls["y"] + rs["y"]) / 2
        hands_above = avg_hand_y < avg_sh_y

        avg_hand_z = (lw.get("z", 0) + rw.get("z", 0)) / 2
        avg_sh_z = (ls.get("z", 0) + rs.get("z", 0)) / 2
        hands_forward = avg_hand_z < avg_sh_z

        lz_d = lw.get("z", 0) - plw.get("z", 0)
        rz_d = rw.get("z", 0) - prw.get("z", 0)
        avg_z_d = (lz_d + rz_d) / 2
        forward = avg_z_d < fwd_z
        backward = avg_z_d > bwd_z

        up = (lw["y"] - plw["y"]) < -0.01 or (rw["y"] - prw["y"]) < -0.01
        down = (lw["y"] - plw["y"]) > 0.01 or (rw["y"] - prw["y"]) > 0.01

        # Tuned for Authority Coding Sheet (professor's coding)
        is_sudden = avg_vel > 0.08  # Sudden: Punching, Dabbing, Flicking, Slashing
        is_sustained = sus_min < avg_vel <= sus_max  # Sustained: Pressing, Gliding, Floating
        is_strong = body_expansion > 1.15 or hands_above
        is_light = body_expansion < 0.9

        lateral_vel = (abs(lw["x"] - plw["x"]) + abs(rw["x"] - prw["x"])) / 2
        has_lateral = lateral_vel > 0.025  # Stricter for Slashing (professor codes rarely)
        # Upward vs forward: more upward = Floating, more forward = Gliding
        hand_y_d = (lw["y"] - plw["y"] + rw["y"] - prw["y"]) / 2
        hand_z_d = (lz_d + rz_d) / 2
        more_upward = hand_y_d < -0.005 and abs(hand_y_d) > abs(hand_z_d)

        # Punching: forceful forward/downward (Sudden, Strong)
        if is_sudden and is_strong and (forward or down):
            counts["Punching"] += 1
        # Dabbing: short, precise (Sudden, Light)
        if is_sudden and is_light and avg_vel > 0.04:
            counts["Dabbing"] += 1
        # Flicking: outward with rebound
        if is_sudden and is_light and not forward:
            counts["Flicking"] += 1
        # Pressing: downward or forward, sustained (Authority Coding Sheet)
        if is_sustained and is_strong and (down or forward):
            counts["Pressing"] += 1
        # Gliding: smooth, sustained, light, often forward
        if is_sustained and is_light and (forward or (up and not more_upward)):
            counts["Gliding"] += 1
        # Floating: light, upward, buoyant
        if is_sustained and is_light and up and more_upward:
            counts["Floating"] += 1
        # Slashing: diagonal/horizontal sweeping - lateral must dominate (not primarily forward)
        if is_sudden and has_lateral and body_expansion > 1.2 and not forward:
            counts["Slashing"] += 1
        # Wringing: twisting, contracting (arms close or spiral)
        if (body_expansion < enc_max or expansion_delta < -0.02) and avg_vel > enc_min_vel:
            counts["Wringing"] += 1

    return counts


# Legacy motion types (original: Advancing, Retreating, Enclosing, Spreading, Directing, Indirecting)
LEGACY_MOTION_TYPES = ["Advancing", "Retreating", "Enclosing", "Spreading", "Directing", "Indirecting"]


def _detect_legacy_motion_counts_for_frames(frames: List[dict]) -> Dict[str, int]:
    """Count legacy motion type occurrences (Advancing, Retreating, Enclosing, Spreading, Directing, Indirecting)."""
    p = TUNE_PARAMS
    enc_max = p.get("enclosing_max_expansion", ENCLOSING_MAX_EXPANSION)
    enc_min_vel = p.get("enclosing_min_velocity", ENCLOSING_MIN_VELOCITY)
    spread_exp = p.get("spreading_expansion_threshold", SPREADING_BODY_EXPANSION_THRESHOLD)
    spread_min_vel = p.get("spreading_min_velocity", SPREADING_MIN_VELOCITY)
    fwd_z = p.get("forward_z_threshold", -0.03)
    bwd_z = p.get("backward_z_threshold", 0.02)
    adv_vel = p.get("advancing_min_velocity", 0.05)
    ret_vel = p.get("retreating_min_velocity", 0.025)
    ind_vel = p.get("indirecting_min_velocity", 0.04)
    ind_exp = p.get("indirecting_min_expansion", 1.0)
    ind_lateral = p.get("indirecting_lateral_threshold", 0.001)
    sus_min = p.get("sustained_vel_min", 0.02)
    sus_max = p.get("sustained_vel_max", 0.09)

    counts = {t: 0 for t in LEGACY_MOTION_TYPES}
    for i in range(1, len(frames)):
        curr = frames[i].get("landmarks", [])
        prev = frames[i - 1].get("landmarks", [])
        if len(curr) < 17 or len(prev) < 17:
            continue

        lw, rw = curr[LEFT_WRIST], curr[RIGHT_WRIST]
        plw, prw = prev[LEFT_WRIST], prev[RIGHT_WRIST]
        ls, rs = curr[LEFT_SHOULDER], curr[RIGHT_SHOULDER]
        if not all(_visible(p) for p in [lw, rw, plw, prw, ls, rs]):
            continue

        lw_vel = math.sqrt((lw["x"] - plw["x"]) ** 2 + (lw["y"] - plw["y"]) ** 2 + (lw.get("z", 0) - plw.get("z", 0)) ** 2)
        rw_vel = math.sqrt((rw["x"] - prw["x"]) ** 2 + (rw["y"] - prw["y"]) ** 2 + (rw.get("z", 0) - prw.get("z", 0)) ** 2)
        avg_vel = (lw_vel + rw_vel) / 2

        wrist_dist = abs(lw["x"] - rw["x"])
        shoulder_width = max(abs(ls["x"] - rs["x"]), 0.1)
        body_expansion = wrist_dist / shoulder_width

        pwrist = abs(plw["x"] - prw["x"])
        pshoulder = max(abs(prev[LEFT_SHOULDER]["x"] - prev[RIGHT_SHOULDER]["x"]), 0.1)
        prev_expansion = pwrist / pshoulder
        expansion_delta = body_expansion - prev_expansion

        avg_hand_z = (lw.get("z", 0) + rw.get("z", 0)) / 2
        avg_sh_z = (ls.get("z", 0) + rs.get("z", 0)) / 2
        hands_forward = avg_hand_z < avg_sh_z

        lz_d = lw.get("z", 0) - plw.get("z", 0)
        rz_d = rw.get("z", 0) - prw.get("z", 0)
        avg_z_d = (lz_d + rz_d) / 2
        forward = avg_z_d < fwd_z
        backward = avg_z_d > bwd_z

        is_sustained = sus_min < avg_vel <= sus_max
        lateral_vel = (abs(lw["x"] - plw["x"]) + abs(rw["x"] - prw["x"])) / 2
        has_lateral = lateral_vel > ind_lateral

        if (body_expansion < enc_max or expansion_delta < -0.02) and avg_vel > enc_min_vel:
            counts["Enclosing"] += 1
        if (body_expansion > spread_exp or expansion_delta > 0.02) and avg_vel > spread_min_vel:
            counts["Spreading"] += 1
        if hands_forward and is_sustained and forward:
            counts["Directing"] += 1
        if (not hands_forward or has_lateral) and avg_vel > ind_vel and body_expansion > ind_exp:
            counts["Indirecting"] += 1
        if forward and avg_vel > adv_vel and is_sustained:
            counts["Advancing"] += 1
        if backward and avg_vel > ret_vel and is_sustained:
            counts["Retreating"] += 1

    return counts


# Motion types from Complete_Effort_Action_Motion_Descriptions.csv
MOTION_TYPES_CSV = ["Gliding", "Floating", "Punching", "Dabbing", "Flicking", "Slashing", "Wringing", "Pressing"]


def analyze_legacy_motion_per_second(frames: List[dict]) -> List[Dict[str, Any]]:
    """
    Legacy motion types per second: Advancing, Retreating, Enclosing, Spreading, Directing, Indirecting.
    Returns list of {second, motions: [{motion_type, count, confidence}]} per second.
    """
    if len(frames) < 2:
        return []

    by_second: Dict[int, List[dict]] = defaultdict(list)
    for f in frames:
        ts = f.get("timestamp", 0)
        sec = int(ts)
        by_second[sec].append(f)

    result = []
    for sec in sorted(by_second.keys()):
        window = by_second[sec]
        if len(window) < 2:
            result.append({"second": sec, "motions": []})
            continue

        counts = _detect_legacy_motion_counts_for_frames(window)
        n_frames = len(window) - 1
        if n_frames <= 0:
            result.append({"second": sec, "motions": [{"motion_type": "Neutral", "count": 0, "confidence": 0}]})
            continue

        aggregated = defaultdict(int)
        for motion_type, count in counts.items():
            if count <= 0:
                continue
            aggregated[motion_type] += count

        if not aggregated:
            result.append({"second": sec, "motions": [{"motion_type": "Neutral", "count": 0, "confidence": 0}]})
            continue

        motion_list = []
        for motion_type, count in aggregated.items():
            confidence = round(count / n_frames * 100, 1)
            motion_list.append({"motion_type": motion_type, "count": count, "confidence": confidence})
        motion_list.sort(key=lambda x: -x["count"])
        result.append({"second": sec, "motions": motion_list})

    return result


def analyze_motion_per_second(frames: List[dict]) -> List[Dict[str, Any]]:
    """
    Integrated movement tracking: analyze ALL motion types for each 1-second window.
    Returns list of {second, motions: [{motion_type, count, confidence}]} per second.
    """
    if len(frames) < 2:
        return []

    # Group frames by second (floor of timestamp)
    by_second: Dict[int, List[dict]] = defaultdict(list)
    for f in frames:
        ts = f.get("timestamp", 0)
        sec = int(ts)
        by_second[sec].append(f)

    result = []
    for sec in sorted(by_second.keys()):
        window = by_second[sec]
        if len(window) < 2:
            result.append({"second": sec, "motions": []})
            continue

        counts = _detect_motion_counts_for_frames(window)
        n_frames = len(window) - 1  # frame pairs

        if n_frames <= 0:
            result.append({
                "second": sec,
                "motions": [{"motion_type": "Neutral", "count": 0, "confidence": 0}],
            })
            continue

        # Aggregate by motion type (already using CSV types)
        aggregated = defaultdict(int)
        for motion_type, count in counts.items():
            if count <= 0:
                continue
            aggregated[motion_type] += count

        if not aggregated:
            result.append({"second": sec, "motions": [{"motion_type": "Neutral", "count": 0, "confidence": 0}]})
            continue

        # Per-motion confidence: % of frame pairs with this motion (Integrated Movement: multiple motions can co-occur)
        motion_list = []
        for motion_type, count in aggregated.items():
            confidence = round(count / n_frames * 100, 1)
            motion_list.append({"motion_type": motion_type, "count": count, "confidence": confidence})
        motion_list.sort(key=lambda x: -x["count"])

        result.append({
            "second": sec,
            "motions": motion_list,
        })

    return result


# Laban Effort factors derived from Complete_Effort_Action_Motion_Descriptions.csv
# Weight: Light (Gliding, Floating, Dabbing, Flicking) | Strong (Punching, Slashing, Wringing, Pressing)
# Time: Sustained (Gliding, Floating, Wringing, Pressing) | Sudden (Punching, Dabbing, Flicking, Slashing)
# Space: Direct (Gliding, Punching, Dabbing, Pressing) | Indirect (Floating, Flicking, Slashing, Wringing)
# Flow: Free (Gliding, Floating, Flicking) | Bound (Punching, Dabbing, Slashing, Wringing, Pressing)
EFFORT_LIGHT = {"Gliding", "Floating", "Dabbing", "Flicking"}
EFFORT_STRONG = {"Punching", "Slashing", "Wringing", "Pressing"}
EFFORT_SUSTAINED = {"Gliding", "Floating", "Wringing", "Pressing"}
EFFORT_SUDDEN = {"Punching", "Dabbing", "Flicking", "Slashing"}
EFFORT_DIRECT = {"Gliding", "Punching", "Dabbing", "Pressing"}
EFFORT_INDIRECT = {"Floating", "Flicking", "Slashing", "Wringing"}
EFFORT_FREE = {"Gliding", "Floating", "Flicking"}
EFFORT_BOUND = {"Punching", "Dabbing", "Slashing", "Wringing", "Pressing"}


def _effort_from_motions(motions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Derive effort factor confidences (0-100) from motion list."""
    conf = {m["motion_type"]: m["confidence"] for m in (motions or []) if m.get("motion_type") != "Neutral"}
    out: Dict[str, float] = {}
    for name, motion_set in [
        ("Light", EFFORT_LIGHT),
        ("Strong", EFFORT_STRONG),
        ("Sustained", EFFORT_SUSTAINED),
        ("Sudden", EFFORT_SUDDEN),
        ("Direct", EFFORT_DIRECT),
        ("Indirect", EFFORT_INDIRECT),
        ("Free", EFFORT_FREE),
        ("Bound", EFFORT_BOUND),
    ]:
        total = sum(conf.get(m, 0) for m in motion_set)
        out[name] = round(min(100.0, total), 1)
    return out


def analyze_effort_per_second(motion_per_second: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Derive effort factor confidence over time from motion_per_second.
    Returns list of {second, efforts: [{effort_type, confidence}]} per second.
    """
    result = []
    for row in motion_per_second:
        efforts = _effort_from_motions(row.get("motions", []))
        result.append({
            "second": row["second"],
            "efforts": [{"effort_type": k, "confidence": v} for k, v in efforts.items()],
        })
    return result


# Movement Combination Summary.csv: subgroups with required motion types
_COMBO_MOTION_TYPES = [
    "Pressing", "Punching", "Slashing", "Gliding",
    "Enclosing", "Spreading", "Directing", "Indirecting", "Advancing", "Retreating",
]


def _load_subgroup_definitions() -> List[Dict[str, Any]]:
    """Load Movement Combination Summary.csv. Returns [{subgroup, required: [motion_types]}]."""
    if not _MOVEMENT_COMBO_CSV.exists():
        return []
    result = []
    with open(_MOVEMENT_COMBO_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subgroup = (row.get("Subgroup") or "").strip()
            if not subgroup:
                continue
            required = [
                m for m in _COMBO_MOTION_TYPES
                if m in row and str(row.get(m, "")).strip() in ("1", "x", "X")
            ]
            result.append({"subgroup": subgroup, "required": required})
    return result


def _merged_motion_confidences(
    legacy_row: Dict[str, Any],
    motion_row: Dict[str, Any],
) -> Dict[str, float]:
    """Merge legacy + motion confidences for combo types. Returns {motion_type: confidence}."""
    out: Dict[str, float] = {m: 0.0 for m in _COMBO_MOTION_TYPES}
    for m in legacy_row.get("motions", []):
        t = m.get("motion_type")
        if t in out:
            out[t] = max(out[t], float(m.get("confidence", 0)))
    for m in motion_row.get("motions", []):
        t = m.get("motion_type")
        if t in out:
            out[t] = max(out[t], float(m.get("confidence", 0)))
    return out


def analyze_subgroup_per_second(
    legacy_motion_per_second: List[Dict[str, Any]],
    motion_per_second: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Match per-second motion combinations to subgroups from Movement Combination Summary.csv.
    Returns list of {second, subgroups: [{subgroup, confidence}]} per second.
    """
    definitions = _load_subgroup_definitions()
    if not definitions:
        return []

    # Index by second
    legacy_by_sec = {r["second"]: r for r in legacy_motion_per_second}
    motion_by_sec = {r["second"]: r for r in motion_per_second}
    all_seconds = sorted(set(legacy_by_sec.keys()) | set(motion_by_sec.keys()))

    result = []
    for sec in all_seconds:
        legacy_row = legacy_by_sec.get(sec, {"motions": []})
        motion_row = motion_by_sec.get(sec, {"motions": []})
        merged = _merged_motion_confidences(legacy_row, motion_row)

        subgroups = []
        for defn in definitions:
            required = defn["required"]
            if not required:
                continue
            # Confidence = min of required motion confidences (match strength)
            confs = [merged.get(m, 0) for m in required]
            confidence = min(confs) if confs else 0
            if confidence > 0:
                subgroups.append({"subgroup": defn["subgroup"], "confidence": round(confidence, 1)})

        subgroups.sort(key=lambda x: -x["confidence"])
        result.append({
            "second": sec,
            "subgroups": subgroups,
            "adv_confidence": round(merged.get("Advancing", 0), 1),
            "ret_confidence": round(merged.get("Retreating", 0), 1),
            "pressing_confidence": round(merged.get("Pressing", 0), 1),
            "punching_confidence": round(merged.get("Punching", 0), 1),
        })

    return result


# Group mapping: A=Authority, C=Confidence, E=Engaging
_GROUP_SUBGROUPS = {
    "Authority": ["A1", "A2", "A4", "A5"],
    "Confidence": ["C1", "C2", "C3", "C4", "C5", "C7", "C8"],
    "Engaging": ["E1", "E2", "E3", "E4", "E5", "E6", "E7"],
}


def _scale_from_percentage(pct: float) -> str:
    """Scale: Low < 5%, Moderate >= 5% and < 10%, High >= 10%."""
    if pct < 5.0:
        return "low"
    if pct < 10.0:
        return "moderate"
    return "high"


def compute_movement_summary(subgroup_per_second: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summary table: count combinations per second, percentage, scale per group.
    Returns {total_seconds, subgroups: [{subgroup, count}], groups: [{name, count, percentage, scale}]}.
    """
    if not subgroup_per_second:
        return {"total_seconds": 0, "subgroups": [], "groups": []}

    total_seconds = max(r["second"] for r in subgroup_per_second) + 1
    if total_seconds <= 0:
        total_seconds = 1

    # Count seconds per subgroup
    subgroup_counts: Dict[str, int] = defaultdict(int)
    group_seconds: Dict[str, set] = defaultdict(set)  # group -> set of seconds

    # Subgroups that require Advancing/Retreating/Pressing/Punching: count only if motion confidence meets threshold
    definitions = _load_subgroup_definitions()
    subgroups_requiring_adv = {d["subgroup"] for d in definitions if d.get("required") and "Advancing" in d["required"]}
    subgroups_requiring_ret = {d["subgroup"] for d in definitions if d.get("required") and "Retreating" in d["required"]}
    subgroups_requiring_pressing = {d["subgroup"] for d in definitions if d.get("required") and "Pressing" in d["required"]}
    subgroups_requiring_punching = {d["subgroup"] for d in definitions if d.get("required") and "Punching" in d["required"]}

    for row in subgroup_per_second:
        sec = row["second"]
        adv_conf = row.get("adv_confidence", 0)
        ret_conf = row.get("ret_confidence", 0)
        pressing_conf = row.get("pressing_confidence", 0)
        punching_conf = row.get("punching_confidence", 0)
        for s in row.get("subgroups", []):
            sg = s.get("subgroup", "")
            if not sg:
                continue
            conf = s.get("confidence", 0)
            # Subgroups requiring Advancing: count only if Advancing confidence > 20%
            if sg in subgroups_requiring_adv and adv_conf <= 20:
                continue
            # Subgroups requiring Retreating: count only if Retreating confidence > 50%
            if sg in subgroups_requiring_ret and ret_conf <= 50:
                continue
            # Subgroups requiring Pressing: count only if Pressing confidence > 5%
            if sg in subgroups_requiring_pressing and pressing_conf <= 5:
                continue
            # Subgroups requiring Punching: count only if Punching confidence > 5%
            if sg in subgroups_requiring_punching and punching_conf <= 5:
                continue
            # Other subgroups: count only if subgroup confidence > 10%
            special_motion_subgroups = subgroups_requiring_adv | subgroups_requiring_ret | subgroups_requiring_pressing | subgroups_requiring_punching
            if sg not in special_motion_subgroups and conf <= 10:
                continue
            # Subgroups with special motions: still need subgroup confidence > 10%
            if conf <= 10:
                continue
            subgroup_counts[sg] += 1
            for group_name, subs in _GROUP_SUBGROUPS.items():
                if sg in subs:
                    group_seconds[group_name].add(sec)

    subgroups_list = [
        {"subgroup": sg, "count": subgroup_counts[sg]}
        for sg in sorted(subgroup_counts.keys())
    ]

    groups_list = []
    for group_name in ["Authority", "Confidence", "Engaging"]:
        count = len(group_seconds.get(group_name, set()))
        pct = round(count / total_seconds * 100, 1)
        scale = _scale_from_percentage(pct)
        groups_list.append({
            "name": group_name,
            "count": count,
            "total_seconds": total_seconds,
            "percentage": pct,
            "scale": scale,
        })

    return {
        "total_seconds": total_seconds,
        "total_subgroups": len([d for d in _load_subgroup_definitions() if d.get("required")]),
        "subgroups": subgroups_list,
        "groups": groups_list,
    }


def analyze_gesture_effort(frames: List[dict]) -> Dict[str, Any]:
    """Matches report_core.analyze_video_mediapipe effort/shape detection and scoring."""
    if len(frames) < 2:
        return {
            "engaging_score": 4, "convince_score": 4, "authority_score": 4,
            "effort_detection": {}, "shape_detection": {},
            "engaging_pos": 257, "convince_pos": 271, "authority_pos": 254,
        }

    effort_counts = {t: 0 for t in MOTION_TYPES_CSV}
    shape_counts = {t: 0 for t in MOTION_TYPES_CSV}

    for i in range(1, len(frames)):
        curr = frames[i].get("landmarks", [])
        prev = frames[i - 1].get("landmarks", [])
        if len(curr) < 17 or len(prev) < 17:
            continue

        lw, rw = curr[LEFT_WRIST], curr[RIGHT_WRIST]
        plw, prw = prev[LEFT_WRIST], prev[RIGHT_WRIST]
        ls, rs = curr[LEFT_SHOULDER], curr[RIGHT_SHOULDER]

        if not all(_visible(p) for p in [lw, rw, plw, prw, ls, rs]):
            continue

        lw_vel = math.sqrt((lw["x"] - plw["x"]) ** 2 + (lw["y"] - plw["y"]) ** 2 + (lw.get("z", 0) - plw.get("z", 0)) ** 2)
        rw_vel = math.sqrt((rw["x"] - prw["x"]) ** 2 + (rw["y"] - prw["y"]) ** 2 + (rw.get("z", 0) - prw.get("z", 0)) ** 2)
        avg_vel = (lw_vel + rw_vel) / 2

        wrist_dist = abs(lw["x"] - rw["x"])
        shoulder_width = max(abs(ls["x"] - rs["x"]), 0.1)
        body_expansion = wrist_dist / shoulder_width

        avg_hand_y = (lw["y"] + rw["y"]) / 2
        avg_sh_y = (ls["y"] + rs["y"]) / 2
        hands_above = avg_hand_y < avg_sh_y

        avg_hand_z = (lw.get("z", 0) + rw.get("z", 0)) / 2
        avg_sh_z = (ls.get("z", 0) + rs.get("z", 0)) / 2
        hands_forward = avg_hand_z < avg_sh_z

        lz_d = lw.get("z", 0) - plw.get("z", 0)
        rz_d = rw.get("z", 0) - prw.get("z", 0)
        avg_z_d = (lz_d + rz_d) / 2
        forward = avg_z_d < -0.03
        backward = avg_z_d > 0.05

        up = (lw["y"] - plw["y"]) < -0.01 or (rw["y"] - prw["y"]) < -0.01
        down = (lw["y"] - plw["y"]) > 0.01 or (rw["y"] - prw["y"]) > 0.01

        is_sudden = avg_vel > 0.08
        is_sustained = 0.02 < avg_vel <= 0.08
        is_strong = body_expansion > 1.2 or hands_above
        is_light = body_expansion < 0.8

        lateral_vel = (abs(lw["x"] - plw["x"]) + abs(rw["x"] - prw["x"])) / 2
        has_lateral = lateral_vel > 0.001
        hand_y_d = (lw["y"] - plw["y"] + rw["y"] - prw["y"]) / 2
        more_upward = hand_y_d < -0.005 and abs(hand_y_d) > abs(avg_z_d)

        if is_sudden and is_strong and forward:
            effort_counts["Punching"] += 1
        if is_sudden and is_light and avg_vel > 0.05:
            effort_counts["Dabbing"] += 1
        if is_sudden and is_light and not forward:
            effort_counts["Flicking"] += 1
        if is_sustained and is_strong and down:
            effort_counts["Pressing"] += 1
            shape_counts["Pressing"] += 1
        if is_sustained and is_light and up and not more_upward:
            effort_counts["Gliding"] += 1
            shape_counts["Gliding"] += 1
        if is_sustained and is_light and up and more_upward:
            effort_counts["Floating"] += 1
            shape_counts["Floating"] += 1
        if is_sudden and has_lateral and body_expansion > 1.2:
            effort_counts["Slashing"] += 1
            shape_counts["Slashing"] += 1
        if body_expansion < ENCLOSING_MAX_EXPANSION and avg_vel > ENCLOSING_MIN_VELOCITY:
            effort_counts["Wringing"] += 1
            shape_counts["Wringing"] += 1

    total_det = max(1, sum(effort_counts.values()))
    effort_detection = {k: round(v / total_det * 100, 1) for k, v in effort_counts.items()}
    effort_detection = _apply_default_retreating_share(effort_detection, retreat_default_pct=1.0)

    total_shape = max(1, sum(shape_counts.values()))
    shape_detection = {k: round(v / total_shape * 100, 1) for k, v in shape_counts.items()}
    shape_detection = _apply_default_retreating_share(shape_detection, retreat_default_pct=1.0)

    analyzed = len(frames) - 1
    dominant_share = max(effort_detection.values()) / 100.0 if effort_detection else 0.0
    variety_floor = max(2, int(max(1, analyzed) * 0.03))
    variety_count = sum(1 for v in effort_counts.values() if v >= variety_floor)
    monotony_penalty = max(0.0, (dominant_share - 0.35) * 8.0)
    variety_factor = max(0.85, min(1.20, 0.85 + (variety_count / 8.0) * 0.35))

    total_f = max(1, analyzed)
    engaging_activity = (
        effort_counts.get("Slashing", 0) + effort_counts.get("Wringing", 0)
        + effort_counts.get("Gliding", 0) + effort_counts.get("Flicking", 0) + effort_counts.get("Floating", 0)
    ) / total_f
    confidence_activity = (
        effort_counts.get("Punching", 0) + effort_counts.get("Pressing", 0)
    ) / total_f
    authority_activity = (
        effort_counts.get("Pressing", 0) + effort_counts.get("Punching", 0)
    ) / total_f

    engaging_boost = max(0.85, min(1.35, 0.85 + engaging_activity * 1.60))
    confidence_boost = max(0.85, min(1.35, 0.85 + confidence_activity * 1.50))
    authority_boost = max(0.85, min(1.35, 0.85 + authority_activity * 1.50))

    engaging_raw = (
        effort_detection.get("Slashing", 0) * 0.34 + effort_detection.get("Wringing", 0) * 0.26
        + effort_detection.get("Gliding", 0) * 0.22 + effort_detection.get("Flicking", 0) * 0.18
        - effort_detection.get("Punching", 0) * 0.12
    )
    confidence_raw = (
        effort_detection.get("Punching", 0) * 0.50 + effort_detection.get("Pressing", 0) * 0.50
        - effort_detection.get("Flicking", 0) * 0.08
    )
    authority_raw = (
        effort_detection.get("Pressing", 0) * 0.50 + effort_detection.get("Punching", 0) * 0.50
        - effort_detection.get("Flicking", 0) * 0.18
    )

    engaging_shape = (
        shape_detection.get("Slashing", 0) * 0.45 + shape_detection.get("Wringing", 0) * 0.35
        + shape_detection.get("Flicking", 0) * 0.20
    )
    confidence_shape = shape_detection.get("Pressing", 0) * 0.60 + shape_detection.get("Punching", 0) * 0.40
    authority_shape = shape_detection.get("Pressing", 0) * 0.55 + shape_detection.get("Punching", 0) * 0.45

    engaging_raw = max(0.0, (engaging_raw * engaging_boost + engaging_shape * 0.22) * variety_factor - monotony_penalty * 0.90)
    confidence_raw = max(0.0, (confidence_raw * confidence_boost + confidence_shape * 0.25) * variety_factor - monotony_penalty * 1.00)
    authority_raw = max(0.0, (authority_raw * authority_boost + authority_shape * 0.25) * variety_factor - monotony_penalty * 1.10)

    raw_vec = np.array([engaging_raw, confidence_raw, authority_raw], dtype=float)
    raw_mean = float(np.mean(raw_vec))
    contrast = 0.95
    engaging_raw = engaging_raw + contrast * (engaging_raw - raw_mean)
    confidence_raw = confidence_raw + contrast * (confidence_raw - raw_mean)
    authority_raw = authority_raw + contrast * (authority_raw - raw_mean)

    total_indicators = 450 + 475 + 445
    engaging_score = min(7, max(1, round(engaging_raw / 5.4 + 1.4)))
    convince_score = min(7, max(1, round(confidence_raw / 5.4 + 1.4)))
    authority_score = min(7, max(1, round(authority_raw / 5.4 + 1.4)))

    return {
        "engaging_score": engaging_score,
        "convince_score": convince_score,
        "authority_score": authority_score,
        "engaging_pos": int(engaging_score / 7 * 450),
        "convince_pos": int(convince_score / 7 * 475),
        "authority_pos": int(authority_score / 7 * 445),
        "effort_detection": effort_detection,
        "shape_detection": shape_detection,
        "total_indicators": total_indicators,
    }


def _first_impression_level(value: float, metric: str = "") -> str:
    if str(metric or "").strip().lower() == "eye_contact":
        return "High"
    v = float(value or 0.0)
    if v >= 75.0:
        return "High"
    if v >= 40.0:
        return "Moderate"
    return "Low"


def generate_eye_contact_text(pct: float) -> list:
    """Exact text from report_core — 10 levels."""
    if pct >= 90:
        return [
            "Your eye contact is outstanding — consistently steady, warm, and completely audience-focused throughout the entire presentation.",
            "You maintain unwavering direct gaze during key moments, which maximizes trust, clarity, and audience engagement.",
            "Every gaze shift is purposeful, natural, and enhances your communication (e.g., thinking pauses, emphasis).",
            "Your eye contact is a masterclass in confidence and credibility, with zero signs of avoidance or nervousness.",
        ]
    elif pct >= 80:
        return [
            "Your eye contact is excellent — steady, engaging, and audience-focused almost throughout the presentation.",
            "You consistently maintain direct gaze during key moments, which strongly increases trust and clarity.",
            "Gaze shifts are purposeful and natural, never detracting from your professional presence.",
            "Your eye contact demonstrates strong confidence and builds exceptional credibility with the audience.",
        ]
    elif pct >= 70:
        return [
            "Your eye contact is very strong — steady and audience-focused for most of the time.",
            "You maintain direct gaze during key moments effectively, which clearly builds trust and clarity.",
            "Occasional gaze shifts are natural and appropriate, showing thoughtfulness without reducing engagement.",
            "Overall, your eye contact strongly supports confidence and credibility with the audience.",
        ]
    elif pct >= 60:
        return [
            "Your eye contact is strong — generally steady and audience-focused during important moments.",
            "You maintain direct gaze during most key moments, which helps establish trust.",
            "Some gaze shifts occur but they're mostly natural and don't significantly impact your presence.",
            "Your eye contact effectively supports good confidence and reasonable credibility.",
        ]
    elif pct >= 50:
        return [
            "Your eye contact shows a good foundation, with several effective moments of direct audience connection.",
            "You maintain direct gaze during many important points, which helps build trust.",
            "With slightly longer eye contact in key moments, your delivery will feel even more confident.",
            "Overall, your audience engagement is positive and can be strengthened further.",
        ]
    elif pct >= 40:
        return [
            "Your eye contact is developing well, with regular moments of direct audience connection.",
            "You make direct eye contact consistently, and sustaining it slightly longer will increase impact.",
            "Some gaze shifts occur more frequently than ideal, but this is very trainable.",
            "Improving consistency will noticeably enhance trust and credibility.",
        ]
    elif pct >= 30:
        return [
            "Your eye contact is in a developing stage, with clear opportunities to become more consistent.",
            "You already connect directly with the audience at times, especially in selected moments.",
            "Extending direct gaze a bit longer will strengthen audience connection and engagement.",
            "With focused practice, trust and credibility cues will improve clearly.",
        ]
    else:
        return [
            "Your eye contact is developing and currently inconsistent in several parts of the presentation.",
            "You connect with the audience at times, and extending direct gaze during key moments will improve trust.",
            "Some gaze shifts are more frequent than ideal, but this can be improved with focused practice.",
            "With steadier eye contact, your confidence and credibility will become more apparent.",
        ]


def generate_uprightness_text(pct: float) -> list:
    """Exact text from report_core."""
    if pct >= 90:
        return [
            "You maintain outstanding upright posture with perfect vertical alignment throughout the entire presentation.",
            "Your chest remains fully open, shoulders are optimally relaxed, and head alignment is impeccable — signaling exceptional balance, readiness, and commanding authority.",
            "Even during active gesturing, your vertical alignment stays completely stable, demonstrating masterful core control.",
            "There is absolutely no visible slouching or collapsing at any point, projecting supreme professional presence.",
        ]
    elif pct >= 80:
        return [
            "You maintain excellent upright posture consistently throughout the presentation.",
            "Your chest stays open, shoulders are naturally relaxed, and head alignment is near-perfect — signaling strong balance, readiness, and authority.",
            "Your vertical alignment remains remarkably stable even when gesturing, demonstrating superior core control.",
            "There is virtually no slouching or collapsing, which projects a highly professional and confident appearance.",
        ]
    elif pct >= 70:
        return [
            "You maintain very strong upright posture throughout most of the presentation.",
            "The chest stays open, shoulders remain relaxed, and head is well-aligned — clearly signaling balance, readiness, and authority.",
            "Your vertical alignment stays stable during most gestures, showing very good core control.",
            "Minimal slouching occurs, which strongly supports a professional and composed appearance.",
        ]
    elif pct >= 60:
        return [
            "You maintain strong upright posture for the majority of the clip.",
            "Your chest generally stays open, shoulders relaxed, and head aligned — signaling good balance and authority.",
            "Vertical alignment remains fairly stable during gestures, showing solid core control.",
            "Occasional minor slouching appears but doesn't detract significantly from your professional presence.",
        ]
    elif pct >= 50:
        return [
            "Your posture has a good foundation, with many moments of stable upright alignment.",
            "Chest and shoulder alignment are generally positive, with room to make consistency even better.",
            "During gestures, vertical alignment is mostly stable and supports clear communication.",
            "Overall posture is strong and can be elevated further with small refinements.",
        ]
    elif pct >= 40:
        return [
            "Your posture is developing well, with a number of solid upright moments throughout the presentation.",
            "Chest and shoulder positioning are often acceptable, and greater consistency will add polish.",
            "Vertical alignment varies in some gesture phases, which is common and trainable.",
            "With steadier posture, your professional image will look even more confident.",
        ]
    elif pct >= 30:
        return [
            "Your posture is in a developing stage, alternating between upright moments and less stable alignment.",
            "There are several points where shoulder and head position can be brought closer to ideal alignment.",
            "Improving vertical consistency will make your presence feel stronger and more composed.",
            "With ongoing practice, professional presence and authority will improve clearly.",
        ]
    else:
        return [
            "Your posture is still developing and varies between upright and less stable alignment.",
            "There are several moments where shoulder and head alignment drift from ideal posture.",
            "Improving core control and maintaining a taller stance will enhance your professional presence.",
            "With consistent practice, your posture can become more stable and confident.",
        ]


def generate_stance_text(stability: float) -> list:
    """Exact text from report_core."""
    if stability >= 90:
        return [
            "Your stance is outstanding — perfectly symmetrical and solidly grounded, with feet placed optimally at shoulder-width apart.",
            "Weight shifts are virtually non-existent, creating an exceptionally stable platform that demonstrates supreme confidence.",
            "You maintain flawless forward orientation toward the audience throughout, maximally reinforcing clarity and engagement.",
            "Your stance conveys rock-solid stability and commanding, welcoming authority ideal for executive leadership communication.",
        ]
    elif stability >= 80:
        return [
            "Your stance is excellent — exceptionally symmetrical and grounded, with feet placed perfectly about shoulder-width apart.",
            "Weight shifts are extremely controlled and minimal, preventing any distraction and demonstrating strong confidence.",
            "You maintain superior forward orientation toward the audience throughout, reinforcing excellent clarity and engagement.",
            "Your stance conveys remarkable stability and a welcoming, authoritative presence ideal for professional leadership.",
        ]
    elif stability >= 70:
        return [
            "Your stance is very strong — highly symmetrical and well-grounded, with feet well-positioned shoulder-width apart.",
            "Weight shifts are well-controlled and quite minimal, showing strong confidence and solid balance.",
            "You maintain very good forward orientation toward the audience, clearly reinforcing engagement and clarity.",
            "The stance conveys strong stability and a professional, welcoming presence suitable for senior communication roles.",
        ]
    elif stability >= 60:
        return [
            "Your stance is strong — symmetrical and grounded, with feet appropriately placed about shoulder-width apart.",
            "Weight shifts are controlled and minimal, preventing distraction and demonstrating good confidence.",
            "You maintain solid forward orientation toward the audience, reinforcing clarity and reasonable engagement.",
            "The stance effectively conveys stability and a professional presence suitable for most business communication.",
        ]
    elif stability >= 50:
        return [
            "Your stance has a good base, with generally stable positioning and only occasional weight shifts.",
            "Feet placement is mostly appropriate, supporting balanced delivery through much of the presentation.",
            "Weight distribution is mostly steady, with minor adjustments that are common and manageable.",
            "Overall stance supports a positive professional presence and can be refined further.",
        ]
    elif stability >= 40:
        return [
            "Your stance is developing well, with many steady moments across the presentation.",
            "Feet placement is generally acceptable, and greater consistency will improve grounding.",
            "Some visible shifts occur in parts of the clip, which is normal and can be reduced with practice.",
            "Better stability will further strengthen your grounded presence and authority.",
        ]
    elif stability >= 30:
        return [
            "Your stance is in a developing stage, with opportunities to become steadier throughout delivery.",
            "Feet placement varies at times, and improving consistency will support better balance.",
            "Some swaying or shifting appears in parts of the presentation, and this can be improved with focused drills.",
            "Enhancing stance stability will clearly strengthen your professional presence.",
        ]
    else:
        return [
            "Your stance is still developing, with noticeable movement and balance variation in several moments.",
            "Feet placement and weight distribution are not yet fully consistent throughout the presentation.",
            "Reducing unnecessary shifts will help you appear more grounded and authoritative.",
            "With regular practice, your stance can become steadier and more professional.",
        ]


def run_analysis(frames: List[dict]) -> Dict[str, Any]:
    """Run full analysis matching report_core + report_worker format."""
    first = analyze_first_impression(frames)
    effort = analyze_gesture_effort(frames)
    legacy_motion_per_second = analyze_legacy_motion_per_second(frames)
    motion_per_second = analyze_motion_per_second(frames)
    effort_per_second = analyze_effort_per_second(motion_per_second)
    subgroup_per_second = analyze_subgroup_per_second(legacy_motion_per_second, motion_per_second)
    movement_summary = compute_movement_summary(subgroup_per_second)

    total = int(effort.get("total_indicators") or 1370)
    engaging_score = int(effort.get("engaging_score") or 4)
    convince_score = int(effort.get("convince_score") or 4)
    authority_score = int(effort.get("authority_score") or 4)

    def _scale(score: int) -> str:
        if score in (3, 4):
            return "moderate"
        if score >= 5:
            return "high"
        return "low"

    engaging_pos = int(effort.get("engaging_pos") or int(engaging_score / 7 * 450))
    convince_pos = int(effort.get("convince_pos") or int(convince_score / 7 * 475))
    authority_pos = int(effort.get("authority_pos") or int(authority_score / 7 * 445))

    categories = [
        {
            "name_en": "Engaging & Connecting",
            "name_th": "การสร้างความเป็นมิตรและสร้างสัมพันธภาพ",
            "score": engaging_score,
            "scale": _scale(engaging_score),
            "positives": engaging_pos,
            "total": total,
            "description": f"Detected {engaging_pos} positive indicators out of {total} total indicators",
        },
        {
            "name_en": "Confidence",
            "name_th": "ความมั่นใจ",
            "score": convince_score,
            "scale": _scale(convince_score),
            "positives": convince_pos,
            "total": total,
            "description": f"Detected {convince_pos} positive indicators out of {total} total indicators",
        },
        {
            "name_en": "Authority",
            "name_th": "ความเป็นผู้นำและอำนาจ",
            "score": authority_score,
            "scale": _scale(authority_score),
            "positives": authority_pos,
            "total": total,
            "description": f"Detected {authority_pos} positive indicators out of {total} total indicators",
        },
    ]

    return {
        "first_impression": {
            "eye_contact": first["eye_contact"],
            "eye_contact_label": _first_impression_level(first["eye_contact"], "eye_contact"),
            "eye_contact_text": generate_eye_contact_text(first["eye_contact"]),
            "uprightness": first["uprightness"],
            "uprightness_label": _first_impression_level(first["uprightness"], "uprightness"),
            "uprightness_text": generate_uprightness_text(first["uprightness"]),
            "stance": first["stance"],
            "stance_label": _first_impression_level(first["stance"], "stance"),
            "stance_text": generate_stance_text(first["stance"]),
        },
        "categories": categories,
        "engagement": {
            "score": engaging_score,
            "label": _scale(engaging_score).capitalize(),
            "text": categories[0]["description"],
        },
        "confidence": {
            "score": convince_score,
            "label": _scale(convince_score).capitalize(),
            "text": categories[1]["description"],
        },
        "authority": {
            "score": authority_score,
            "label": _scale(authority_score).capitalize(),
            "text": categories[2]["description"],
        },
        "effort_detection": effort.get("effort_detection", {}),
        "shape_detection": effort.get("shape_detection", {}),
        "legacy_motion_per_second": legacy_motion_per_second,
        "motion_per_second": motion_per_second,
        "effort_per_second": effort_per_second,
        "subgroup_per_second": subgroup_per_second,
        "movement_summary": movement_summary,
    }
