"""Debug: print per-frame metrics for first 2 seconds to understand GT vs algorithm."""
import json
import math
from pathlib import Path

SKELETON_PATH = Path(__file__).parent / "debug_skeleton_cache" / "f26c6a522f4c62e8.json"
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

def _visible(lm):
    return lm and (abs(lm.get("x",0)) > 1e-6 or abs(lm.get("y",0)) > 1e-6)

with open(SKELETON_PATH) as f:
    frames = json.load(f)["frames"]

print("Frame | time | body_exp | avg_vel | exp_delta | lat_vel | avg_z_d | fwd | bwd | enc | spread | direct | indirect | adv | ret")
for i in range(1, min(65, len(frames))):  # First ~2 sec at 30fps
    curr = frames[i]["landmarks"]
    prev = frames[i-1]["landmarks"]
    if len(curr) < 17:
        continue
    lw, rw = curr[LEFT_WRIST], curr[RIGHT_WRIST]
    plw, prw = prev[LEFT_WRIST], prev[RIGHT_WRIST]
    ls, rs = curr[LEFT_SHOULDER], curr[RIGHT_SHOULDER]
    if not all(_visible(p) for p in [lw, rw, plw, prw, ls, rs]):
        continue

    lw_vel = math.sqrt((lw["x"]-plw["x"])**2 + (lw["y"]-plw["y"])**2 + (lw.get("z",0)-plw.get("z",0))**2)
    rw_vel = math.sqrt((rw["x"]-prw["x"])**2 + (rw["y"]-prw["y"])**2 + (rw.get("z",0)-prw.get("z",0))**2)
    avg_vel = (lw_vel + rw_vel) / 2

    wrist_dist = abs(lw["x"] - rw["x"])
    shoulder_width = max(abs(ls["x"] - rs["x"]), 0.1)
    body_exp = wrist_dist / shoulder_width

    pwrist = abs(plw["x"] - prw["x"])
    pshoulder = max(abs(prev[LEFT_SHOULDER]["x"] - prev[RIGHT_SHOULDER]["x"]), 0.1)
    prev_exp = pwrist / pshoulder
    exp_delta = body_exp - prev_exp

    lateral_vel = (abs(lw["x"] - plw["x"]) + abs(rw["x"] - prw["x"])) / 2

    avg_hand_z = (lw.get("z",0) + rw.get("z",0)) / 2
    avg_sh_z = (ls.get("z",0) + rs.get("z",0)) / 2
    hands_forward = avg_hand_z < avg_sh_z

    lz_d = lw.get("z",0) - plw.get("z",0)
    rz_d = rw.get("z",0) - prw.get("z",0)
    avg_z_d = (lz_d + rz_d) / 2
    forward = avg_z_d < -0.03
    backward = avg_z_d > 0.05

    is_sustained = 0.02 < avg_vel <= 0.08
    enc = 1 if (body_exp < 0.8 or exp_delta < -0.02) and avg_vel > 0.03 else 0
    spread = 1 if (body_exp > 1.3 or exp_delta > 0.02) and avg_vel > 0.03 else 0
    direct = 1 if hands_forward and is_sustained and forward else 0
    has_lateral = lateral_vel > 0.02
    indirect = 1 if (not hands_forward or has_lateral) and avg_vel > 0.04 and body_exp > 1.0 else 0
    adv = 1 if forward and avg_vel > 0.06 and is_sustained else 0
    ret = 1 if backward and avg_vel > 0.07 and is_sustained else 0

    ts = frames[i]["timestamp"]
    print(f"{i:4} | {ts:.3f} | {body_exp:.3f} | {avg_vel:.4f} | {exp_delta:+.3f} | {lateral_vel:.4f} | {avg_z_d:+.4f} | {1 if forward else 0} | {1 if backward else 0} | {enc} | {spread} | {direct} | {indirect} | {adv} | {ret}")
