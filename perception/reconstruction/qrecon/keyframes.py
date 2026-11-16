"""Keyframe selection. One fixed rule, so every depth method gets the same keyframes."""
import numpy as np
from .geometry import rotation_angle_deg


def select(poses, times, min_trans_m, min_rot_deg, min_dt_s):
    keep = [0]
    for i in range(1, len(poses)):
        a, b = poses[keep[-1]], poses[i]
        if times[i] - times[keep[-1]] < min_dt_s:
            continue
        moved = np.linalg.norm(b[:3, 3] - a[:3, 3]) >= min_trans_m
        turned = rotation_angle_deg(a[:3, :3], b[:3, :3]) >= min_rot_deg
        if moved or turned:
            keep.append(i)
    return keep
