"""Builds the JSON configuration of the headset program (questpipe/qrecon_quest.cc) from the same YAML files the
desktop harness uses, so both run with identical parameters."""
import os
from . import config as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def camera_block(calib, scale=1.0):
    K = [list(map(float, r)) for r in calib["K"]]
    if scale != 1.0:
        K = [[K[0][0] * scale, 0.0, (K[0][2] + 0.5) * scale - 0.5], [0.0, K[1][1] * scale, (K[1][2] + 0.5) * scale - 0.5], [0, 0, 1.0]]
    w, h = calib["size"]
    return {"K": K, "dist": list(map(float, calib.get("dist", [0, 0, 0, 0, 0]))), "size": [int(round(w * scale)), int(round(h * scale))]}


def build(frames, trajectory, camera, depth, anchor=None, max_keyframes=None, max_pose_dt_s=0.05):
    """frames, trajectory: paths as the program will see them. camera: camera_block of the tracking frames.
    depth: {"source": "onnx", "model": path, "threads": n} or {"source": "oracle", "oracle": {...}}.
    anchor: None, or {"board_frames": path, "camera": camera_block at full resolution, "recording": name,
    "mode": "strict" (refuse when a check fails) or "best_effort" (anchor by whichever board can be fitted; failed checks
    become warnings)}."""
    q = C.load(os.path.join(REPO, "configs", "qrecon.yaml"))
    cfg = {"frames": frames, "trajectory": trajectory, "camera": camera, "depth": depth, "max_pose_dt_s": max_pose_dt_s,
           "keyframes": q["keyframes"], "landmarks": q["landmarks"], "calibration": q["calibration"],
           "consistency": q["consistency"], "fusion": q["fusion"], "max_keyframes": max_keyframes,
           "anchor": {"enabled": False}}
    if anchor:
        ref = C.load(os.path.join(REPO, "configs", "references.yaml"))
        cfg["anchor"] = {"enabled": True, "board_frames": anchor["board_frames"], "camera": anchor["camera"],
                         "boards": ref["boards"], "windows": ref["recordings"][anchor["recording"]]["windows_s"],
                         "params": ref["anchor"], "mode": anchor.get("mode", "strict")}
    return cfg


def stella_camera_yaml(calib, scale=1.0, fps=30.0, name="replayed video"):
    """stella_vslam camera file for frames resized by `scale` (OpenCV pixel-centre convention)."""
    K = calib["K"]
    d = list(calib.get("dist", [])) + [0.0] * 5
    w, h = calib["size"]
    s = scale
    return f"""# generated from a calibration, scale {s}
Camera:
  name: "{name}"
  setup: "monocular"
  model: "perspective"
  fx: {K[0][0]*s:.6f}
  fy: {K[1][1]*s:.6f}
  cx: {(K[0][2]+0.5)*s-0.5:.6f}
  cy: {(K[1][2]+0.5)*s-0.5:.6f}
  k1: {d[0]:.8f}
  k2: {d[1]:.8f}
  p1: {d[2]:.8f}
  p2: {d[3]:.8f}
  k3: {d[4]:.8f}
  fps: {fps}
  cols: {int(round(w*s))}
  rows: {int(round(h*s))}
  color_order: "BGR"

Preprocessing:
  min_size: 800

Feature:
  name: "default ORB feature extraction setting"
  scale_factor: 1.2
  num_levels: 8
  ini_fast_threshold: 20
  min_fast_threshold: 7

Mapping:
  baseline_dist_thr_ratio: 0.02
  redundant_obs_ratio_thr: 0.9
  num_covisibilities_for_landmark_generation: 20
  num_covisibilities_for_landmark_fusion: 20
"""
