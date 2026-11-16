"""The reconstruction stages in order, shared by every run type."""
import os
import numpy as np
import cv2
from .geometry import backproject
from . import keyframes as KF, landmarks as LM, calibrate as CAL, consistency as CONS, metrics as M
from .fusion import VoxelMap, write_ply


def region_mask(depth_ref, cfg):
    r = cfg["evaluated_region"]
    return (depth_ref >= r["min_depth_m"]) & (depth_ref <= r["max_depth_m"])


def choose_keyframes(src, cfg, poses=None):
    poses = poses if poses is not None else src.poses()
    k = cfg["keyframes"]
    return KF.select(poses, src.times(), k["min_trans_m"], k["min_rot_deg"], k["min_dt_s"])


def _fuse(depths, frames, poses, K, cfg, masks=None):
    f = cfg["fusion"]
    vm = VoxelMap(f["voxel_m"], f["max_voxels"], f["min_observations"])
    for i, d in depths.items():
        dd = np.where(d <= f["max_depth_m"], d, 0)
        pts, uv = backproject(dd, K, poses[i], stride=f["pixel_stride"], mask=None if masks is None else masks[i])
        if len(pts) == 0:
            continue
        col = frames[i].bgr[uv[:, 1].astype(int), uv[:, 0].astype(int)][:, ::-1].astype(np.float64)
        vm.integrate(pts, col)
    return vm


def run_reference(src, kf, cfg, out_dir):
    """Run A: reference poses and measured depth. Also the reference surface for every other run."""
    frames = [src.frame(i) for i in kf]
    poses = [f.T_wc for f in frames]
    depths = {i: f.depth for i, f in enumerate(frames)}
    masks = {i: region_mask(f.depth, cfg) for i, f in enumerate(frames)}
    vm = _fuse(depths, frames, poses, frames[0].K, cfg, masks)
    pts, cols = vm.cloud()
    write_ply(os.path.join(out_dir, "reference.ply"), pts, cols)
    return {"keyframes": [int(i) for i in kf], "fusion": vm.stats(), "points": int(len(pts))}


def run_dense(src, kf, poses_used, provider, cfg, out_dir, log=print):
    """Depth prediction -> metric calibration -> cross-view filter -> fusion, on the given keyframes and poses.
    poses_used: one pose per keyframe, metric. Reference depth is used only to score, never to fit."""
    frames = [src.frame(i) for i in kf]
    K = frames[0].K
    gray = [cv2.cvtColor(f.bgr, cv2.COLOR_BGR2GRAY) for f in frames]
    log(f"predicting depth for {len(frames)} keyframes with {provider.name}")
    raw = provider.predict(frames, poses_used, K)
    per_kf, metric = [], {}
    for i in range(len(frames)):
        px, z, _ = LM.landmarks_for(i, gray, poses_used, K, cfg["landmarks"])
        rec = {"keyframe": int(kf[i]), "landmarks": int(len(z))}
        if provider.kind == "inverse_relative":
            c = CAL.fit_inverse_depth(raw[i], px, z, cfg["calibration"], seed=int(kf[i]))
            if c.ok:
                metric[i] = CAL.apply_inverse_depth(raw[i], c, cfg["fusion"]["max_depth_m"])
        else:
            c = CAL.check_scale(raw[i], px, z, cfg["calibration"])
            if c.ok:
                d = raw[i].astype(np.float32)
                if cfg["arms"]["da3s"].get("apply_landmark_scale", False):
                    d = d * np.float32(c.a)
                metric[i] = np.where(np.isfinite(d) & (d > 0), d, 0).astype(np.float32)
        rec.update(calibrated=bool(c.ok), reason=c.reason, calibration=c.stats)
        per_kf.append(rec)
    filtered = {}
    for i in metric:
        filtered[i], st = CONS.filter_depth(i, metric, poses_used, K, cfg["consistency"])
        per_kf[i]["consistency"] = st
    masks, ratios = {}, []
    for i in filtered:
        f = frames[i]
        if f.depth is not None:
            masks[i] = region_mask(f.depth, cfg)
            dm = M.depth_metrics(filtered[i], f.depth, masks[i])
            dm["outside_region_share"] = float(((filtered[i] > 0) & ~masks[i]).sum() / max((filtered[i] > 0).sum(), 1))
            per_kf[i]["depth"] = dm
            ratios.append(dm.get("scale_ratio_median"))
    vm = _fuse(filtered, frames, poses_used, K, cfg, masks if masks else None)
    pts, cols = vm.cloud()
    write_ply(os.path.join(out_dir, "predicted.ply"), pts, cols)
    valid = sum(1 for r in per_kf if r["calibrated"])
    summary = {"arm": provider.name, "keyframes": len(frames), "valid_keyframes": valid,
               "valid_keyframe_share": valid / max(len(frames), 1), "scale_error": M.scale_error(ratios),
               "fusion": vm.stats(), "points": int(len(pts))}
    return summary, per_kf


def judge(summary, cloud, cfg):
    """Screening limits from the specification. Reported with the measurements, never instead of them."""
    g = cfg["gates"]
    checks = {
        "scale_error_within_limit": summary.get("scale_error") is not None and summary["scale_error"] <= g["max_scale_error"],
        "surface_median_within_limit": cloud.get("accuracy_median_m") is not None and cloud["accuracy_median_m"] <= g["max_surface_median_m"],
        "enough_valid_keyframes": summary["valid_keyframe_share"] >= g["min_valid_keyframe_share"],
    }
    return {"checks": checks, "passes_screening": all(checks.values())}
