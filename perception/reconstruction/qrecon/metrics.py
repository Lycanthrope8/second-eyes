"""Measurement definitions (specification, section 6). No scale or pose is fitted here: clouds and depths are
compared as they are. Alignment for trajectories without a metric anchor is done elsewhere and labelled diagnostic."""
import numpy as np
from scipy.spatial import cKDTree


def depth_metrics(pred, ref, mask):
    """Per-keyframe depth against reference depth on the evaluated region (mask)."""
    ok = mask & (pred > 0)
    out = {"evaluated_pixels": int(mask.sum()), "predicted_pixels": int(ok.sum()),
           "coverage": float(ok.sum() / max(mask.sum(), 1))}
    if ok.sum() == 0:
        return out
    p, r = pred[ok].astype(np.float64), ref[ok].astype(np.float64)
    err = np.abs(p - r)
    ratio = p / r
    out.update({"abs_rel": float(np.mean(err / r)), "median_abs_err_m": float(np.median(err)),
                "p95_abs_err_m": float(np.percentile(err, 95)), "within_5cm": float(np.mean(err < 0.05)),
                "delta1": float(np.mean(np.maximum(ratio, 1 / ratio) < 1.25)),
                "scale_ratio_median": float(np.median(ratio))})
    return out


def voxel_downsample(pts, voxel_m):
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def cloud_metrics(pred, ref, taus_m, voxel_m):
    """Predicted fused cloud against the reference cloud, both thinned to one point per voxel."""
    pred, ref = voxel_downsample(pred, voxel_m), voxel_downsample(ref, voxel_m)
    out = {"pred_points": int(len(pred)), "ref_points": int(len(ref))}
    if len(pred) == 0 or len(ref) == 0:
        return out
    d_pr = cKDTree(ref).query(pred)[0]         # accuracy: predicted -> reference
    d_rp = cKDTree(pred).query(ref)[0]         # completeness: reference -> predicted
    out.update({"accuracy_median_m": float(np.median(d_pr)), "accuracy_p90_m": float(np.percentile(d_pr, 90)),
                "accuracy_p95_m": float(np.percentile(d_pr, 95)), "completeness_median_m": float(np.median(d_rp))})
    for tau in taus_m:
        p, r = float(np.mean(d_pr < tau)), float(np.mean(d_rp < tau))
        key = f"{int(round(tau * 100))}cm"
        out[f"precision_{key}"] = p
        out[f"recall_{key}"] = r
        out[f"fscore_{key}"] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return out


def scale_error(ratios):
    """Sequence-level scale error from per-keyframe median depth ratios (predicted / reference)."""
    r = np.asarray([x for x in ratios if x is not None and np.isfinite(x)])
    if len(r) == 0:
        return None
    return float(abs(np.median(r) - 1.0))


def trajectory_errors(est_pos, ref_pos):
    d = np.linalg.norm(est_pos - ref_pos, axis=1)
    return {"ate_rmse_m": float(np.sqrt(np.mean(d ** 2))), "ate_median_m": float(np.median(d)), "ate_max_m": float(d.max())}
