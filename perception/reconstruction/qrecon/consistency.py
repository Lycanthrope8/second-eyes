"""Cross-view check: a depth value is kept only if at least one neighbouring keyframe agrees with it.
Depth edges are dropped first, because single-image depth smears across them."""
import numpy as np
from .geometry import backproject, project


def edge_mask(depth, max_rel_grad):
    d = depth.astype(np.float64)
    gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(d > 0, np.maximum(gx, gy) / d, np.inf)
    return rel < max_rel_grad


def filter_depth(i, depths, poses, K, cfg):
    """Returns (filtered depth, stats). depths: dict keyframe -> metric depth (missing = rejected keyframe)."""
    d = depths[i]
    keep = (d > 0) & edge_mask(d, cfg["max_rel_depth_gradient"])
    pts, uv = backproject(d, K, poses[i], stride=1, mask=keep)
    votes = np.zeros(len(pts), dtype=int)
    seen = np.zeros(len(pts), dtype=int)
    for off in cfg["neighbour_offsets"]:
        j = i + off
        if j not in depths:
            continue
        u, v, z = project(pts, K, poses[j])
        h, w = depths[j].shape
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        dj = np.zeros(len(pts))
        dj[inside] = depths[j][vi[inside], ui[inside]]
        valid = inside & (dj > 0)
        seen += valid
        votes += valid & (np.abs(dj - z) <= cfg["max_rel_diff"] * z)
    ok = votes >= cfg["min_agreeing_views"]
    if cfg["keep_unverified"]:
        ok |= seen == 0
    out = np.zeros_like(d)
    uu, vv = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
    out[vv, uu] = d[vv, uu]
    n0 = int((d > 0).sum())
    stats = {"input_pixels": n0, "after_edges": int(keep.sum()), "kept": int(ok.sum()),
             "kept_share": float(ok.sum() / max(n0, 1)), "checked_share": float((seen > 0).mean()) if len(seen) else 0.0,
             "agree_share_of_checked": float((votes[seen > 0] > 0).mean()) if (seen > 0).any() else 0.0}
    return out, stats
