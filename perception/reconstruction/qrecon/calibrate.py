"""Turn a network's relative output into metric depth using sparse metric landmarks.

Relative (disparity-like) models: fit scale and shift in inverse depth, 1/z = a*d + b.
Depth-valued models: a single robust scale, reported and optionally applied.
Every fit is checked on landmarks it did not see; a keyframe without enough support is rejected, not guessed."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Calib:
    ok: bool
    reason: str = ""
    a: float = 0.0
    b: float = 0.0
    stats: dict = field(default_factory=dict)


def _sample(img, px):
    h, w = img.shape
    u = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
    return img[v, u]


def _support(px, z, shape, cfg):
    h, w = shape
    g = int(cfg["grid"])
    cells = set(zip((px[:, 0] * g // w).astype(int).clip(0, g - 1), (px[:, 1] * g // h).astype(int).clip(0, g - 1)))
    spread = float(np.percentile(z, 90) / max(np.percentile(z, 10), 1e-6))
    return len(cells), spread


def _lsq(x, y):
    A = np.stack([x, np.ones_like(x)], axis=1)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fit_inverse_depth(rel, px, z, cfg, seed=0):
    """rel: network output image; px, z: landmark pixels and metric camera-space depths."""
    n = len(z)
    if n < cfg["min_landmarks"]:
        return Calib(False, "too few landmarks", stats={"landmarks": n})
    cells, spread = _support(px, z, rel.shape, cfg)
    if cells < cfg["min_cells"] or spread < cfg["min_depth_spread"]:
        return Calib(False, "landmarks not spread over image and depth", stats={"landmarks": n, "cells": cells, "spread": spread})
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_hold = max(1, int(round(n * cfg["holdout_fraction"])))
    hold, fit = order[:n_hold], order[n_hold:]
    x, y = _sample(rel, px).astype(np.float64), 1.0 / z
    best = None
    for _ in range(int(cfg["ransac_iters"])):
        i, j = rng.choice(fit, 2, replace=False)
        if abs(x[i] - x[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        b = y[i] - a * x[i]
        inv_pred = a * x[fit] + b
        good = inv_pred > 1e-6
        rel_err = np.full(len(fit), np.inf)
        rel_err[good] = np.abs(1.0 / inv_pred[good] - z[fit][good]) / z[fit][good]
        inl = rel_err < cfg["inlier_rel_err"]
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < cfg["min_inliers"]:
        return Calib(False, "no consistent fit", stats={"landmarks": n, "inliers": 0 if best is None else int(best.sum())})
    a, b = _lsq(x[fit][best], y[fit][best])
    inv_hold = a * x[hold] + b
    if np.any(inv_hold <= 1e-6):
        hold_err = np.inf
    else:
        hold_err = float(np.median(np.abs(1.0 / inv_hold - z[hold]) / z[hold]))
    positive = float(np.mean(a * rel + b > 1e-6))
    stats = {"landmarks": n, "inliers": int(best.sum()), "cells": cells, "spread": spread,
             "holdout_median_rel_err": hold_err, "positive_share": positive}
    if positive < cfg["min_positive_share"]:
        return Calib(False, "fit gives non-positive depth over part of the image", a, b, stats)
    if hold_err > cfg["max_holdout_rel_err"]:
        return Calib(False, "fit fails on held-out landmarks", a, b, stats)
    return Calib(True, "", float(a), float(b), stats)


def apply_inverse_depth(rel, c, max_depth_m):
    invd = c.a * rel.astype(np.float64) + c.b
    depth = np.zeros(rel.shape, dtype=np.float32)
    ok = invd > 1.0 / max_depth_m
    depth[ok] = (1.0 / invd[ok]).astype(np.float32)
    return depth


def check_scale(depth, px, z, cfg):
    """For models that already return depth: robust scale against landmarks, with the same support rules."""
    n = len(z)
    if n < cfg["min_landmarks"]:
        return Calib(False, "too few landmarks", stats={"landmarks": n})
    d = _sample(depth, px).astype(np.float64)
    ok = d > 0
    if ok.sum() < cfg["min_inliers"]:
        return Calib(False, "no depth at landmarks", stats={"landmarks": n})
    ratio = z[ok] / d[ok]
    s = float(np.median(ratio))
    rel_err = float(np.median(np.abs(s * d[ok] - z[ok]) / z[ok]))
    cells, spread = _support(px[ok], z[ok], depth.shape, cfg)
    stats = {"landmarks": n, "scale": s, "median_rel_err_after_scale": rel_err, "cells": cells, "spread": spread}
    if rel_err > cfg["max_holdout_rel_err"]:
        return Calib(False, "depth disagrees with landmarks even after one scale", s, 0.0, stats)
    return Calib(True, "", s, 0.0, stats)
