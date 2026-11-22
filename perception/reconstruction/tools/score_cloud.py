#!/usr/bin/env python3
"""Scores a merged cloud against a comparison reference (D141): how close our points are to the reference's surfaces,
and how much of the reference our points cover.

    python tools/score_cloud.py --reference PATH\\registered.ply --cloud house_all6 [--align] [--by-run]

--cloud is a merge name (runs/NAME.ply) or a path. The reference must already be in the same board frame. Both clouds are
thinned to one point per 2 cm cell (the reference also to 5 cm cells for coverage). Reports the median distance from our
points to the reference, precision (our points within 5 and 10 cm of it), recall (reference cells within 5 and 10 cm of
ours) and the F-score; --by-run adds the same per map (needs runs/NAME_by_run.ply from merge_clouds.py --color-by-run);
--align first refines the rigid alignment and reports the correction. The reference is a comparison, never ground truth
(D131). Writes runs/NAME_score.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1", "char": "i1",
         "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2", "uint": "<u4", "uint32": "<u4",
         "int": "<i4", "int32": "<i4"}


def read_ply(path):
    """Points (float32) and colours (uint8, or None) of a binary little-endian PLY, read without copying the file."""
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no PLY header")
            head += line
    props, n = [], 0
    for line in head.decode("ascii", "replace").splitlines():
        w = line.split()
        if w[:2] == ["element", "vertex"]:
            n = int(w[2])
        elif w[:1] == ["property"] and len(w) == 3 and n:
            props.append((w[2], TYPES[w[1]]))
        elif w[:1] == ["element"] and w[1] != "vertex" and props:
            break
    v = np.memmap(path, dtype=np.dtype(props), mode="r", offset=len(head), shape=(n,))
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    C = np.stack([v["red"], v["green"], v["blue"]], 1) if "red" in v.dtype.names else None
    return P, C


def thin(P, cell):
    """One point per occupied cell."""
    k = np.floor(P / cell).astype(np.int64)
    k -= k.min(0)
    key = (k[:, 0] * (k[:, 1].max() + 1) + k[:, 1]) * (k[:, 2].max() + 1) + k[:, 2]
    _, i = np.unique(key, return_index=True)
    return P[i]


def icp(src, tree, ref, iters=40):
    R, t = np.eye(3), np.zeros(3)
    for it in range(iters):
        Q = src @ R.T + t
        d, ii = tree.query(Q, k=1)
        keep = d < max(0.25 * 0.9 ** it, 0.08)
        A, B = Q[keep], ref[ii[keep]]
        ma, mb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
        dR = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        R, t = dR @ R, dR @ (t - ma) + mb
    return R, t


def score(ours, tree2, ref5):
    from scipy.spatial import cKDTree
    d, _ = tree2.query(ours, k=1)
    dc, _ = cKDTree(ours).query(ref5, k=1)
    out = {"points_2cm": int(len(ours)), "median_m": round(float(np.median(d)), 4)}
    for th in (0.05, 0.10):
        p, r = float((d < th).mean()), float((dc < th).mean())
        out[f"precision_{int(th * 100)}cm"], out[f"recall_{int(th * 100)}cm"] = round(p, 3), round(r, 3)
        out[f"f_{int(th * 100)}cm"] = round(2 * p * r / (p + r), 3) if p + r else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="the comparison reference (PLY), in the same board frame")
    ap.add_argument("--cloud", required=True, help="a merge name in runs/ or a PLY path")
    ap.add_argument("--align", action="store_true", help="refine the rigid alignment first and report it")
    ap.add_argument("--by-run", action="store_true", help="also score each map (needs NAME_by_run.ply and NAME.json)")
    a = ap.parse_args()
    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError:
        sys.exit("scipy is not installed in this Python: activate the project's environment")
    cloud = Path(a.cloud) if a.cloud.lower().endswith(".ply") else HERE / "runs" / f"{a.cloud}.ply"
    name = cloud.stem
    R_ref, _ = read_ply(a.reference)
    ref2, ref5 = thin(R_ref, 0.02), thin(R_ref, 0.05)
    del R_ref
    tree2 = cKDTree(ref2)
    P, _ = read_ply(cloud)
    R, t = np.eye(3), np.zeros(3)
    report = {"reference": str(a.reference), "cloud": str(cloud), "note": "comparison, not ground truth (D131, D141)"}
    if a.align:
        R, t = icp(thin(P, 0.05).astype(np.float64), tree2, ref2)
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        report["alignment_correction"] = {"rotation_deg": round(ang, 2), "translation_m": round(float(np.linalg.norm(t)), 3)}
        print(f"alignment refined by {ang:.2f} deg and {100 * np.linalg.norm(t):.1f} cm")
    ours = thin(P, 0.02).astype(np.float64) @ R.T + t
    report["whole"] = w = score(ours, tree2, ref5)
    print(f"{name}: median {100 * w['median_m']:.1f} cm; F-score {w['f_5cm']:.2f} at 5 cm, {w['f_10cm']:.2f} at 10 cm "
          f"(precision {w['precision_10cm']:.2f}, recall {w['recall_10cm']:.2f} at 10 cm)")
    if a.by_run:
        meta = json.loads((cloud.parent / f"{name}.json").read_text())
        _, C = read_ply(cloud.parent / f"{name}_by_run.ply")
        import view_maps as VM
        info, _ = VM.map_info(cloud.parent)
        report["runs"] = {}
        print(f"  {'map':22s} {'median':>7s} {'within 5 cm':>11s} {'within 10 cm':>12s}  run")
        for run, col in meta["run_colors"].items():
            Q = thin(P[np.all(C == np.array(col, np.uint8), axis=1)], 0.02).astype(np.float64) @ R.T + t
            d, _ = tree2.query(Q, k=1)
            r = {"points_2cm": int(len(Q)), "median_m": round(float(np.median(d)), 4),
                 "within_5cm": round(float((d < 0.05).mean()), 3), "within_10cm": round(float((d < 0.10).mean()), 3)}
            report["runs"][run] = r
            title = info.get(run, {}).get("title", run)
            print(f"  {title:22s} {100 * r['median_m']:6.1f}  {100 * r['within_5cm']:10.0f}% {100 * r['within_10cm']:11.0f}%  {run}")
    out = HERE / "runs" / f"{name}_score.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
