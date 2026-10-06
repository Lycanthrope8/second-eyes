"""Headset detections against the PC reference (A1.10c, D85).

A detection is [class, score, x1, y1, x2, y2]. Two detections match when they have the same class, a box overlap (IoU,
continuous coordinates) of at least 0.95 and scores within 0.02. Matching is one-to-one, greedy by descending score.
For each image the PC reference and the headset must have the same detections at or above the 0.3 threshold; a
detection within 0.02 of the threshold on either side that finds no partner is reported as borderline, not as a
failure, because the score tolerance alone could move it across the threshold. The parity path (exact 416 x 416
canvases) decides the result; the texture path and camera snapshots are reported beside it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import package as P

SESSION_LOG = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9A-Za-z]+[.]jsonl$")
IOU_MIN, SCORE_TOL, MARGIN = 0.95, 0.02, 0.02


def session_logs(target, include_before_run=False) -> list:
    t = Path(target)
    if t.is_file():
        return [t]
    raw = P.REPO / "runs" / str(target) / "raw"
    if not raw.is_dir():
        raise SystemExit(f"no log file and no run folder {raw}")
    files = sorted(p for p in raw.glob("*.jsonl") if SESSION_LOG.match(p.name))
    if include_before_run:
        files += sorted(p for p in (raw / "before_run").glob("*.jsonl") if SESSION_LOG.match(p.name))
    if not files:
        raise SystemExit(f"no session logs in {raw}")
    return files


def results(paths) -> list:
    out = []
    for path in paths:
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                e = json.loads(line)
                if e.get("ev") == "detector.result":
                    out.append((Path(path).name, n, e))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    iy = max(0.0, min(a[5], b[5]) - max(a[3], b[3]))
    inter = ix * iy
    union = (a[4] - a[2]) * (a[5] - a[3]) + (b[4] - b[2]) * (b[5] - b[3]) - inter
    return inter / union if union > 0 else 0.0


def compare(ref: list, got: list, threshold=P.SCORE_THRESHOLD) -> dict:
    """Match one image's detections; failures are unmatched detections clearly above the threshold."""
    ref = sorted(ref, key=lambda d: -d[1])
    got = sorted(got, key=lambda d: -d[1])
    used, pairs = set(), []
    for i, r in enumerate(ref):
        best = None
        for j, g in enumerate(got):
            if j in used or g[0] != r[0] or abs(g[1] - r[1]) > SCORE_TOL:
                continue
            o = iou(r, g)
            if o >= IOU_MIN and (best is None or o > best[1]):
                best = (j, o)
        if best is not None:
            used.add(best[0])
            pairs.append({"class": r[0], "pc_score": r[1], "headset_score": got[best[0]][1], "iou": best[1]})
    matched_ref = {id(ref[i]) for i in range(len(ref)) if any(p["pc_score"] == ref[i][1] and p["class"] == ref[i][0]
                                                              for p in pairs)}
    lone_ref = [r for r in ref if id(r) not in matched_ref and r[1] >= threshold - MARGIN]
    lone_got = [g for j, g in enumerate(got) if j not in used]
    fail = [("pc_only", r) for r in lone_ref if r[1] >= threshold + MARGIN] + \
           [("headset_only", g) for g in lone_got if g[1] >= threshold + MARGIN]
    border = [("pc_only", r) for r in lone_ref if r[1] < threshold + MARGIN] + \
             [("headset_only", g) for g in lone_got if g[1] < threshold + MARGIN]
    return {"pass": not fail, "pairs": pairs, "failures": fail, "borderline": border,
            "max_score_diff": max((abs(p["pc_score"] - p["headset_score"]) for p in pairs), default=0.0),
            "min_iou": min((p["iou"] for p in pairs), default=1.0)}


def evaluate(events, reference: dict) -> dict:
    ref = {e["id"]: e for e in reference["images"]}
    rows, seen = [], set()
    for log, line, ev in events:
        d = ev["data"]
        if d.get("path") not in ("tensor", "texture_srgb", "texture_linear") or d.get("image_id") not in ref:
            continue
        r = ref[d["image_id"]]
        if d["path"] == "tensor":
            want = r["canvas_detections"]
        elif "full_detections" in r:
            want = r["full_detections"]
        else:
            continue
        c = compare(want, d["detections"])
        rows.append({"log": log, "line": line, "path": d["path"], "backend": d.get("backend"), "image": d["image_id"],
                     **c})
        if d["path"] == "tensor" and d.get("backend") == "gpu_compute":
            seen.add(d["image_id"])
    gpu = [x for x in rows if x["path"] == "tensor" and x["backend"] == "gpu_compute"]
    missing = sorted(set(ref) - seen)
    return {"rows": rows, "missing_gpu_images": missing,
            "parity_pass": bool(gpu) and not missing and all(x["pass"] for x in gpu)}


def snapshot(events, image_path, snapshot_id) -> dict:
    """A saved camera canvas against the headset's detections for it, both in canvas pixels (strict criteria)."""
    from . import reference as R
    hits = [e for _, _, e in events if e["data"].get("snapshot") == snapshot_id]
    if len(hits) != 1:
        raise SystemExit(f"expected one detector.result with snapshot {snapshot_id!r}, found {len(hits)}")
    d = hits[0]["data"]
    got = [[c, s, x1 * d["ratio"], y1 * d["ratio"], x2 * d["ratio"], y2 * d["ratio"]]
           for c, s, x1, y1, x2, y2 in d["detections"]]
    man = P.load_manifest()
    sess = R.session(P.PACKAGE_DIR / man["model_file"])
    want = R.detect_canvas(sess, R.read_png(image_path), 1.0, 0.25)
    return {"backend": d.get("backend"), **compare(want, got)}
