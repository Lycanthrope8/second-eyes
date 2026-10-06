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


def compare(ref: list, got: list, threshold=P.SCORE_THRESHOLD, got_floor=None) -> dict:
    """Match one image's detections and sort what is left into the D85 categories (D88's reporting).

    Both lists may hold detections below the threshold, down to each side's floor (the PC reference keeps 0.25; the
    headset logs its own floor from D88 on, earlier builds only 0.3). Matching is one-to-one, greedy by descending
    PC score, same class, IoU >= 0.95 and scores within 0.02. Then:
    - strict: matched pairs with both scores at or above the threshold;
    - crossing: matched pairs on opposite sides of the threshold (a verified threshold-crossing exception);
    - borderline: an unmatched detection within 0.02 of the threshold, at or above it on its own side, whose partner
      could not be checked or was not found (unresolved);
    - failure: an unmatched detection at or above threshold + 0.02, or an unmatched one the other side's floor could
      have shown.
    """
    if got_floor is None:
        got_floor = min([d[1] for d in got] + [threshold])
    ref = sorted(ref, key=lambda d: -d[1])
    got = sorted(got, key=lambda d: -d[1])
    used, pairs, matched_ref = set(), [], set()
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
            matched_ref.add(i)
            pairs.append({"class": r[0], "pc_score": r[1], "headset_score": got[best[0]][1], "iou": best[1]})
    crossing = [x for x in pairs if (x["pc_score"] >= threshold) != (x["headset_score"] >= threshold)]
    lone_ref = [r for i, r in enumerate(ref) if i not in matched_ref and r[1] >= threshold]
    lone_got = [g for j, g in enumerate(got) if j not in used and g[1] >= threshold]
    fail, border = [], []
    for side, lone, other_floor in (("pc_only", lone_ref, got_floor), ("headset_only", lone_got, 0.25)):
        for d in lone:
            if d[1] >= threshold + MARGIN:
                fail.append((side, d))
            elif d[1] - SCORE_TOL >= other_floor + 1e-9 and other_floor < threshold:
                fail.append((side, d))       # the other side's floor could have shown a partner, and did not
            else:
                border.append((side, d))
    in_pairs = [x for x in pairs if x["pc_score"] >= threshold or x["headset_score"] >= threshold]
    return {"pass": not fail, "strict": not fail and not crossing and not border, "pairs": in_pairs,
            "crossings": crossing, "failures": fail, "borderline": border,
            "max_score_diff": max((abs(p["pc_score"] - p["headset_score"]) for p in in_pairs), default=0.0),
            "min_iou": min((p["iou"] for p in in_pairs), default=1.0)}


def evaluate(events, reference: dict, extra_dir=None) -> dict:
    """Every self-check result against the reference; extra canvases (a snapshot pushed to the headset) are judged
    against a PC reference made now from the PNG of the same name in extra_dir."""
    ref = {e["id"]: e for e in reference["images"]}
    extra = {}
    rows, seen = [], set()
    for log, line, ev in events:
        d = ev["data"]
        path, image = d.get("path"), d.get("image_id")
        if path not in ("tensor", "texture_srgb", "texture_linear") or image is None:
            continue
        if image in ref:
            r = ref[image]
            want = r["canvas_detections"] if path == "tensor" else r.get("full_detections")
            if want is None:
                continue
            kind = "reference"
        elif path == "tensor" and extra_dir is not None and (Path(extra_dir) / f"{image}.png").is_file():
            if image not in extra:
                from . import reference as R
                man = P.load_manifest()
                extra[image] = R.detect_canvas(R.session(P.PACKAGE_DIR / man["model_file"]),
                                               R.read_png(Path(extra_dir) / f"{image}.png"), 1.0, 0.25)
            want, kind = extra[image], "extra"
        else:
            continue
        c = compare(want, d["detections"], got_floor=d.get("floor"))
        rows.append({"log": log, "line": line, "path": path, "backend": d.get("backend"), "image": image, "kind": kind,
                     "steps_per_frame": d.get("steps_per_frame"), **c})
        if path == "tensor" and d.get("backend") == "gpu_compute" and kind == "reference":
            seen.add(image)
    gpu = [x for x in rows if x["path"] == "tensor" and x["backend"] == "gpu_compute"]
    missing = sorted(set(ref) - seen)
    return {"rows": rows, "missing_gpu_images": missing,
            "parity_pass": bool(gpu) and not missing and all(x["pass"] for x in gpu),
            "strict": bool(gpu) and not missing and all(x["strict"] for x in gpu)}


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
