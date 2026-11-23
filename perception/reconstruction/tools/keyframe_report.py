#!/usr/bin/env python3
"""Why keyframes of a reconstruction were not calibrated: counts by reason, by piece of the one map and by stretch of
video, and the calibration statistics of calibrated and failed keyframes.

    python tools/keyframe_report.py RUN [--summary runs/<name>_recording.json]

RUN is a reconstruction run (runs/RUN/recon/keyframes.json) or a keyframes.json path. A keyframe's depth map is made
metric by fitting it to points triangulated against its neighbouring keyframes; it is refused when too few points were
triangulated, when they cover too little of the image or of the depth range, or when the depth map does not agree with
them (held-out points, positive depth, a consistent fit). Writes RUN/recon/keyframe_report.json beside keyframes.json.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
STATS = ["landmarks", "cells", "spread", "inliers", "holdout_median_rel_err", "positive_share", "regions_left_out"]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q / 100 * (len(v) - 1)))))]


def report(kfs, pieces=()):
    def piece_of(t):
        for a, b, name in pieces:
            if a - 0.05 <= t <= b + 0.05:
                return name
        return "outside any piece"
    out = {"keyframes": len(kfs), "calibrated": sum(1 for k in kfs if k.get("calibrated")),
           "reasons": dict(Counter(k.get("reason") or "calibrated" for k in kfs).most_common()), "pieces": {}, "by_20s": {}, "stats": {}}
    byp = defaultdict(list)
    for k in kfs:
        byp[piece_of(k["t"])].append(k)
    for name, ks in sorted(byp.items(), key=lambda kv: min(k["t"] for k in kv[1])):
        out["pieces"][name] = {"calibrated": sum(1 for k in ks if k.get("calibrated")), "keyframes": len(ks),
                               "reasons": dict(Counter(k.get("reason") for k in ks if not k.get("calibrated")).most_common())}
    for k in kfs:
        b = f"{int(k['t'] // 20) * 20}-{int(k['t'] // 20) * 20 + 20} s"
        d = out["by_20s"].setdefault(b, [0, 0])
        d[0] += int(bool(k.get("calibrated"))); d[1] += 1
    for label, group in (("calibrated", [k for k in kfs if k.get("calibrated")]), ("failed", [k for k in kfs if not k.get("calibrated")])):
        out["stats"][label] = {}
        for key in STATS:
            v = [float(k.get("calibration", {})[key]) for k in group if k.get("calibration", {}).get(key) is not None]
            if v:
                out["stats"][label][key] = [round(pct(v, 50), 4), round(pct(v, 10), 4), round(pct(v, 90), 4)]
    few = [k for k in kfs if k.get("reason") == "too few landmarks"]
    out["too_few_with_none"] = sum(1 for k in few if not k.get("landmarks"))
    return out


SHORT = {"too few landmarks": "too few points", "landmarks not spread over image and depth": "not spread",
         "fit fails on held-out landmarks": "held-out off", "fit gives non-positive depth over part of the image": "negative depth",
         "no consistent fit": "no fit", "most of the view disagrees with its landmarks": "mostly disagrees"}


def contact_sheet(kfs, frames_dir, out, between=None, max_failed=25, max_ok=5, thumb_w=320, cols=5):
    """A picture grid of keyframes, the refused ones first (red, with the reason and the held-out error) and a few
    calibrated ones from the same stretch for contrast (green), to see what the refused views show."""
    import cv2
    import numpy as np
    rows = [ln.split() for ln in Path(frames_dir, "rgb.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    paths = [Path(frames_dir) / r[1] for r in rows]
    sel = [k for k in kfs if between is None or between[0] <= k["t"] <= between[1]]
    def spread(ks, n):
        return [ks[int(round(i))] for i in np.linspace(0, len(ks) - 1, min(n, len(ks)))] if ks else []
    chosen = spread([k for k in sel if not k.get("calibrated")], max_failed) + spread([k for k in sel if k.get("calibrated")], max_ok)
    tiles = []
    for k in chosen:
        if k["frame"] >= len(paths):
            continue
        img = cv2.imread(str(paths[k["frame"]]))
        if img is None:
            continue
        img = cv2.resize(img, (thumb_w, int(img.shape[0] * thumb_w / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok = bool(k.get("calibrated"))
        ho = (k.get("calibration") or {}).get("holdout_median_rel_err")
        text = f"{k['t']:.1f}s " + ("calibrated" if ok else SHORT.get(k.get("reason"), k.get("reason") or "refused"))
        if ho is not None and ho < 9:
            text += f" {100 * ho:.0f}%"
        mask = (k.get("calibration") or {}).get("region_mask") if ok else None
        if mask:                                    # the parts of the view left out of its depth, darkened
            g = int(round(len(mask) ** 0.5))
            for c, keep in enumerate(mask):
                if keep == "0":
                    y0, x0 = (c // g) * img.shape[0] // g, (c % g) * img.shape[1] // g
                    y1, x1 = (c // g + 1) * img.shape[0] // g, (c % g + 1) * img.shape[1] // g
                    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.3).astype(img.dtype)
            text_extra = f" {mask.count('0')} parts out"
        else:
            text_extra = ""
        text += text_extra
        colour = (60, 170, 60) if ok else (40, 40, 220)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), colour, 4)
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a reconstruction run id, or a keyframes.json path")
    ap.add_argument("--summary", help="the recording summary, to name the pieces of its one map")
    ap.add_argument("--sheet", action="store_true", help="also write keyframe_sheet.jpg: a picture grid of refused keyframes")
    ap.add_argument("--between", nargs=2, type=float, metavar=("T0", "T1"), help="with --sheet: only keyframes in this stretch (s)")
    ap.add_argument("--frames", help="with --sheet: the frames folder (default: the summary's)")
    a = ap.parse_args()
    path = Path(a.run) if a.run.endswith(".json") else HERE / "runs" / a.run / "recon" / "keyframes.json"
    kfs = json.loads(path.read_text())
    pieces = []
    if a.summary:
        om = json.loads(Path(a.summary).read_text()).get("one_map") or {}
        pieces = [(p["first_t"], p["last_t"], f"map {p['map']}" + (f" of {p['source']}" if p.get("source") else "") + f" ({p['how']})")
                  for p in om.get("pieces", [])]
    r = report(kfs, pieces)
    print(f"{r['calibrated']} of {r['keyframes']} keyframes calibrated ({100 * r['calibrated'] / max(1, r['keyframes']):.0f}%)")
    for reason, n in r["reasons"].items():
        print(f"  {n:4d}  {reason}" + (f" ({r['too_few_with_none']} with none at all)" if reason == "too few landmarks" else ""))
    print("by piece:")
    for name, d in r["pieces"].items():
        print(f"  {name}: {d['calibrated']}/{d['keyframes']}" + ("; " + ", ".join(f"{k}: {v}" for k, v in d["reasons"].items()) if d["reasons"] else ""))
    print("by 20 s of video: " + "  ".join(f"{b} {c}/{n}" for b, (c, n) in r["by_20s"].items()))
    for label, st in r["stats"].items():
        print(f"{label}: " + "; ".join(f"{k} {v[0]:.3g} ({v[1]:.3g}-{v[2]:.3g})" for k, v in st.items()))
    if a.sheet:
        frames = a.frames or (json.loads(Path(a.summary).read_text()).get("frames") if a.summary else None)
        if not frames:
            raise SystemExit("--sheet needs --frames or a --summary that names the frames folder")
        name = "keyframe_sheet" + (f"_{a.between[0]:.0f}-{a.between[1]:.0f}s" if a.between else "") + ".jpg"
        sheet = contact_sheet(kfs, frames, path.parent / name, a.between)
        print(f"-> {sheet}" if sheet else "no keyframes to show in that stretch")
    out = path.parent / "keyframe_report.json"
    try:
        out.write_text(json.dumps(r, indent=1))
        print(f"-> {out}")
    except OSError:
        print(f"(could not write {out})")


if __name__ == "__main__":
    main()
