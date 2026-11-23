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
STATS = ["landmarks", "cells", "spread", "inliers", "holdout_median_rel_err", "positive_share"]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a reconstruction run id, or a keyframes.json path")
    ap.add_argument("--summary", help="the recording summary, to name the pieces of its one map")
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
    out = path.parent / "keyframe_report.json"
    try:
        out.write_text(json.dumps(r, indent=1))
        print(f"-> {out}")
    except OSError:
        print(f"(could not write {out})")


if __name__ == "__main__":
    main()
