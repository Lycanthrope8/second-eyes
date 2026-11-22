#!/usr/bin/env python3
"""Compares tracking presets on one recording: how many maps (breaks in tracking) and how much of the video each gives.

    python tools/tracking_sweep.py --video 20261111_121118.mp4 --calib calib_s26_lines.json --name h121118
                                   [--profiles baseline fps30 fps30_features fps30_features_keep]

The presets are in qrecon/tracking_profiles.py. Frames come from data/<name>_15fps or data/<name>_30fps (extracted, with
the video's own timestamps, if missing). Each preset runs tracking on the headset at capture pace, so each takes about
as long as the video. No reconstruction is done. Prints, per preset: maps saved, maps the tracker discarded, breaks,
tracked share (whole recording and from initialisation), the longest map, and how many maps last 10 s or more; and
writes runs/<name>_tracking_sweep.json. A preset's tracking run can be reused afterwards with
process_recording.py --profile <preset> --tracking-run <its run>.
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
try:
    import cv2
except ModuleNotFoundError as e:
    sys.exit(f"{e.name} is not installed in this Python. Activate the project's environment first:\n"
             "  Windows:       .venv-qrecon\\Scripts\\Activate.ps1\n  Linux / macOS: source .venv-qrecon/bin/activate")
from qrecon import questconfig as QC  # noqa: E402
from qrecon.tracking_profiles import PROFILES  # noqa: E402


def runs_now():
    d = HERE / "runs"
    return set(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("qrecon_")) if d.is_dir() else set()


def summarize(tracking_dir):
    """The sweep's numbers for one tracking run."""
    d = Path(tracking_dir)
    cov = json.loads((d / "coverage.json").read_text())
    maps = json.loads((d / "maps.json").read_text()) if (d / "maps.json").is_file() else []
    stats = json.loads((d / "stats.json").read_text()) if (d / "stats.json").is_file() else {}
    saved = [m for m in maps if m.get("status") == "saved"]
    lengths = [m["last_t"] - m["first_t"] for m in saved]
    span = cov["processed"]["span_s"]
    return {"maps_saved": len(saved), "maps_discarded": sum(m.get("status") == "discarded" for m in maps),
            "breaks": stats.get("map_restarts", max(0, len(saved) - 1)),
            "tracked_share_whole": cov["tracked_share_whole"], "tracked_share_from_init": cov["tracked_share_from_init"],
            "longest_map_s": round(max(lengths), 1) if lengths else 0.0,
            "longest_map_share": round(max(lengths) / span, 3) if lengths and span else 0.0,
            "maps_10s_or_more": sum(x >= 10.0 for x in lengths), "track_ms_median": stats.get("track_ms_median")}


def table(rows):
    head = f"{'preset':22s} {'maps':>5s} {'discarded':>9s} {'breaks':>6s} {'tracked':>8s} {'from init':>9s} {'longest map':>14s} {'maps >=10 s':>11s}"
    lines = [head]
    for name, r in rows:
        lines.append(f"{name:22s} {r['maps_saved']:5d} {r['maps_discarded']:9d} {r['breaks']:6d} {r['tracked_share_whole']:8.3f} "
                     f"{r['tracked_share_from_init']:9.3f} {r['longest_map_s']:7.1f} s ({100 * r['longest_map_share']:2.0f}%) "
                     f"{r['maps_10s_or_more']:11d}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--calib", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    py, vt, data = sys.executable, HERE / "scripts" / "video_tools.py", HERE / "data"
    data.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    calib = json.loads(Path(a.calib).read_text())
    out = HERE / "runs" / f"{a.name}_tracking_sweep.json"
    results = json.loads(out.read_text()) if out.is_file() else {}
    rows = []
    for name in sorted(a.profiles, key=lambda n: PROFILES[n]["every"], reverse=True):     # 15 fps first, then 30 fps
        prof = PROFILES[name]
        fps = video_fps / prof["every"]
        frames = data / f"{a.name}_{int(round(fps))}fps"
        if not (frames / "rgb.txt").is_file():
            subprocess.run([py, vt, "frames", "--video", a.video, "--every", str(prof["every"]), "--scale", str(a.scale),
                            "--out", str(frames)], cwd=HERE, check=True)
        elif not (frames / "source_times.csv").is_file():
            subprocess.run([py, vt, "source-times", "--video", a.video, "--frames", str(frames)], cwd=HERE, check=True)
        cam = data / f"{a.name}_camera_{name}.yaml"
        cam.write_text(QC.stella_camera_yaml(calib, a.scale, round(fps, 3), name=a.name, overrides=prof["stella"]))
        print(f"\n== {name}: {prof['about']}", flush=True)
        before = runs_now()
        r = subprocess.run([py, str(HERE / "tools" / "track_on_quest.py"), "--frames", str(frames), "--camera", str(cam),
                            "--name", f"{a.name}_{int(round(fps))}fps", "--restart-after", str(prof["restart_after"])], cwd=HERE)
        made = sorted(runs_now() - before)
        if r.returncode != 0 or not made:
            print(f"  {name}: tracking failed (exit {r.returncode})")
            continue
        s = summarize(HERE / "runs" / made[-1] / "tracking")
        results[name] = dict(s, tracking_run=made[-1], frames=str(frames))
        out.write_text(json.dumps(results, indent=1))
    for name in a.profiles:
        if name in results:
            rows.append((name, results[name]))
    print("\n" + table(rows))
    for name, r in rows:
        print(f"  {name}: tracking run {r['tracking_run']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
