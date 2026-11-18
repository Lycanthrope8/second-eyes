#!/usr/bin/env python3
"""One recording, end to end, with nothing recording-specific to enter.

    python tools/process_recording.py --video 20261111_121118.mp4 --calib calib_s26_lines.json --name h121118

Steps (each reuses earlier output when it is there):
  1. Where each board is seen: video_tools.py board-spans -> data/<name>_boards.json
  2. Tracking frames (default every 2nd frame, half size) -> data/<name>_<fps>fps, with source_times.csv
  3. Full-size frames inside the board windows -> data/<name>_autoboards
  4. The tracker's camera file, from --calib -> data/<name>_camera.yaml
  5. Tracking on the headset, starting a new map whenever tracking has been lost for --restart-after seconds;
     the coverage report (D135) is printed
  6. Each saved map that tracked at least --min-map-s seconds and saw a board is reconstructed on the headset and
     anchored by whichever board it saw (best effort); maps that saw no board are listed as unplaced
  7. runs/<name>_recording.json: coverage, every map and what happened to it, the reconstruction runs
Then place the maps that saw no board, through the views they share with placed maps of any recording, and join:
  python tools/place_maps.py --recordings runs/<name>_recording.json [...]
  python tools/merge_clouds.py --recordings runs/<name>_recording.json [...] --out NAME

--focal-px replaces the calibration's focal length in both the tracker and the reconstruction: a diagnostic for how
sensitive the results are to it (it cannot tell which component causes an error)."""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
try:
    import cv2, numpy, yaml  # noqa: F401,E401  (the steps need the project's environment)
except ModuleNotFoundError as e:
    sys.exit(f"{e.name} is not installed in this Python. Activate the project's environment first:\n"
             "  Windows:       .venv-qrecon\\Scripts\\Activate.ps1\n  Linux / macOS: source .venv-qrecon/bin/activate")
from qrecon import questconfig as QC  # noqa: E402


def run(cmd):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"step failed (exit {r.returncode}): {' '.join(str(c) for c in cmd[:3])} ...")


def runs_now():
    d = HERE / "runs"
    return set(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("qrecon_")) if d.is_dir() else set()


def new_run(before):
    made = sorted(runs_now() - before)
    if not made:
        sys.exit("the step finished but made no run folder")
    return made[-1]


def plan_maps(maps, windows, min_map_s):
    """Which saved maps to reconstruct: (map record, boards seen while it tracked, reason it is skipped or None)."""
    out = []
    for m in maps:
        if m.get("status") != "saved":
            continue
        a, b = m.get("first_t", -1), m.get("last_t", -1)
        seen = sorted(k for k, ws in windows.items() if any(w0 <= b and w1 >= a for w0, w1 in ws))
        why = None
        if b - a < min_map_s:
            why = f"tracked for {b - a:.1f} s, under --min-map-s {min_map_s}"
        elif not seen:
            why = "no board seen while this map was tracking: it cannot be placed"
        out.append((m, seen, why))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--calib", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--every", type=int, default=2, help="tracking frames: every Nth video frame")
    ap.add_argument("--scale", type=float, default=0.5, help="tracking frames: size factor")
    ap.add_argument("--model", default=str(Path("models") / "dav2s_960x540.onnx"))
    ap.add_argument("--max-keyframes", type=int, default=60, help="per map, to bound the time on the headset")
    ap.add_argument("--min-map-s", type=float, default=5.0)
    ap.add_argument("--restart-after", type=float, default=2.0)
    ap.add_argument("--tracking-run", help="reuse this tracking run instead of tracking again")
    ap.add_argument("--skip-reconstruction", action="store_true", help="stop after tracking and the coverage report")
    ap.add_argument("--focal-px", type=float, help="diagnostic: replace the calibration's focal length")
    a = ap.parse_args()
    py, vt = sys.executable, HERE / "scripts" / "video_tools.py"
    video, data = Path(a.video), HERE / "data"
    data.mkdir(exist_ok=True)
    calib = json.loads(Path(a.calib).read_text())
    calib_used = Path(a.calib)
    if a.focal_px:
        calib = dict(calib, K=[[a.focal_px, 0, calib["K"][0][2]], [0, a.focal_px, calib["K"][1][2]], [0, 0, 1]])
        calib_used = data / f"{a.name}_calib_f{int(a.focal_px)}.json"
        calib_used.write_text(json.dumps(calib, indent=1))
        print(f"diagnostic: focal length {a.focal_px} px -> {calib_used}")
    t0 = time.time()

    spans = data / f"{a.name}_boards.json"
    if not spans.is_file():
        run([py, vt, "board-spans", "--video", video, "--out", spans])
    windows = QC.merge_windows(json.loads(spans.read_text())["windows_s"])

    cap = cv2.VideoCapture(str(video))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / a.every
    cap.release()
    frames = data / f"{a.name}_{int(round(fps))}fps"
    if not (frames / "rgb.txt").is_file():
        run([py, vt, "frames", "--video", video, "--every", a.every, "--scale", a.scale, "--out", frames])
    elif not (frames / "source_times.csv").is_file():
        run([py, vt, "source-times", "--video", video, "--frames", frames])
    boards = data / f"{a.name}_autoboards"
    if any(windows.values()) and not (boards / "rgb.txt").is_file():
        run([py, vt, "frames", "--video", video, "--every", 2, "--board-spans", spans, "--out", boards])
    cam = data / (f"{a.name}_camera.yaml" if not a.focal_px else f"{a.name}_camera_f{int(a.focal_px)}.yaml")
    cam.write_text(QC.stella_camera_yaml(calib, a.scale, round(fps, 3), name=a.name))

    trk = a.tracking_run
    if not trk:
        before = runs_now()
        run([py, HERE / "tools" / "track_on_quest.py", "--frames", frames, "--camera", cam, "--name", a.name,
             "--restart-after", a.restart_after])
        trk = new_run(before)
    tdir = HERE / "runs" / trk / "tracking"
    maps = json.loads((tdir / "maps.json").read_text()) if (tdir / "maps.json").is_file() else []
    coverage = json.loads((tdir / "coverage.json").read_text()) if (tdir / "coverage.json").is_file() else None
    summary = {"video": video.name, "name": a.name, "calib": str(calib_used), "tracking_run": trk, "coverage": coverage,
               "board_windows_s": windows, "frames": str(frames), "frame_scale": a.scale, "fps": round(fps, 3),
               "model": a.model, "max_keyframes": a.max_keyframes, "maps": [], "recon_runs": []}
    for m, seen, why in plan_maps(maps, windows, a.min_map_s):
        entry = {"map": m["map"], "first_t": m["first_t"], "last_t": m["last_t"], "tracked_frames": m["tracked_frames"],
                 "boards_seen": seen}
        if why or a.skip_reconstruction:
            entry["skipped"] = why or "--skip-reconstruction"
        else:
            before = runs_now()
            run([py, HERE / "tools" / "reconstruct_on_quest.py", "--name", a.name, "--frames", frames, "--board-frames", boards,
                 "--board-spans", spans, "--calib", calib_used, "--frame-scale", a.scale, "--fps", round(fps, 3),
                 "--model", a.model, "--tracking-run", trk, "--map", m["map"], "--anchor", "best-effort",
                 "--max-keyframes", a.max_keyframes])
            rid = new_run(before)
            rec = HERE / "runs" / rid / "recon"
            an = json.loads((rec / "anchor.json").read_text()) if (rec / "anchor.json").is_file() else {}
            st = json.loads((rec / "stats.json").read_text()) if (rec / "stats.json").is_file() else {}
            entry.update(recon_run=rid, frame=an.get("frame"), warnings=an.get("warnings", []),
                         keyframes=st.get("keyframes"), valid_keyframe_share=st.get("valid_keyframe_share"))
            summary["recon_runs"].append(rid)
        summary["maps"].append(entry)
    out = HERE / "runs" / f"{a.name}_recording.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\n{a.name}: tracking {trk}, {len(summary['maps'])} saved map(s), {len(summary['recon_runs'])} reconstructed, "
          f"{(time.time() - t0) / 60:.0f} min")
    for e in summary["maps"]:
        what = (f"skipped: {e['skipped']}" if "skipped" in e else
                f"{e['recon_run']}, frame {e['frame']}, valid keyframes {e['valid_keyframe_share']}")
        print(f"  map {e['map']} ({e['first_t']:.1f}-{e['last_t']:.1f} s, boards {', '.join(e['boards_seen']) or 'none'}): {what}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
