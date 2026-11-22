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
With --one-map (D142) the pieces that see a board are only anchored, then tools/one_map.py joins all the pieces into one
map and reconstructs it once. A board anchor whose trajectory implies an impossible camera speed (--max-speed) is redone for the stretch around the
board only. --recheck SUMMARY ... applies that check to reconstructions made before it existed.
Then place the maps that saw no board, through the views they share with placed maps of any recording, and join:
  python tools/place_maps.py --recordings runs/<name>_recording.json [...]
  python tools/merge_clouds.py --recordings runs/<name>_recording.json [...] --out NAME

--focal-px replaces the calibration's focal length in both the tracker and the reconstruction: a diagnostic for how
sensitive the results are to it (it cannot tell which component causes an error)."""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
try:
    import cv2, numpy, yaml  # noqa: F401,E401  (the steps need the project's environment)
except ModuleNotFoundError as e:
    sys.exit(f"{e.name} is not installed in this Python. Activate the project's environment first:\n"
             "  Windows:       .venv-qrecon\\Scripts\\Activate.ps1\n  Linux / macOS: source .venv-qrecon/bin/activate")
from qrecon import questconfig as QC, mapalign as MA  # noqa: E402
from qrecon.tracking_profiles import PROFILES  # noqa: E402


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


def anchored_speed(rid):
    """Camera speed (m/s, 95th percentile) along a reconstruction's anchored trajectory, or None."""
    p = HERE / "runs" / rid / "recon" / "anchored_trajectory.txt"
    if not p.is_file():
        return None
    t, T = MA.read_tum(p)
    return MA.implied_speed(t, T[:, :3, 3], 1.0)


def board_span(first_t, last_t, board_windows, pad_s):
    """The stretch of a map around the windows in which its anchoring board was seen, or None."""
    ws = [w for w in board_windows if w[0] <= last_t and w[1] >= first_t]
    if not ws:
        return None
    return (max(first_t, min(w[0] for w in ws) - pad_s), min(last_t, max(w[1] for w in ws) + pad_s))


def trimmed_trajectory(src, span, dst):
    lo, hi = span
    lines = [l for l in open(src) if l.strip() and not l.startswith("#") and lo - 1e-6 <= float(l.split()[0]) <= hi + 1e-6]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("".join(lines))
    return dst


def reconstruct_board_map(ctx, m, first_t, last_t, windows, max_speed, pad_s):
    """Reconstructs a map anchored by a board. If the anchored trajectory implies an impossible camera speed (the
    tracker's scale drifted inside the map, so the board's scale holds only near the board), the map is reconstructed
    again limited to the stretch around the board. Returns (run id or None, fields for the summary)."""
    def once(trajectory=None):
        before = runs_now()
        run([sys.executable, HERE / "tools" / "reconstruct_on_quest.py", "--name", ctx["name"], "--frames", ctx["frames"],
             "--board-frames", ctx["boards"], "--board-spans", ctx["spans"], "--calib", ctx["calib"],
             "--frame-scale", ctx["scale"], "--fps", ctx["fps"], "--model", ctx["model"], "--tracking-run", ctx["trk"],
             "--map", m, "--anchor", "best-effort", "--max-keyframes", ctx["max_keyframes"]]
            + (["--trajectory", trajectory] if trajectory else []))
        return new_run(before)
    rid = once()
    return check_board_map(ctx, m, rid, first_t, last_t, windows, max_speed, pad_s, once)


def check_board_map(ctx, m, rid, first_t, last_t, windows, max_speed, pad_s, once):
    an = json.loads((HERE / "runs" / rid / "recon" / "anchor.json").read_text())
    frame, speed = an.get("frame"), anchored_speed(rid)
    if frame in (None, "tracker") or speed is None or speed <= max_speed:
        return rid, {"kept": "full", "speed_p95_mps": None if speed is None else round(speed, 2)}
    span = board_span(first_t, last_t, windows.get(frame, []), pad_s)
    if span is None:
        return None, {"skipped": f"its board anchor implies a camera speed of {speed:.1f} m/s, and no stretch around the board to keep"}
    print(f"  map {m}: its anchor implies {speed:.1f} m/s over the whole map; reconstructing {span[0]:.1f}-{span[1]:.1f} s only")
    traj = trimmed_trajectory(HERE / "runs" / ctx["trk"] / "tracking" / ctx["map_dirs"].get(m, f"maps/map_{m:02d}") / "frame_trajectory.txt",
                              span, HERE / "runs" / "trims" / f"{ctx['name']}_map_{m:02d}_trajectory.txt")
    rid2 = once(traj)
    speed2 = anchored_speed(rid2)
    if speed2 is not None and speed2 <= max_speed:
        return rid2, {"kept": "trimmed", "span_s": [round(span[0], 3), round(span[1], 3)], "speed_p95_mps": round(speed2, 2),
                      "replaced_run": rid}
    return None, {"skipped": f"its board anchor implies a camera speed of {speed:.1f} m/s, still {speed2 or 0:.1f} m/s when trimmed"}


def recheck(summary_paths, max_speed, pad_s, model, max_keyframes):
    """Applies the speed check to the board-anchored reconstructions already in the given summaries."""
    sys.path.insert(0, str(HERE / "tools"))
    import place_maps as PM
    for sp in summary_paths:
        rec = PM.load_recording(sp)
        S = rec["summary"]
        name = S["name"]
        ctx = {"name": name, "frames": rec["frames"], "boards": HERE / "data" / f"{name}_autoboards",
               "spans": HERE / "data" / f"{name}_boards.json", "calib": S["calib"], "scale": rec["scale"], "fps": rec["fps"],
               "model": S.get("model", model), "trk": S["tracking_run"], "max_keyframes": S.get("max_keyframes", max_keyframes),
               "map_dirs": {k: v.get("dir") for k, v in rec["maps"].items() if v.get("dir")}}
        windows = QC.merge_windows(S.get("board_windows_s", {}))
        changed = False
        for e in S["maps"]:
            rid = e.get("recon_run")
            if not rid or e.get("placed_by") or e.get("frame") in (None, "tracker") or e.get("kept"):
                continue

            def once(trajectory=None, m=e["map"]):
                before = runs_now()
                run([sys.executable, HERE / "tools" / "reconstruct_on_quest.py", "--name", name, "--frames", ctx["frames"],
                     "--board-frames", ctx["boards"], "--board-spans", ctx["spans"], "--calib", ctx["calib"],
                     "--frame-scale", ctx["scale"], "--fps", ctx["fps"], "--model", ctx["model"], "--tracking-run", ctx["trk"],
                     "--map", m, "--anchor", "best-effort", "--max-keyframes", ctx["max_keyframes"]]
                    + (["--trajectory", trajectory] if trajectory else []))
                return new_run(before)
            new, fields = check_board_map(ctx, e["map"], rid, e["first_t"], e["last_t"], windows, max_speed, pad_s, once)
            print(f"{name} map {e['map']} ({rid}): " + ("speed plausible, kept" if new == rid else
                  f"replaced by {new}, {fields.get('span_s')}" if new else f"left out: {fields['skipped']}"))
            if new == rid:
                e.update(fields)
                changed = True
                continue
            runs = S.setdefault("recon_runs", [])
            if rid in runs:
                runs.remove(rid)
            if new:
                e.update(fields, recon_run=new)
                st = HERE / "runs" / new / "recon" / "stats.json"
                stats = json.loads(st.read_text()) if st.is_file() else {}
                e.update(keyframes=stats.get("keyframes"), valid_keyframe_share=stats.get("valid_keyframe_share"))
                runs.append(new)
            else:
                for k in ("recon_run", "frame", "keyframes", "valid_keyframe_share", "warnings"):
                    e.pop(k, None)
                e.update(fields, replaced_run=rid)
            changed = True
        if changed:
            shutil.copy2(sp, str(sp) + ".bak")
            Path(sp).write_text(json.dumps(S, indent=1))
            print(f"-> {sp} (previous version kept as .bak)")


def anchor_entry(rid, first_t, last_t, windows, max_speed, pad_s):
    """Summary fields for a map anchored by an anchor-only run (one-map mode): the frame, whether the anchor passed its
    checks, and the stretch kept if the anchored path implies an impossible camera speed."""
    an = json.loads((HERE / "runs" / rid / "recon" / "anchor.json").read_text())
    F = an.get("frame")
    ok = F not in (None, "tracker") and bool(an.get(F, {}).get("ok"))
    fields = {"anchor_run": rid, "frame": F, "anchor_ok": ok, "warnings": an.get("warnings", [])}
    speed = anchored_speed(rid) if F not in (None, "tracker") else None
    fields["speed_p95_mps"] = None if speed is None else round(speed, 2)
    if speed is not None and speed > max_speed:
        span = board_span(first_t, last_t, windows.get(F, []), pad_s)
        fields.update(kept="trimmed", span_s=[round(span[0], 3), round(span[1], 3)]) if span else fields.update(anchor_ok=False)
        if not span:
            fields["warnings"] = fields["warnings"] + [f"anchor implies {speed:.1f} m/s and no stretch around the board to keep"]
    else:
        fields["kept"] = "full"
    return fields


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
    ap.add_argument("--video"); ap.add_argument("--calib"); ap.add_argument("--name")
    ap.add_argument("--profile", choices=sorted(PROFILES),
                    help="tracking preset (qrecon/tracking_profiles.py); sets --every, --restart-after and tracker settings")
    ap.add_argument("--one-map", action="store_true",
                    help="one map for the video (D142): anchor the pieces that see a board (anchor only), join all pieces "
                         "with tools/one_map.py and reconstruct the joined map once")
    ap.add_argument("--helpers", nargs="*", default=[], help="with --one-map: other processed recordings to match against")
    ap.add_argument("--use-link", nargs="*", default=[], metavar="RUN", help="with --one-map: runs whose board links to use (D140)")
    ap.add_argument("--one-map-keyframes", type=int, default=200, help="with --one-map: keyframes for the one reconstruction")
    ap.add_argument("--recheck", nargs="+", metavar="SUMMARY",
                    help="only apply the speed check to the board-anchored reconstructions in these recording summaries")
    ap.add_argument("--max-speed", type=float, default=2.0, help="m/s; a board anchor implying faster camera motion is trimmed")
    ap.add_argument("--trim-pad-s", type=float, default=10.0, help="seconds kept either side of the board when trimming")
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
    if a.recheck:
        return recheck(a.recheck, a.max_speed, a.trim_pad_s, a.model, a.max_keyframes)
    if not (a.video and a.calib and a.name):
        sys.exit("give --video, --calib and --name (or --recheck SUMMARY ...)")
    overrides = {}
    if a.profile:
        prof = PROFILES[a.profile]
        a.every, a.restart_after, overrides = prof["every"], prof["restart_after"], prof["stella"]
        print(f"tracking profile {a.profile}: {prof['about']}")
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
    cam.write_text(QC.stella_camera_yaml(calib, a.scale, round(fps, 3), name=a.name, overrides=overrides))

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
               "model": a.model, "max_keyframes": a.max_keyframes, "profile": a.profile or "custom",
               "maps": [], "recon_runs": []}
    for m, seen, why in plan_maps(maps, windows, a.min_map_s):
        entry = {"map": m["map"], "first_t": m["first_t"], "last_t": m["last_t"], "tracked_frames": m["tracked_frames"],
                 "boards_seen": seen}
        if a.one_map:
            if seen and m["last_t"] - m["first_t"] >= a.min_map_s:
                before = runs_now()
                run([py, HERE / "tools" / "reconstruct_on_quest.py", "--name", a.name, "--frames", frames, "--board-frames", boards,
                     "--board-spans", spans, "--calib", calib_used, "--frame-scale", a.scale, "--fps", round(fps, 3),
                     "--model", a.model, "--tracking-run", trk, "--map", m["map"], "--anchor", "best-effort", "--anchor-only"])
                entry.update(anchor_entry(new_run(before), m["first_t"], m["last_t"], windows, a.max_speed, a.trim_pad_s))
            elif m["last_t"] - m["first_t"] < a.min_map_s:
                entry["skipped"] = f"tracked for {m['last_t'] - m['first_t']:.1f} s, under --min-map-s {a.min_map_s}"
        elif why or a.skip_reconstruction:
            entry["skipped"] = why or "--skip-reconstruction"
        else:
            ctx = {"name": a.name, "frames": frames, "boards": boards, "spans": spans, "calib": calib_used, "scale": a.scale,
                   "fps": round(fps, 3), "model": a.model, "trk": trk, "max_keyframes": a.max_keyframes,
                   "map_dirs": {k["map"]: k.get("dir") for k in maps if k.get("dir")}}
            rid, fields = reconstruct_board_map(ctx, m["map"], m["first_t"], m["last_t"], windows, a.max_speed, a.trim_pad_s)
            entry.update(fields)
            if rid:
                rec = HERE / "runs" / rid / "recon"
                an = json.loads((rec / "anchor.json").read_text()) if (rec / "anchor.json").is_file() else {}
                st = json.loads((rec / "stats.json").read_text()) if (rec / "stats.json").is_file() else {}
                entry.update(recon_run=rid, frame=an.get("frame"), warnings=an.get("warnings", []),
                             keyframes=st.get("keyframes"), valid_keyframe_share=st.get("valid_keyframe_share"))
                summary["recon_runs"].append(rid)
        summary["maps"].append(entry)
    out = HERE / "runs" / f"{a.name}_recording.json"
    out.write_text(json.dumps(summary, indent=1))
    if a.one_map:
        run([py, HERE / "tools" / "one_map.py", "--recording", out, "--max-keyframes", a.one_map_keyframes, "--model", a.model]
            + (["--helpers", *a.helpers] if a.helpers else []) + (["--use-link", *a.use_link] if a.use_link else []))
        summary = json.loads(out.read_text())
    print(f"\n{a.name}: tracking {trk}, {len(summary['maps'])} saved map(s), {len(summary['recon_runs'])} reconstructed, "
          f"{(time.time() - t0) / 60:.0f} min")
    for e in summary["maps"]:
        kept = f", kept {e['span_s'][0]:.1f}-{e['span_s'][1]:.1f} s only" if e.get("kept") == "trimmed" else ""
        what = (f"skipped: {e['skipped']}" if "skipped" in e else
                f"{e['recon_run']}, frame {e['frame']}, valid keyframes {e['valid_keyframe_share']}{kept}" if e.get("recon_run") else
                (f"anchored by {e['frame']} ({'checks passed' if e.get('anchor_ok') else 'failed its checks'}, {e['anchor_run']}){kept}"
                 if e.get("frame") not in (None, "tracker") else f"no board could be fitted ({e['anchor_run']}): to be joined by shared views")
                if e.get("anchor_run") else "no board: to be joined by shared views")
        print(f"  map {e['map']} ({e['first_t']:.1f}-{e['last_t']:.1f} s, boards {', '.join(e['boards_seen']) or 'none'}): {what}")
    om = summary.get("one_map")
    if om:
        share = f" ({100 * om['share_of_video']:.0f}% of the video)" if om.get("share_of_video") else ""
        print(f"one map: {om.get('recon_run')}, {len(om['pieces'])} piece(s) joined in {om['frame']}'s frame, {om['joined_s']} s{share}; "
              f"{len(om['left_out'])} left out; valid keyframes {om.get('valid_keyframe_share')}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
