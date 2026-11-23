#!/usr/bin/env python3
"""Reconstruct a recording on the Quest: tracking, anchoring, depth, calibration, filtering and fusion all run on
the headset; this script only copies inputs over adb, starts the two programs and brings the results back.

House video (anchored with its ChArUco boards):
    python tools/reconstruct_on_quest.py --name house --frames data/house_15fps --board-frames data/house_boards
        --recording house_20261110_191353 --calib calib_s26.json --frame-scale 0.5 --fps 15
        --model models/dav2s_960x540.onnx
Public sequence (no anchor; scored afterwards with qrecon.py evaluate-cloud, diagnostic alignment):
    python tools/reconstruct_on_quest.py --name tum_fr3 --frames data/tum/rgbd_dataset_freiburg3_long_office_household
        --seq fr3_long_office --model models/dav2s_640x480.onnx

--tracking-run RUN reuses a trajectory from an earlier run instead of tracking again. Start the Second Eyes app with
its detector before running, so the programs share the headset with its normal load."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from qrecon import framelist as FL, registry as REG, questconfig as QC, config as C  # noqa: E402


def apply_settings(cfg, settings):
    """Overrides of the reconstruction settings, as SECTION.KEY=VALUE (VALUE read as JSON when it can be: numbers,
    lists, true/false), for instance calibration.max_holdout_rel_err=0.15. Returns the overrides applied."""
    done = {}
    for item in settings or []:
        key, _, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        if not raw or not name or section not in cfg or not isinstance(cfg[section], dict):
            sys.exit(f"--set {item}: expected SECTION.KEY=VALUE with SECTION one of "
                     f"{', '.join(k for k, v in cfg.items() if isinstance(v, dict))}")
        if name not in cfg[section]:
            sys.exit(f"--set {item}: {section} has no setting {name} (it has {', '.join(cfg[section])})")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        cfg[section][name] = value
        done[f"{section}.{name}"] = value
    return done


def spread_keyframes(src, dst, max_n, group=4):
    """The headset program takes keyframes in time order and stops at max_keyframes, which would leave the end of a long
    path out. For a path placed by a fixed similarity (no board fit needs its dense poses), keep only the poses the
    keyframe rule picks (configs/qrecon.yaml), thinned to max_n in groups of consecutive keyframes spread evenly over the
    whole path. Returns (candidates, kept)."""
    import numpy as np
    from qrecon import keyframes as KFS
    from qrecon.mapalign import read_tum
    lines = [l for l in open(src) if l.strip() and not l.startswith("#")]
    t, T = read_tum(src)
    kc = C.load(os.path.join(str(HERE), "configs", "qrecon.yaml"))["keyframes"]
    idx = KFS.select(T, t, kc["min_trans_m"], kc["min_rot_deg"], kc["min_dt_s"])
    n_all = len(idx)
    if n_all > max_n:
        # in groups of consecutive keyframes: each keyframe's depth calibration triangulates against its neighbours
        # (configs/qrecon.yaml, landmarks.neighbour_offsets -2..2), so neighbours must stay close; the groups are spread
        # evenly over the whole path, the last group ending at the path's end
        g = max(1, min(group, max_n))
        starts = np.round(np.linspace(0, n_all - g, max(1, max_n // g))).astype(int)
        idx = [idx[c] for c in sorted({int(s0) + k for s0 in starts for k in range(g)})]
    Path(dst).write_text("".join(lines[i] for i in idx))
    return n_all, len(idx)

REMOTE = "/data/local/tmp/qtrack"
RUNTIME = HERE / "tracking" / "out" / "android-arm64"


def adb(*args, check=True, capture=True):
    r = subprocess.run(["adb", *args], capture_output=capture, text=True)
    if check and r.returncode != 0:
        sys.exit(f"adb failed: {' '.join(args[:3])} ...\n{r.stderr if capture else ''}")
    return r.stdout if capture else ""


def remote(path):
    return subprocess.run(["adb", "shell", f"cat {path} 2>/dev/null"], capture_output=True, text=True).stdout.strip()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def push_frames(local, name):
    local_list = (Path(local) / "rgb.txt").read_text(encoding="utf-8", errors="replace")
    n_local = FL.count(local_list)
    if FL.differs(local_list, remote(f"{REMOTE}/frames/{name}/rgb.txt")):
        print(f"pushing {n_local} frames of {name} (this can take a while)")
        adb("shell", f"rm -rf {REMOTE}/frames/{name} && mkdir -p {REMOTE}/frames")
        adb("push", str(local), f"{REMOTE}/frames/{name}", capture=False)
    return n_local


def thermal():
    return adb("shell", "dumpsys thermalservice | grep -m3 -E 'Thermal Status|mStatus|Status'", check=False).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True); ap.add_argument("--frames", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--calib"); ap.add_argument("--frame-scale", type=float, default=1.0); ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seq", help="a public sequence from configs/sequences.yaml (its intrinsics; no anchor)")
    ap.add_argument("--recording"); ap.add_argument("--board-frames")
    ap.add_argument("--board-spans", help="board windows found by video_tools.py board-spans (instead of --recording)")
    ap.add_argument("--map", type=int, help="with --tracking-run: reconstruct this map of a multi-map tracking run")
    ap.add_argument("--trajectory", help="with --tracking-run: use this trajectory file (for instance part of a map) instead "
                                         "of the run's own")
    ap.add_argument("--fixed-anchor", help="a JSON file {frame, sim3: {s, R, t}, by}: put the cloud into that frame with this "
                                           "similarity instead of fitting a board (tools/place_maps.py writes these)")
    ap.add_argument("--tracking-run"); ap.add_argument("--max-keyframes", type=int); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--map-points", help="the tracker's 3-D points in the trajectory's frame (one_map writes them), for "
                                         "calibrating keyframes that have nothing to triangulate")
    ap.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                    help="change a reconstruction setting for this run (repeatable), e.g. calibration.max_holdout_rel_err=0.15")
    ap.add_argument("--no-loop", action="store_true"); ap.add_argument("--purpose", default="reconstruction on the headset")
    ap.add_argument("--anchor-only", action="store_true",
                    help="compute anchor.json only (seconds): for instance a board link from a run whose cloud already exists")
    ap.add_argument("--anchor", choices=["strict", "best-effort"], default="strict",
                    help="strict: refuse when an anchor check fails; best-effort: anchor by whichever board can be fitted and "
                         "record failed checks as warnings, so a reconstruction is always made")
    a = ap.parse_args()
    for b in ("qtrack", "qrecon_quest"):
        if not (RUNTIME / "bin" / b).is_file():
            sys.exit("runtime not built: run python tools/build_tracking.py first")
    if a.recording and a.board_spans:
        sys.exit("give --recording or --board-spans, not both")
    if a.fixed_anchor and (a.recording or a.board_spans or a.board_frames):
        sys.exit("--fixed-anchor replaces --recording, --board-spans and --board-frames")
    if bool(a.recording or a.board_spans) != bool(a.board_frames):
        sys.exit("--board-frames goes with --recording or --board-spans")
    if a.map is not None and not a.tracking_run:
        sys.exit("--map needs --tracking-run")
    if a.seq:
        cam = C.load(str(HERE / "configs" / "sequences.yaml"))["sequences"][a.seq]["camera"]
        calib = {"K": [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], "dist": cam["dist"], "size": [cam["width"], cam["height"]]}
    elif a.calib:
        calib = json.load(open(a.calib))
    else:
        sys.exit("give --calib (a recording) or --seq (a public sequence)")
    if a.seq and any(calib["dist"]):
        print("note: this sequence's frames are distorted; the tracker models the distortion, the reconstruction program "
              "assumes undistorted frames, so expect some error near the image edges")
    devs = [l for l in adb("devices").splitlines()[1:] if l.strip().endswith("device")]
    if len(devs) != 1:
        sys.exit(f"expected exactly one headset over adb, found {len(devs)}")

    build = json.loads((RUNTIME / "build.json").read_text())
    stamp = hashlib.sha256(json.dumps(build["files"], sort_keys=True).encode()).hexdigest()
    if remote(f"{REMOTE}/runtime.sha256") != stamp:
        print("pushing the reconstruction runtime")
        adb("shell", f"mkdir -p {REMOTE}")
        for d in ("bin", "lib", "share"):
            adb("push", str(RUNTIME / d), f"{REMOTE}/", capture=False)
        adb("shell", f"chmod 755 {REMOTE}/bin/qtrack {REMOTE}/bin/qrecon_quest && echo {stamp} > {REMOTE}/runtime.sha256")
    msha = sha(a.model)
    rmodel = f"{REMOTE}/models/{Path(a.model).name}"
    if remote(rmodel + ".sha256") != msha:
        print("pushing the depth model")
        adb("shell", f"mkdir -p {REMOTE}/models")
        adb("push", a.model, rmodel, capture=False)
        adb("shell", f"echo {msha} > {rmodel}.sha256")
    n_frames = push_frames(a.frames, a.name)
    if a.board_frames:
        push_frames(a.board_frames, a.name + "_boards")

    run_id, path = REG.new_run(str(HERE / "runs"), "reconstruction",
                               {"purpose": a.purpose, "name": a.name, "frames": a.frames, "board_frames": a.board_frames,
                                "recording": a.recording, "sequence": a.seq, "calib": a.calib, "frame_scale": a.frame_scale,
                                "model": Path(a.model).name, "model_sha256": msha, "runtime_build": build["files"],
                                "tracking_run": a.tracking_run, "max_keyframes": a.max_keyframes, "threads": a.threads,
                                "anchor_mode": a.anchor, "anchor_only": a.anchor_only, "map": a.map,
                                "board_spans": a.board_spans, "fixed_anchor": a.fixed_anchor, "trajectory": a.trajectory})
    path = Path(path)
    rout = f"{REMOTE}/out/{run_id}"
    adb("shell", f"mkdir -p {rout}/tracking {rout}/recon")
    device = {"foreground_app": adb("shell", "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'", check=False).strip(),
              "thermal_before": thermal()}

    # tracking
    t0 = time.time()
    if a.tracking_run:
        traj = HERE / "runs" / a.tracking_run / "tracking" / (f"maps/map_{a.map:02d}/frame_trajectory.txt" if a.map is not None
                                                                else "frame_trajectory.txt")
        if a.trajectory:
            traj = Path(a.trajectory)
        if not traj.is_file():
            sys.exit(f"{traj} not found")
        if a.fixed_anchor and a.max_keyframes:
            n_all, n_kept = spread_keyframes(traj, path / "keyframe_poses.txt", a.max_keyframes)
            device["keyframes_spread"] = {"candidates": n_all, "kept": n_kept}
            print(f"keyframes: {n_kept} of {n_all} candidates, in groups spread over the whole path")
            traj = path / "keyframe_poses.txt"
        adb("push", str(traj), f"{rout}/tracking/frame_trajectory.txt", capture=False)
        device["tracking"] = f"reused from {a.tracking_run}" + (f", map {a.map}" if a.map is not None else "")
    else:
        cam_yaml = path / "camera.yaml"
        cam_yaml.write_text(QC.stella_camera_yaml(calib, a.frame_scale, a.fps, a.name))
        adb("push", str(cam_yaml), f"{rout}/tracking/camera.yaml", capture=False)
        print(f"{run_id}: tracking {n_frames} frames on the headset")
        res = adb("shell", f"cd {REMOTE} && LD_LIBRARY_PATH={REMOTE}/lib ./bin/qtrack --config {rout}/tracking/camera.yaml "
                           f"--vocab share/orb_vocab.fbow --frames frames/{a.name}/rgb.txt --out {rout}/tracking"
                           f"{' --no-loop' if a.no_loop else ''} > {rout}/tracking/stdout.txt 2>&1; echo exit=$?", check=False).strip()
        device["tracking"] = res
        if not res.endswith("exit=0"):
            (path / "tracking").mkdir(exist_ok=True)
            adb("pull", f"{rout}/tracking/.", str(path / "tracking"), check=False)
            REG.save_json(str(path), "device.json", device)
            sys.exit("tracking failed on the headset; send me runs/<run>/tracking/stdout.txt")
    device["tracking_wall_s"] = time.time() - t0

    # reconstruction
    anchor = None
    if a.fixed_anchor:
        anchor = {"fixed": json.load(open(a.fixed_anchor))}
    elif a.recording or a.board_spans:
        anchor = {"board_frames": f"{REMOTE}/frames/{a.name}_boards/rgb.txt", "camera": QC.camera_block(calib, 1.0),
                  "mode": "best_effort" if a.anchor == "best-effort" else "strict"}
        if a.board_spans:
            anchor["board_windows"] = json.load(open(a.board_spans))["windows_s"]
        else:
            anchor["recording"] = a.recording
    cfg = QC.build(f"{REMOTE}/frames/{a.name}/rgb.txt", f"{rout}/tracking/frame_trajectory.txt",
                   QC.camera_block(calib, a.frame_scale), {"source": "onnx", "model": rmodel, "threads": a.threads},
                   anchor=anchor, max_keyframes=a.max_keyframes)
    if a.anchor_only:
        if not anchor:
            sys.exit("--anchor-only needs board frames and --recording or --board-spans")
        cfg["anchor_only"] = True                    # anchor.json only: no depth, no cloud
    if a.map_points:
        rmp = f"{rout}/recon/map_points.txt"
        adb("push", a.map_points, rmp, capture=False)
        cfg["map_points"] = {"file": rmp, "window_s": 1.0, "max_per_keyframe": 3000}
        device["map_points"] = a.map_points
    cfg.setdefault("calibration", {}).setdefault("model", "affine")      # older configs/qrecon.yaml lack it
    if a.set:
        device["settings_changed"] = apply_settings(cfg, a.set)
        print("settings changed: " + ", ".join(f"{k}={v}" for k, v in device["settings_changed"].items()))
    (path / "recon.json").write_text(json.dumps(cfg, indent=1))
    adb("push", str(path / "recon.json"), f"{rout}/recon/recon.json", capture=False)
    print(f"{run_id}: reconstructing on the headset")
    t1 = time.time()
    res = adb("shell", f"cd {REMOTE} && LD_LIBRARY_PATH={REMOTE}/lib ./bin/qrecon_quest --config {rout}/recon/recon.json "
                       f"--out {rout}/recon > {rout}/recon/stdout.txt 2>&1; echo exit=$?", check=False).strip()
    device.update(recon=res, recon_wall_s=time.time() - t1, thermal_after=thermal())
    for sub in ("tracking", "recon"):
        (path / sub).mkdir(exist_ok=True)
        adb("pull", f"{rout}/{sub}/.", str(path / sub), check=False)
    REG.save_json(str(path), "device.json", device)
    rec = path / "recon"
    summary = {"run": run_id}
    for f in ("stats.json", "anchor.json"):
        if (rec / f).is_file():
            summary[f[:-5]] = json.loads((rec / f).read_text())
    if (path / "tracking" / "stats.json").is_file():
        summary["tracking"] = json.loads((path / "tracking" / "stats.json").read_text())
    elif a.tracking_run and (HERE / "runs" / a.tracking_run / "tracking" / "stats.json").is_file():
        summary["tracking"] = {"reused_from": a.tracking_run,
                               **json.loads((HERE / "runs" / a.tracking_run / "tracking" / "stats.json").read_text())}
    REG.save_json(str(path), "summary.json", summary)
    print(json.dumps({k: summary.get(k) for k in ("run", "tracking", "stats")}, indent=1)[:3000])
    an = summary.get("anchor") or {}
    if an.get("frame"):
        print(f"anchor: cloud in {an['frame']}'s frame" + (f", by {an['anchored_by']['window']}" if an.get("anchored_by", {}).get("window") else ""))
        for b, L in (an.get("links") or {}).items():
            if "sim3_board_in_frame" in L:
                q = (f"single views, spread {1000 * L['position_spread_m']:.1f} mm" if "position_spread_m" in L
                     else f"board fit {1000 * L.get('corner_fit_rms_m', float('nan')):.1f} mm")
                print(f"link: {b} seen at {L['position_m']:.2f} m from {an['frame']} ({q})")
    for wmsg in an.get("warnings", []):
        print(f"warning: {wmsg}")
    if not res.endswith("exit=0"):
        sys.exit("reconstruction failed on the headset; send me runs/<run>/recon/stdout.txt and anchor.json")
    print(f"anchor only: {rec / 'anchor.json'}" if a.anchor_only else f"cloud: {rec / 'cloud.ply'}")


if __name__ == "__main__":
    main()
