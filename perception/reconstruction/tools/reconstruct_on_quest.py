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
from qrecon import registry as REG, questconfig as QC, config as C  # noqa: E402

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
    n_local = sum(1 for l in open(Path(local) / "rgb.txt") if l.strip() and not l.startswith("#"))
    n_remote = sum(1 for l in remote(f"{REMOTE}/frames/{name}/rgb.txt").splitlines() if l.strip() and not l.startswith("#"))
    if n_local != n_remote:
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
    ap.add_argument("--tracking-run"); ap.add_argument("--max-keyframes", type=int); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-loop", action="store_true"); ap.add_argument("--purpose", default="reconstruction on the headset")
    a = ap.parse_args()
    for b in ("qtrack", "qrecon_quest"):
        if not (RUNTIME / "bin" / b).is_file():
            sys.exit("runtime not built: run python tools/build_tracking.py first")
    if bool(a.recording) != bool(a.board_frames):
        sys.exit("--recording and --board-frames go together")
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
                                "tracking_run": a.tracking_run, "max_keyframes": a.max_keyframes, "threads": a.threads})
    path = Path(path)
    rout = f"{REMOTE}/out/{run_id}"
    adb("shell", f"mkdir -p {rout}/tracking {rout}/recon")
    device = {"foreground_app": adb("shell", "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'", check=False).strip(),
              "thermal_before": thermal()}

    # tracking
    t0 = time.time()
    if a.tracking_run:
        traj = HERE / "runs" / a.tracking_run / "tracking" / "frame_trajectory.txt"
        if not traj.is_file():
            sys.exit(f"{traj} not found")
        adb("push", str(traj), f"{rout}/tracking/frame_trajectory.txt", capture=False)
        device["tracking"] = f"reused from {a.tracking_run}"
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
    if a.recording:
        anchor = {"board_frames": f"{REMOTE}/frames/{a.name}_boards/rgb.txt", "camera": QC.camera_block(calib, 1.0),
                  "recording": a.recording}
    cfg = QC.build(f"{REMOTE}/frames/{a.name}/rgb.txt", f"{rout}/tracking/frame_trajectory.txt",
                   QC.camera_block(calib, a.frame_scale), {"source": "onnx", "model": rmodel, "threads": a.threads},
                   anchor=anchor, max_keyframes=a.max_keyframes)
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
    if not res.endswith("exit=0"):
        sys.exit("reconstruction failed on the headset; send me runs/<run>/recon/stdout.txt and anchor.json")
    print(f"cloud: {rec / 'cloud.ply'}")


if __name__ == "__main__":
    main()
