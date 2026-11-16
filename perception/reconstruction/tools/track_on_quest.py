#!/usr/bin/env python3
"""Run the tracker on the Quest over adb and bring the results back.

    python tools/track_on_quest.py --frames DIR --camera CAM.yaml --name NAME [--no-loop] [--first N --last M] [--fast]

DIR holds rgb.txt and the images it lists (scripts/video_tools.py frames, or a TUM sequence folder). The runtime from
tools/build_tracking.py, the frames and the camera file are pushed to /data/local/tmp/qtrack/ (only what changed), qtrack
runs on the headset's CPU, and its trajectory, timing and memory come back into runs/<run ID>/tracking/.
For the resource test, start the Second Eyes app with its detector first, so tracking shares the headset with its
normal load; the app in the foreground is recorded.

Frames are fed at the rate they were captured at (from rgb.txt), as a live camera would deliver them. Fed faster, the
tracker outruns stella's mapping thread: fewer map points are made in new areas, poses get noisier and tracking is lost
more easily. --fast feeds them as fast as possible, for timing tests only."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from qrecon import registry as REG  # noqa: E402

REMOTE = "/data/local/tmp/qtrack"
RUNTIME = HERE / "tracking" / "out" / "android-arm64"


def adb(*args, check=True, capture=False):
    cmd = ["adb", *args]
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if check and r.returncode != 0:
        sys.exit(f"adb failed: {' '.join(args[:3])} ...\n{r.stderr if capture else ''}")
    return r.stdout if capture else r


def remote_text(path):
    r = subprocess.run(["adb", "shell", f"cat {path} 2>/dev/null"], capture_output=True, text=True)
    return r.stdout


def capture_fps(rgb_txt):
    """Frame rate of a frame list from its timestamps (median spacing)."""
    t = [float(l.split()[0]) for l in open(rgb_txt) if l.strip() and not l.startswith("#")]
    gaps = sorted(b - a for a, b in zip(t, t[1:]) if b > a)
    if not gaps:
        sys.exit(f"{rgb_txt}: cannot tell the frame rate from its timestamps; use --fast")
    return 1.0 / gaps[len(gaps) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True); ap.add_argument("--camera", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--no-loop", action="store_true"); ap.add_argument("--first", type=int); ap.add_argument("--last", type=int)
    ap.add_argument("--purpose", default="tracking on the headset")
    ap.add_argument("--fast", action="store_true", help="feed frames as fast as possible instead of at the capture rate")
    a = ap.parse_args()
    frames = Path(a.frames)
    if not (frames / "rgb.txt").is_file():
        sys.exit(f"{frames}/rgb.txt not found")
    fps = 0.0 if a.fast else capture_fps(frames / "rgb.txt")
    if not (RUNTIME / "bin" / "qtrack").is_file():
        sys.exit("tracker runtime not built: run python tools/build_tracking.py first")
    devs = [l for l in adb("devices", capture=True).splitlines()[1:] if l.strip().endswith("device")]
    if len(devs) != 1:
        sys.exit(f"expected exactly one headset over adb, found {len(devs)}")

    build = json.loads((RUNTIME / "build.json").read_text())
    stamp = build["files"]["bin/qtrack"]
    if remote_text(f"{REMOTE}/runtime.sha256").strip() != stamp:
        print("pushing tracker runtime")
        adb("shell", f"mkdir -p {REMOTE}")
        for d in ("bin", "lib", "share"):
            adb("push", str(RUNTIME / d), f"{REMOTE}/")
        adb("shell", f"chmod 755 {REMOTE}/bin/qtrack && echo {stamp} > {REMOTE}/runtime.sha256")
    n_local = sum(1 for l in open(frames / "rgb.txt") if l.strip() and not l.startswith("#"))
    remote_list = remote_text(f"{REMOTE}/frames/{a.name}/rgb.txt")
    n_remote = sum(1 for l in remote_list.splitlines() if l.strip() and not l.startswith("#"))
    if n_remote != n_local:
        print(f"pushing {n_local} frames (this can take a while)")
        adb("shell", f"rm -rf {REMOTE}/frames/{a.name} && mkdir -p {REMOTE}/frames")
        adb("push", str(frames), f"{REMOTE}/frames/{a.name}")
    adb("shell", f"mkdir -p {REMOTE}/configs")
    adb("push", a.camera, f"{REMOTE}/configs/{a.name}.yaml")

    run_id, path = REG.new_run(str(HERE / "runs"), "tracking", {"purpose": a.purpose, "frames": str(frames), "name": a.name,
                                                               "camera": Path(a.camera).name, "runtime_build": build["files"],
                                                               "loop_detector": not a.no_loop,
                                                               "realtime_fps": round(fps, 3)})
    top = adb("shell", "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'", capture=True).strip()
    thermal0 = adb("shell", "dumpsys thermalservice | grep -m3 -E 'Thermal Status|mStatus|Status'", check=False, capture=True).strip()
    out_remote = f"{REMOTE}/out/{run_id}"
    opts = (" --no-loop" if a.no_loop else "") + (f" --first {a.first}" if a.first is not None else "") + \
           (f" --last {a.last}" if a.last is not None else "") + (f" --realtime-fps {fps:.4f}" if fps > 0 else "")
    cmd = (f"cd {REMOTE} && mkdir -p {out_remote} && LD_LIBRARY_PATH={REMOTE}/lib ./bin/qtrack --config configs/{a.name}.yaml "
           f"--vocab share/orb_vocab.fbow --frames frames/{a.name}/rgb.txt --out {out_remote}{opts} > {out_remote}/stdout.txt 2>&1; "
           f"echo exit=$?")
    pace = f"at {fps:.1f} frames a second, as captured" if fps > 0 else "as fast as possible (--fast)"
    print(f"{run_id}: tracking on the headset, {pace}")
    t0 = time.time()
    res = adb("shell", cmd, capture=True).strip()
    wall = time.time() - t0
    thermal1 = adb("shell", "dumpsys thermalservice | grep -m3 -E 'Thermal Status|mStatus|Status'", check=False, capture=True).strip()
    local = Path(path) / "tracking"
    local.mkdir(parents=True, exist_ok=True)
    adb("pull", f"{out_remote}/.", str(local))
    REG.save_json(path, "device.json", {"foreground_app": top, "thermal_before": thermal0, "thermal_after": thermal1,
                                        "adb_wall_s": wall, "result": res, "realtime_fps": fps})
    stats = local / "stats.json"
    if res.endswith("exit=0") and stats.is_file():
        print(stats.read_text())
        print(f"trajectory: {local / 'frame_trajectory.txt'}")
    else:
        print((local / "stdout.txt").read_text() if (local / "stdout.txt").is_file() else res)
        sys.exit("qtrack did not finish; send me runs/<run ID>/tracking/stdout.txt")


if __name__ == "__main__":
    main()
