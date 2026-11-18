#!/usr/bin/env python3
"""Places maps that never saw a board, by the views they share with maps that did (O42, option a).

    python tools/place_maps.py --recordings runs/h121118_recording.json runs/h120450_recording.json [--dry-run]

In the recording summaries (from process_recording.py), maps reconstructed with a board anchor that passed its checks
are "placed": their anchored_trajectory.txt is metric, in a board's frame. Maps skipped because they saw no board are
"unplaced". Every unplaced map is compared with the placed maps of all the given recordings (qrecon/mapalign.py); when
enough of its frames are located and agree on one similarity, it is reconstructed on the headset in that board's frame
(reconstruct_on_quest.py --fixed-anchor) and counts as placed, so a later round can place further maps through it.
Rounds stop when one places nothing. --dry-run locates and fits only, without the headset.

The summaries are updated (a copy of each is kept as .bak) and runs/placement_<time>.json records every attempt.
Then: python tools/merge_clouds.py --recordings ... --out NAME

The laptop does this part (matching and geometry, no neural network), like the merge. Scale drift inside a long map is
not modelled; the similarity is an average over the frames that were located.
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
try:
    import numpy as np
    import cv2
except ModuleNotFoundError as e:
    sys.exit(f"{e.name} is not installed in this Python. Activate the project's environment first:\n"
             "  Windows:       .venv-qrecon\\Scripts\\Activate.ps1\n  Linux / macOS: source .venv-qrecon/bin/activate")
from qrecon import mapalign as MA, questconfig as QC  # noqa: E402


def runs_now():
    d = HERE / "runs"
    return set(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("qrecon_")) if d.is_dir() else set()


def load_recording(path):
    """Everything needed about one recording: its frames, camera and maps."""
    S = json.loads(Path(path).read_text())
    trk = HERE / "runs" / S["tracking_run"]
    man = json.loads((trk / "manifest.json").read_text())
    frames = Path(S.get("frames") or man["frames"])
    if not frames.is_absolute():
        frames = HERE / frames
    head = open(frames / "rgb.txt").readline()
    scale = S.get("frame_scale") or (float(head.split("scale=")[1].split()[0]) if "scale=" in head else 1.0)
    rgb = {}
    for line in open(frames / "rgb.txt"):
        if line.strip() and not line.startswith("#"):
            t, p = line.split()[:2]
            rgb[round(float(t), 4)] = str(frames / p)
    calib = json.loads((HERE / S["calib"]).read_text() if not Path(S["calib"]).is_absolute() else Path(S["calib"]).read_text())
    K = np.array(QC.camera_block(calib, scale)["K"])
    maps = {m["map"]: m for m in json.loads((trk / "tracking" / "maps.json").read_text())}
    return {"path": Path(path), "summary": S, "frames": frames, "scale": scale, "rgb": rgb, "K": K, "maps": maps,
            "fps": S.get("fps") or man.get("realtime_fps"), "tracking_dir": trk / "tracking"}


def frames_of(rec, traj_path, frame=None, mapkey=None, sim3=None):
    """Frames of one map with their poses (optionally carried into a frame by a similarity), images loaded lazily."""
    t, T = MA.read_tum(traj_path)
    if sim3 is not None:
        s, R, tt = sim3["s"], np.array(sim3["R"]), np.array(sim3["t"])
        T = T.copy()
        T[:, :3, 3] = s * T[:, :3, 3] @ R.T + tt
        T[:, :3, :3] = np.einsum("ij,njk->nik", R, T[:, :3, :3])
    out = []
    for ti, Ti in zip(t, T):
        p = rec["rgb"].get(round(float(ti), 4))
        if p:
            out.append({"t": float(ti), "T": Ti, "image": (lambda p=p: cv2.imread(p, cv2.IMREAD_GRAYSCALE)),
                        "frame": frame, "map": mapkey})
    return out


def anchor_status(recon_run):
    a = json.loads((HERE / "runs" / recon_run / "recon" / "anchor.json").read_text())
    F = a.get("frame")
    ok = "fixed" in a or bool(a.get(F, {}).get("ok"))
    return F, ok


def reconstruct(rec, m, fixed_path, args):
    S = rec["summary"]
    before = runs_now()
    cmd = [sys.executable, str(HERE / "tools" / "reconstruct_on_quest.py"), "--name", S["name"], "--frames", str(rec["frames"]),
           "--calib", S["calib"], "--frame-scale", str(rec["scale"]), "--fps", str(rec["fps"]), "--model", args.model,
           "--tracking-run", S["tracking_run"], "--map", str(m), "--fixed-anchor", str(fixed_path),
           "--max-keyframes", str(args.max_keyframes)]
    print("$ " + " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        return None
    made = sorted(runs_now() - before)
    return made[-1] if made else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true", help="locate and fit only; no reconstruction, summaries unchanged")
    ap.add_argument("--include-failed-anchors", action="store_true",
                    help="also locate against maps whose board anchor failed its check, if at least half of their "
                         "keyframes were usable")
    ap.add_argument("--max-center-m", type=float, default=0.15, help="located frames must agree within this distance")
    ap.add_argument("--min-frames", type=int, default=4, help="located frames needed to place a map")
    ap.add_argument("--min-map-s", type=float, default=5.0)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--model", default=str(Path("models") / "dav2s_960x540.onnx"))
    ap.add_argument("--max-keyframes", type=int, default=60)
    a = ap.parse_args()
    recs = [load_recording(p) for p in a.recordings]
    placed, unplaced, report = [], [], {"recordings": a.recordings, "rounds": [], "dry_run": a.dry_run}
    for r in recs:
        for e in r["summary"]["maps"]:
            m = e["map"]
            if e.get("recon_run"):
                F, ok = anchor_status(e["recon_run"])
                usable = (e.get("valid_keyframe_share") or 0) >= 0.5
                if F and F != "tracker" and (ok or (a.include_failed_anchors and usable)):
                    placed.append((r, m, F, HERE / "runs" / e["recon_run"] / "recon" / "anchored_trajectory.txt", None))
            elif e.get("last_t", 0) - e.get("first_t", 0) >= a.min_map_s:
                unplaced.append((r, m))
    print(f"{len(placed)} placed map(s) to locate against, {len(unplaced)} unplaced map(s) to place")
    pdb_cache = {}
    for rnd in range(1, a.max_rounds + 1):
        pdb = []
        for r, m, F, traj, sim3 in placed:
            key = f"{r['summary']['name']} map {m}"
            if key not in pdb_cache:
                src = r["tracking_dir"] / (r["maps"][m].get("dir") or f"maps/map_{m:02d}") / "frame_trajectory.txt" if sim3 else traj
                pdb_cache[key] = MA.p_database(frames_of(r, src, F, key, sim3), r["K"], step_s=1.0)
                print(f"  {key}: {len(pdb_cache[key])} frames with 3-D points, in {F}'s frame")
            pdb += pdb_cache[key]
        if not pdb or not unplaced:
            break
        vocab = MA.Vocab([p["d"] for p in pdb])
        done, attempts = [], []
        for r, m in unplaced:
            key = f"{r['summary']['name']} map {m}"
            mdir = r["maps"][m].get("dir") or f"maps/map_{m:02d}"
            uf = frames_of(r, r["tracking_dir"] / mdir / "frame_trajectory.txt")
            t0 = time.time()
            res, rep = MA.place(uf, pdb, r["K"], vocab=vocab, step_s=0.5, max_center_m=a.max_center_m, min_frames=a.min_frames)
            entry = {"map": key, "sampled": rep["sampled"], "located": len(rep["located"]), "seconds": round(time.time() - t0, 1)}
            if res is None:
                entry["placed"] = False
                entry["why"] = rep.get("reason")
                print(f"  {key}: not placed: {entry['why']}")
            else:
                sim3 = {"s": res["s"], "R": res["R"].tolist(), "t": res["t"].tolist()}
                entry.update(placed=True, frame=res["frame"], sim3=sim3, inliers=len(res["inliers"]),
                             center_rms_m=round(res["center_rms_m"], 3), rotation_rms_deg=round(res["rotation_rms_deg"], 2),
                             baseline_m=round(res["baseline_m"], 2), with_maps=res["with_maps"])
                print(f"  {key}: placed in {res['frame']}'s frame by {len(res['inliers'])} of {len(rep['located'])} located "
                      f"frames (centres {100 * res['center_rms_m']:.1f} cm rms, {res['rotation_rms_deg']:.1f} deg rms, "
                      f"scale {res['s']:.3f}), shared with {', '.join(res['with_maps'])}")
                done.append((r, m, res["frame"], sim3, entry))
            attempts.append(entry)
        report["rounds"].append(attempts)
        if not done:
            break
        for r, m, F, sim3, entry in done:
            unplaced.remove((r, m))
            placed.append((r, m, F, None, sim3))
            if a.dry_run:
                continue
            fdir = HERE / "runs" / "placements"
            fdir.mkdir(parents=True, exist_ok=True)
            fpath = fdir / f"{r['summary']['name']}_map_{m:02d}.json"
            fpath.write_text(json.dumps({"frame": F, "sim3": sim3, "by": f"shared views with {', '.join(entry['with_maps'])}"}, indent=1))
            rid = reconstruct(r, m, fpath, a)
            entry["recon_run"] = rid
            for e in r["summary"]["maps"]:
                if e["map"] == m:
                    e.pop("skipped", None)
                    e["placed_by"] = {k: entry[k] for k in ("frame", "inliers", "center_rms_m", "rotation_rms_deg", "baseline_m",
                                                            "with_maps", "sim3")}
                    if rid:
                        st = HERE / "runs" / rid / "recon" / "stats.json"
                        stats = json.loads(st.read_text()) if st.is_file() else {}
                        e.update(recon_run=rid, frame=F, keyframes=stats.get("keyframes"),
                                 valid_keyframe_share=stats.get("valid_keyframe_share"))
                        r["summary"].setdefault("recon_runs", []).append(rid)
                    else:
                        e["placed_by"]["reconstruction"] = "failed"
    if not a.dry_run:
        for r in recs:
            shutil.copy2(r["path"], str(r["path"]) + ".bak")
            r["path"].write_text(json.dumps(r["summary"], indent=1))
    out = HERE / "runs" / f"placement_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    n = sum(1 for rnd in report["rounds"] for e in rnd if e.get("placed"))
    print(f"\n{n} map(s) placed{' (dry run: nothing reconstructed)' if a.dry_run else ''}; {len(unplaced)} left unplaced -> {out}")


if __name__ == "__main__":
    main()
