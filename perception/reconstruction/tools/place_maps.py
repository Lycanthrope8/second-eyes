#!/usr/bin/env python3
"""Places maps that never saw a board, by the views they share with maps that did (O42, D139).

    python tools/place_maps.py --recordings runs/h121118_recording.json runs/h120450_recording.json
                               [--include-failed-anchors] [--replace-failed] [--redo] [--dry-run]

References: maps of the given recordings reconstructed with a board anchor that passed its checks (and, with
--include-failed-anchors, maps whose anchor failed but that kept at least half their keyframes), plus maps placed by
shared views earlier. Their anchored trajectories are metric, in a board's frame.
Candidates: maps that saw no board; with --replace-failed also maps whose board anchor failed and that kept under half
their keyframes; with --redo also maps placed by shared views before, so that they are placed again by the current
rules (a map that no longer passes loses its earlier reconstruction from the summary).

Each candidate is compared with the references (qrecon/mapalign.py). When enough of its frames are located and agree
on one similarity, the speed at which that similarity says the camera moved is checked: if it is plausible (under
--max-speed) the whole map is kept; if not, which happens when the tracker's scale drifted inside the map, only the
stretch around the frames that placed it (--trim-pad-s either side) is kept, if that part is plausible; otherwise the
map is not placed. A placed map is reconstructed on the headset in that board's frame (reconstruct_on_quest.py
--fixed-anchor) and becomes a reference, so a later round can place further maps through it. --dry-run locates and
fits only. The summaries are updated (a copy of each is kept as .bak) and runs/placement_<time>.json records every
attempt. Then: python tools/merge_clouds.py --recordings ... --out NAME

The laptop does this part (matching and geometry, no neural network), like the merge.
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
    cpath = Path(S["calib"]) if Path(S["calib"]).is_absolute() else HERE / S["calib"]
    K = np.array(QC.camera_block(json.loads(cpath.read_text()), scale)["K"])
    maps = {m["map"]: m for m in json.loads((trk / "tracking" / "maps.json").read_text())}
    return {"path": Path(path), "summary": S, "frames": frames, "scale": scale, "rgb": rgb, "K": K, "maps": maps,
            "fps": S.get("fps") or man.get("realtime_fps"), "tracking_dir": trk / "tracking"}


def map_trajectory(rec, m):
    return rec["tracking_dir"] / (rec["maps"][m].get("dir") or f"maps/map_{m:02d}") / "frame_trajectory.txt"


def frames_of(rec, traj_path, frame=None, mapkey=None, sim3=None, span=None):
    """Frames of one map with their poses (optionally carried into a frame by a similarity, and limited to a time span),
    images loaded lazily."""
    t, T = MA.read_tum(traj_path)
    if sim3 is not None:
        s, R, tt = sim3["s"], np.array(sim3["R"]), np.array(sim3["t"])
        T = T.copy()
        T[:, :3, 3] = s * T[:, :3, 3] @ R.T + tt
        T[:, :3, :3] = np.einsum("ij,njk->nik", R, T[:, :3, :3])
    out = []
    for ti, Ti in zip(t, T):
        if span and not (span[0] - 1e-6 <= ti <= span[1] + 1e-6):
            continue
        p = rec["rgb"].get(round(float(ti), 4))
        if p:
            out.append({"t": float(ti), "T": Ti, "image": (lambda p=p: cv2.imread(p, cv2.IMREAD_GRAYSCALE)),
                        "frame": frame, "map": mapkey})
    return out


def anchor_status(recon_run):
    """(frame, anchor ok, placed by a fixed similarity)."""
    a = json.loads((HERE / "runs" / recon_run / "recon" / "anchor.json").read_text())
    F = a.get("frame")
    fixed = "fixed" in a
    return F, fixed or bool(a.get(F, {}).get("ok")), fixed


def reconstruct(rec, m, fixed_path, args, trajectory=None):
    S = rec["summary"]
    before = runs_now()
    cmd = [sys.executable, str(HERE / "tools" / "reconstruct_on_quest.py"), "--name", S["name"], "--frames", str(rec["frames"]),
           "--calib", S["calib"], "--frame-scale", str(rec["scale"]), "--fps", str(rec["fps"]), "--model", args.model,
           "--tracking-run", S["tracking_run"], "--map", str(m), "--fixed-anchor", str(fixed_path),
           "--max-keyframes", str(args.max_keyframes)] + (["--trajectory", str(trajectory)] if trajectory else [])
    print("$ " + " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        return None
    made = sorted(runs_now() - before)
    return made[-1] if made else None


def entry_of(rec, m):
    return next(e for e in rec["summary"]["maps"] if e["map"] == m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true", help="locate and fit only; no reconstruction, summaries unchanged")
    ap.add_argument("--include-failed-anchors", action="store_true",
                    help="also use maps whose board anchor failed its check as references, if they kept half their keyframes")
    ap.add_argument("--replace-failed", action="store_true",
                    help="also place maps whose board anchor failed and that kept under half their keyframes")
    ap.add_argument("--redo", action="store_true", help="place again the maps placed by shared views before")
    ap.add_argument("--min-map-s", type=float, default=5.0)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--max-center-m", type=float, default=0.15, help="located frames must agree within this distance")
    ap.add_argument("--min-frames", type=int, default=4, help="located frames needed to place a map")
    ap.add_argument("--max-speed", type=float, default=2.0, help="m/s; a placement implying faster camera motion is trimmed or refused")
    ap.add_argument("--trim-pad-s", type=float, default=10.0, help="seconds kept either side of the frames that placed a map")
    ap.add_argument("--model", default=str(Path("models") / "dav2s_960x540.onnx"))
    ap.add_argument("--max-keyframes", type=int, default=60)
    a = ap.parse_args()
    recs = [load_recording(p) for p in a.recordings]
    refs, cands = [], []           # refs: (rec, map, frame, trajectory or None, sim3 or None, span); cands: (rec, map, old run)
    for r in recs:
        for e in r["summary"]["maps"]:
            m, run = e["map"], e.get("recon_run")
            if run:
                F, ok, fixed = anchor_status(run)
                usable = (e.get("valid_keyframe_share") or 0) >= 0.5
                if fixed and a.redo:
                    cands.append((r, m, run))
                elif fixed and e.get("placed_by", {}).get("sim3"):            # placed by shared views: same poses, no file needed
                    pb = e["placed_by"]
                    refs.append((r, m, F, None, pb["sim3"], pb.get("span_s")))
                elif F and F != "tracker" and (ok or (a.include_failed_anchors and usable)):
                    traj = HERE / "runs" / run / "recon" / "anchored_trajectory.txt"
                    if traj.is_file():
                        refs.append((r, m, F, traj, None, None))
                    else:
                        print(f"  note: {run} has no anchored_trajectory.txt; not used as a reference")
                elif a.replace_failed and not ok and not usable:
                    cands.append((r, m, run))
            elif e.get("last_t", 0) - e.get("first_t", 0) >= a.min_map_s:
                cands.append((r, m, None))
    print(f"{len(refs)} reference map(s), {len(cands)} map(s) to place")
    report = {"recordings": a.recordings, "dry_run": a.dry_run, "options": {k: v for k, v in vars(a).items() if k != "recordings"},
              "rounds": []}
    pdb_cache, last = {}, {}
    for rnd in range(1, a.max_rounds + 1):
        if not cands:
            break
        pdb = []
        for r, m, F, traj, sim3, span in refs:
            key = f"{r['summary']['name']} map {m}"
            if key not in pdb_cache:
                src = map_trajectory(r, m) if sim3 else traj
                pdb_cache[key] = MA.p_database(frames_of(r, src, F, key, sim3, span), r["K"], step_s=1.0)
                print(f"  {key}: {len(pdb_cache[key])} frames with 3-D points, in {F}'s frame")
            pdb += pdb_cache[key]
        if not pdb or not cands:
            break
        vocab = MA.Vocab([p["d"] for p in pdb])
        done, attempts = [], []
        for r, m, old in cands:
            key = f"{r['summary']['name']} map {m}"
            traj = map_trajectory(r, m)
            t0 = time.time()
            res, rep = MA.place(frames_of(r, traj), pdb, r["K"], vocab=vocab, step_s=0.5, max_center_m=a.max_center_m,
                                min_frames=a.min_frames)
            entry = {"map": key, "sampled": rep["sampled"], "located": len(rep["located"]), "seconds": round(time.time() - t0, 1),
                     "placed": False}
            if res is None:
                entry["why"] = rep.get("reason")
            else:
                tt, TT = MA.read_tum(traj)
                kind, span, speed = MA.choose_span(tt, TT[:, :3, 3], res["s"], res["inlier_times"], a.max_speed, a.trim_pad_s)
                sim3 = {"s": res["s"], "R": res["R"].tolist(), "t": res["t"].tolist()}
                entry.update(frame=res["frame"], sim3=sim3, inliers=len(res["inliers"]), center_rms_m=round(res["center_rms_m"], 3),
                             rotation_rms_deg=round(res["rotation_rms_deg"], 2), baseline_m=round(res["baseline_m"], 2),
                             with_maps=res["with_maps"], speed_p95_mps=round(speed, 2), kept=kind,
                             span_s=[round(x, 3) for x in span] if span else None)
                if kind == "rejected":
                    entry["why"] = f"implied camera speed {speed:.1f} m/s (over {a.max_speed}): a wrong placement or scale drift"
                else:
                    entry["placed"] = True
                    done.append((r, m, old, entry))
            if entry["placed"]:
                e = entry
                print(f"  {key}: placed in {e['frame']}'s frame by {e['inliers']} of {e['located']} located frames (centres "
                      f"{100 * e['center_rms_m']:.1f} cm rms, {e['rotation_rms_deg']:.1f} deg rms, scale {e['sim3']['s']:.3f}), "
                      f"shared with {', '.join(e['with_maps'])}; camera speed {e['speed_p95_mps']:.1f} m/s, "
                      + ("whole map kept" if e["kept"] == "full" else f"kept {e['span_s'][0]:.1f}-{e['span_s'][1]:.1f} s only"))
            else:
                print(f"  {key}: not placed: {entry['why']}")
            attempts.append(entry)
            last[(id(r), m)] = entry
        report["rounds"].append(attempts)
        if not done:
            break
        for r, m, old, entry in done:
            cands.remove((r, m, old))
            refs.append((r, m, entry["frame"], None, entry["sim3"], entry["span_s"]))
            if a.dry_run:
                continue
            fdir = HERE / "runs" / "placements"
            fdir.mkdir(parents=True, exist_ok=True)
            name = r["summary"]["name"]
            fpath = fdir / f"{name}_map_{m:02d}.json"
            fpath.write_text(json.dumps({"frame": entry["frame"], "sim3": entry["sim3"],
                                         "by": f"shared views with {', '.join(entry['with_maps'])}"}, indent=1))
            tpath = None
            if entry["kept"] == "trimmed":
                tpath = fdir / f"{name}_map_{m:02d}_trajectory.txt"
                lo, hi = entry["span_s"]
                lines = [l for l in open(map_trajectory(r, m)) if l.strip() and not l.startswith("#")
                         and lo - 1e-6 <= float(l.split()[0]) <= hi + 1e-6]
                tpath.write_text("".join(lines))
            rid = reconstruct(r, m, fpath, a, tpath)
            entry["recon_run"] = rid
            e = entry_of(r, m)
            e.pop("skipped", None)
            e["placed_by"] = {k: entry[k] for k in ("frame", "inliers", "center_rms_m", "rotation_rms_deg", "baseline_m", "with_maps",
                                                    "sim3", "speed_p95_mps", "kept", "span_s")}
            if rid:
                st = HERE / "runs" / rid / "recon" / "stats.json"
                stats = json.loads(st.read_text()) if st.is_file() else {}
                e.update(recon_run=rid, frame=entry["frame"], keyframes=stats.get("keyframes"),
                         valid_keyframe_share=stats.get("valid_keyframe_share"))
                e.pop("warnings", None)
                runs = r["summary"].setdefault("recon_runs", [])
                if old:
                    e["replaced_run"] = old
                    if old in runs:
                        runs.remove(old)
                runs.append(rid)
            else:
                e["placed_by"]["reconstruction"] = "failed"
    left = []
    for r, m, old in cands:
        why = last.get((id(r), m), {}).get("why", "not tried")
        left.append(f"{r['summary']['name']} map {m}")
        if a.dry_run or not old:
            continue
        e = entry_of(r, m)
        if "placed_by" in e and not e.get("replaced_run"):            # placed by shared views before, refused now (--redo)
            runs = r["summary"].get("recon_runs", [])
            if old in runs:
                runs.remove(old)
            for k in ("placed_by", "recon_run", "frame", "keyframes", "valid_keyframe_share"):
                e.pop(k, None)
            e["skipped"] = f"no longer placed: {why}"
    if not a.dry_run:
        for r in recs:
            shutil.copy2(r["path"], str(r["path"]) + ".bak")
            r["path"].write_text(json.dumps(r["summary"], indent=1))
    out = HERE / "runs" / f"placement_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    n = sum(1 for rnd in report["rounds"] for e in rnd if e.get("placed"))
    print(f"\n{n} map(s) placed{' (dry run: nothing reconstructed)' if a.dry_run else ''}; "
          f"{len(left)} left unplaced{': ' + ', '.join(left) if left else ''} -> {out}")


if __name__ == "__main__":
    main()
