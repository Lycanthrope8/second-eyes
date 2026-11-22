#!/usr/bin/env python3
"""Joins the maps of one video into one map (D142) and reconstructs it on the headset in one run.

    python tools/one_map.py --recording runs/h121118_recording.json [--helpers runs/h191353_recording.json ...]
                            [--use-link qrecon_20261117_r007] [--target ref01] [--max-keyframes 200] [--dry-run]

The tracker breaks wherever the images give it nothing to follow, so a video's tracking comes in pieces (map restarts,
D136). This joins the pieces. Pieces anchored by a board whose checks passed are the starting points. Every other piece,
including one whose board anchor failed its checks, is placed by the views it shares with pieces already placed, in this
video or, with --helpers, in other processed recordings, with the same agreement and speed checks as place_maps.py. A
piece that shared views cannot place but that has a board anchor of its own, though a check failed, is joined by that
anchor as a last resort and flagged; it never serves to place other pieces.
Pieces in another board's frame are carried into the target board's frame by a board link (--use-link names the runs
whose links to use, D140). The joined camera path, one trajectory in one metric frame, is then reconstructed on the
headset in a single run, giving one cloud for the video. Pieces that cannot be joined are listed with the reason;
nothing is split silently. Before the reconstruction every piece is refined for scale drift (qrecon/refine.py): each
stretch of it gets its own similarity, from the fits of the board windows inside it (an anchor-only run per window), the
frames located by shared views (for pieces anchored by a board, against the other recordings) and continuity between
stretches; --no-refine keeps one similarity per piece. The recording summary records the join and its run, which
merge_clouds.py and score_cloud.py then use (a copy of the summary is kept as .bak).
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
try:
    import numpy as np
    import cv2
except ModuleNotFoundError as e:
    sys.exit(f"{e.name} is not installed in this Python. Activate the project's environment first:\n"
             "  Windows:       .venv-qrecon\\Scripts\\Activate.ps1\n  Linux / macOS: source .venv-qrecon/bin/activate")
from qrecon import mapalign as MA, refine as RF  # noqa: E402
import place_maps as PM  # noqa: E402
import merge_clouds as MC  # noqa: E402


def apply_sim3(poses, s, R, t):
    out = poses.copy()
    out[:, :3, 3] = s * poses[:, :3, 3] @ np.asarray(R).T + np.asarray(t)
    out[:, :3, :3] = np.einsum("ij,njk->nik", np.asarray(R), poses[:, :3, :3])
    return out


def in_span(times, span):
    return np.ones(len(times), bool) if not span else (times >= span[0] - 1e-6) & (times <= span[1] + 1e-6)


def anchor_info(run):
    p = HERE / "runs" / run / "recon" / "anchor.json"
    a = json.loads(p.read_text()) if p.is_file() else {}
    F = a.get("frame")
    return a, F, ("fixed" in a) or bool(a.get(F, {}).get("ok")) if F not in (None, "tracker") else False


def pieces_of(rec, own):
    """Metric pieces a recording already has: [{key, frame, ok, how, times, poses, run, map}]. For the recording being
    joined (own=True) only its board anchors count, everything else is placed again; for a helper, everything metric."""
    S, out = rec["summary"], []
    om = S.get("one_map")
    if not own and om and om.get("recon_run"):
        a, F, ok = anchor_info(om["recon_run"])
        t, T = MA.read_tum(HERE / "runs" / om["recon_run"] / "recon" / "anchored_trajectory.txt")
        return [{"key": f"{S['name']} one map", "frame": F, "ok": True, "how": "one map", "times": t, "poses": T,
                 "run": om["recon_run"], "map": None}]
    for e in S.get("maps", []):
        run = e.get("anchor_run") or e.get("recon_run")
        if not run or (own and e.get("placed_by")):
            continue
        a, F, ok = anchor_info(run)
        p = HERE / "runs" / run / "recon" / "anchored_trajectory.txt"
        if F in (None, "tracker") or not p.is_file():
            continue
        t, T = MA.read_tum(p)
        keep = in_span(t, e.get("span_s") if e.get("anchor_run") and e.get("kept") == "trimmed" else None)
        out.append({"key": f"{S['name']} map {e['map']}", "frame": F, "ok": ok, "how": "placed" if e.get("placed_by") else "board",
                    "times": t[keep], "poses": T[keep], "run": run, "map": e["map"], "warnings": a.get("warnings", [])})
    return out


def frames_from(rec, times, poses, frame, key):
    out = []
    for ti, Ti in zip(times, poses):
        p = rec["rgb"].get(round(float(ti), 4))
        if p:
            out.append({"t": float(ti), "T": Ti, "image": (lambda p=p: cv2.imread(p, cv2.IMREAD_GRAYSCALE)), "frame": frame, "map": key})
    return out


def join(rec, helpers, args):
    """Places the recording's pieces. Returns (joined pieces, left-out list, attempts)."""
    S = rec["summary"]
    own = pieces_of(rec, own=True)
    joined = [p for p in own if p["ok"]]
    failed = {p["map"]: p for p in own if not p["ok"]}
    refs = joined + [p for h in helpers for p in pieces_of(h, own=False) if p["ok"]]
    span = {e["map"]: (e.get("first_t", 0), e.get("last_t", 0)) for e in S.get("maps", [])}
    cands = [m for m, (a, b) in span.items() if m not in {p["map"] for p in joined} and b - a >= args.min_map_s
             and rec["maps"].get(m, {}).get("dir")]
    attempts, why, pdb_cache = [], {}, {}
    for rnd in range(args.max_rounds):
        if not cands or not refs:
            break
        pdb = []
        for p in refs:
            if p["key"] not in pdb_cache:
                src = rec if p["key"].startswith(S["name"] + " ") else next(h for h in helpers if p["key"].startswith(h["summary"]["name"] + " "))
                pdb_cache[p["key"]] = MA.p_database(frames_from(src, p["times"], p["poses"], p["frame"], p["key"]), src["K"], step_s=1.0)
            pdb += pdb_cache[p["key"]]
        if not pdb:
            break
        vocab, placed_now = MA.Vocab([q["d"] for q in pdb]), []
        for m in list(cands):
            traj = PM.map_trajectory(rec, m)
            res, rep = MA.place(PM.frames_of(rec, traj), pdb, rec["K"], vocab=vocab, step_s=0.5,
                                max_center_m=args.max_center_m, min_frames=args.min_frames)
            entry = {"map": m, "round": rnd + 1, "located": len(rep["located"]), "placed": False}
            if res is None:
                why[m] = rep.get("reason")
            else:
                t, T = MA.read_tum(traj)
                kind, kspan, speed = MA.choose_span(t, T[:, :3, 3], res["s"], res["inlier_times"], args.max_speed, args.trim_pad_s)
                if kind == "rejected":
                    why[m] = f"implied camera speed {speed:.1f} m/s: a wrong placement or scale drift"
                else:
                    keep = in_span(t, kspan if kind == "trimmed" else None)
                    piece = {"key": f"{S['name']} map {m}", "frame": res["frame"], "ok": True, "how": "shared views",
                             "located": rep["located"],
                             "times": t[keep], "poses": apply_sim3(T[keep], res["s"], res["R"], res["t"]), "map": m,
                             "with": res["with_maps"], "inliers": len(res["inliers"]), "center_rms_m": round(res["center_rms_m"], 3),
                             "speed_p95_mps": round(speed, 2), "kept": kind, "span_s": [round(x, 3) for x in kspan] if kspan else None}
                    placed_now.append(piece)
                    entry.update(placed=True, frame=res["frame"], with_maps=res["with_maps"], inliers=len(res["inliers"]),
                                 center_rms_m=round(res["center_rms_m"], 3), scale=round(res["s"], 4), kept=kind)
            attempts.append(entry)
            print(f"  {S['name']} map {m}: " + (f"joined in {entry['frame']}'s frame by {entry['inliers']} of {entry['located']} located "
                  f"frames, shared with {', '.join(entry['with_maps'])}" + (", trimmed" if entry.get('kept') == 'trimmed' else "")
                  if entry["placed"] else f"not joined yet: {why[m]}"), flush=True)
        if not placed_now:
            break
        for p in placed_now:
            cands.remove(p["map"])
            joined.append(p)
            refs.append(p)
    helper_refs = [p for h in helpers for p in pieces_of(h, own=False) if p["ok"]]
    if getattr(args, "refine", False) and helper_refs:
        pdb_h = []
        for p in helper_refs:
            if p["key"] not in pdb_cache:
                src = next(h for h in helpers if p["key"].startswith(h["summary"]["name"] + " "))
                pdb_cache[p["key"]] = MA.p_database(frames_from(src, p["times"], p["poses"], p["frame"], p["key"]), src["K"], step_s=1.0)
            pdb_h += pdb_cache[p["key"]]
        if pdb_h:
            vocab_h = MA.Vocab([q["d"] for q in pdb_h])
            for p in joined:
                if p["how"] == "board" and len(p["times"]):
                    _, rep = MA.place(PM.frames_of(rec, PM.map_trajectory(rec, p["map"]), span=(p["times"][0], p["times"][-1])),
                                      pdb_h, rec["K"], vocab=vocab_h, step_s=1.0, max_center_m=args.max_center_m, min_frames=args.min_frames)
                    p["located"] = rep["located"]
    for m in [m for m in cands if m in failed and len(failed[m]["times"])]:
        p = failed[m]                                   # last resort, flagged: its own board anchor although a check failed
        p["how"] = "board, failed its checks"
        p["fallback_reason"] = why.get(m, "no frame matched a placed piece")
        joined.append(p)
        cands.remove(m)
    left = [{"map": m, "seconds": round(span[m][1] - span[m][0], 1),
             "reason": why.get(m, "no frame matched a placed piece") + ("; its board anchor failed its checks" if m in failed else "")}
            for m in cands]
    return joined, left, attempts


def anchor_window(rec, m, board, window, args):
    """An anchor-only run on the headset for one board window of one piece: that window's own fit."""
    S = rec["summary"]
    spans = HERE / "runs" / "one_map" / f"{S['name']}_map{m:02d}_{board}_{window[0]:.1f}.json"
    spans.parent.mkdir(parents=True, exist_ok=True)
    spans.write_text(json.dumps({"windows_s": {board: [list(window)]}}))
    before = PM.runs_now()
    cmd = [sys.executable, str(HERE / "tools" / "reconstruct_on_quest.py"), "--name", S["name"], "--frames", str(rec["frames"]),
           "--board-frames", str(HERE / "data" / f"{S['name']}_autoboards"), "--board-spans", str(spans), "--calib", S["calib"],
           "--frame-scale", str(rec["scale"]), "--fps", str(rec["fps"]), "--model", args.model, "--tracking-run", S["tracking_run"],
           "--map", str(m), "--anchor", "best-effort", "--anchor-only"]
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    made = sorted(PM.runs_now() - before)
    return made[-1] if r.returncode == 0 and made else None


def window_fits(rec, p, tu, Tu, args, cache):
    """Each board window inside a piece, fitted on its own (cached by piece, board and window):
    [{board, window, run, ok, sim3 from tracker units into the board's frame}]."""
    out = []
    lo, hi = float(tu[0]), float(tu[-1])
    for board, ws in (rec["summary"].get("board_windows_s") or {}).items():
        for w in ws:
            if w[1] < lo or w[0] > hi:
                continue
            key = f"{p['map']}:{board}:{w[0]:.2f}-{w[1]:.2f}"
            run = cache.get(key)
            if not run or not (HERE / "runs" / run / "recon" / "anchor.json").is_file():
                if args.dry_run:
                    continue
                run = anchor_window(rec, p["map"], board, w, args)
                if not run:
                    continue
                cache[key] = run
            _, F, ok = anchor_info(run)
            path = HERE / "runs" / run / "recon" / "anchored_trajectory.txt"
            entry = {"board": board, "window": [float(w[0]), float(w[1])], "run": run, "ok": bool(ok and F == board)}
            if entry["ok"] and path.is_file():
                ta, Ta = MA.read_tum(path)
                i = np.searchsorted(tu, ta)
                good = (i < len(tu)) & (np.abs(tu[np.clip(i, 0, len(tu) - 1)] - ta) < 1e-3)
                if good.sum() >= 3:
                    entry["sim3"] = RF.umeyama(Tu[i[good], :3, 3], Ta[good, :3, 3])
            if "sim3" not in entry:
                entry["ok"] = False
            out.append(entry)
    return out


def refine_pieces(rec, joined, transforms, target, args, cache):
    """Drift-aware refinement (qrecon/refine.py) of every joined piece, in the target frame."""
    for p in joined:
        if p.get("map") is None or p["frame"] not in transforms or not len(p["times"]):
            continue
        traj = PM.map_trajectory(rec, p["map"])
        if not Path(traj).is_file():
            continue
        tu, Tu = MA.read_tum(traj)
        keep = in_span(tu, (p["times"][0], p["times"][-1]))
        tu, Tu = tu[keep], Tu[keep]
        if len(tu) < 10:
            continue
        cur = apply_sim3(p["poses"], *transforms[p["frame"]])
        i = np.searchsorted(tu, p["times"])
        good = (i < len(tu)) & (np.abs(tu[np.clip(i, 0, len(tu) - 1)] - p["times"]) < 1e-3)
        sim0 = RF.umeyama(Tu[i[good], :3, 3], cur[good, :3, 3])
        wins, used = [], []
        for wf in window_fits(rec, p, tu, Tu, args, cache):
            used.append({k: wf[k] for k in ("board", "window", "run", "ok")})
            if not wf["ok"] or wf["board"] not in transforms:
                continue
            sb, Rb, tb = transforms[wf["board"]]
            sw, Rw, tw = wf["sim3"]
            ix = np.flatnonzero((tu >= wf["window"][0]) & (tu <= wf["window"][1]))
            wins.append((ix[::max(1, len(ix) // 12)], (sb * sw, Rb @ Rw, sb * Rb @ tw + tb)))
        locs = []
        for L in p.get("located", []):
            if L["frame"] not in transforms:
                continue
            j = int(np.argmin(np.abs(tu - L["t"])))
            if abs(tu[j] - L["t"]) < 1e-3:
                locs.append((j, apply_sim3(np.asarray(L["T_p"])[None], *transforms[L["frame"]])[0]))
        W, info = RF.refine_piece(tu, Tu, sim0, wins, locs, max_len_s=args.refine_stretch_s)
        before = RF.apply(RF._params(*sim0), Tu)[:, :3, 3]
        moved = np.linalg.norm(W[:, :3, 3] - before, axis=1)
        speed = MA.implied_speed(tu, W[:, :3, 3], 1.0)
        info.update(board_windows=used, speed_p95_mps=round(speed, 2), moved_median_m=round(float(np.median(moved)), 3),
                    moved_max_m=round(float(moved.max()), 3))
        if speed > args.max_speed:
            info["kept"] = f"unrefined: the refined path implies {speed:.1f} m/s"
        else:
            p["times"], p["poses"], p["frame"] = tu, W, target
        p["refine"] = info
        print(f"  map {p['map']} refined: {info['stretches']} stretch(es), {info['windows']} board window(s), {info['located']} located "
              f"frame(s); scale along it x{info['scale_change'][0]}-{info['scale_change'][1]}; moved {100 * info['moved_median_m']:.0f} cm "
              f"median, {100 * info['moved_max_m']:.0f} cm at most" + (f"; {info['kept']}" if "kept" in info else ""), flush=True)


def combine(joined, transforms, target):
    """One trajectory in one frame: the frame reaching the most joined seconds, the target when it can."""
    seconds = {}
    for p in joined:
        seconds[p["frame"]] = seconds.get(p["frame"], 0) + (p["times"][-1] - p["times"][0] if len(p["times"]) else 0)
    frame = target if all(p["frame"] in transforms for p in joined) else max(seconds, key=seconds.get)
    tf = transforms if frame == target else {frame: (1.0, np.eye(3), np.zeros(3))}
    times, poses, used, dropped = [], [], [], []
    for p in joined:
        if p["frame"] not in tf:
            dropped.append(p)
            continue
        s, R, t = tf[p["frame"]]
        times.append(p["times"]); poses.append(apply_sim3(p["poses"], s, R, t)); used.append(p)
    t = np.concatenate(times) if times else np.zeros(0)
    T = np.concatenate(poses) if poses else np.zeros((0, 4, 4))
    o = np.argsort(t, kind="stable")
    t, T = t[o], T[o]
    keep = np.r_[True, np.diff(t) > 1e-6] if len(t) else np.zeros(0, bool)
    return frame, t[keep], T[keep], used, dropped


def write_tum(path, times, poses):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(poses[:, :3, :3]).as_quat() if len(poses) else np.zeros((0, 4))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ti, P, qi in zip(times, poses, q):
            f.write(f"{ti:.6f} {P[0, 3]:.6f} {P[1, 3]:.6f} {P[2, 3]:.6f} {qi[0]:.9f} {qi[1]:.9f} {qi[2]:.9f} {qi[3]:.9f}\n")


def reconstruct(rec, traj, fixed, args):
    S = rec["summary"]
    before = PM.runs_now()
    cmd = [sys.executable, str(HERE / "tools" / "reconstruct_on_quest.py"), "--name", S["name"], "--frames", str(rec["frames"]),
           "--calib", S["calib"], "--frame-scale", str(rec["scale"]), "--fps", str(rec["fps"]), "--model", args.model,
           "--tracking-run", S["tracking_run"], "--trajectory", str(traj), "--fixed-anchor", str(fixed),
           "--max-keyframes", str(args.max_keyframes)]
    print("$ " + " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        return None
    made = sorted(PM.runs_now() - before)
    return made[-1] if made else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--helpers", nargs="*", default=[], help="other processed recordings whose metric maps may be matched")
    ap.add_argument("--use-link", nargs="*", default=[], metavar="RUN", help="runs whose board links join the frames (D140)")
    ap.add_argument("--target", default="ref01", help="board frame of the joined map")
    ap.add_argument("--max-keyframes", type=int, default=200, help="keyframes for the one reconstruction")
    ap.add_argument("--model", default=str(Path("models") / "dav2s_960x540.onnx"))
    ap.add_argument("--min-map-s", type=float, default=5.0)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--max-center-m", type=float, default=0.15)
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--max-speed", type=float, default=2.0)
    ap.add_argument("--trim-pad-s", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true", help="join and write the trajectory, no reconstruction")
    ap.add_argument("--no-refine", dest="refine", action="store_false",
                    help="one similarity per piece, without the drift-aware refinement (qrecon/refine.py)")
    ap.add_argument("--refine-stretch-s", type=float, default=15.0, help="longest stretch of a piece with its own similarity")
    a = ap.parse_args(argv)
    rec = PM.load_recording(a.recording)
    helpers = [PM.load_recording(h) for h in a.helpers]
    S = rec["summary"]
    print(f"{S['name']}: joining its maps into one")
    joined, left, attempts = join(rec, helpers, a)
    link_runs = list(a.use_link) or sorted({p["run"] for p in joined if p.get("run")})
    edges = []
    for run in link_runs:
        an, _, _ = anchor_info(run)
        if an:
            edges += MC.run_links(run, an)
    frames = {p["frame"] for p in joined}
    paths = MC.best_paths(edges, a.target) if edges else {}
    transforms = {a.target: (1.0, np.eye(3), np.zeros(3))}
    for F in frames - {a.target}:
        if F in paths:
            transforms[F] = paths[F][1]
    cache = dict((S.get("one_map") or {}).get("window_fits", {}))
    if a.refine:
        print(f"{S['name']}: refining the pieces for drift")
        refine_pieces(rec, joined, transforms, a.target, a, cache)
    frame, t, T, used, dropped = combine(joined, transforms, a.target)
    for p in dropped:
        left.append({"map": p["map"], "seconds": round(float(p["times"][-1] - p["times"][0]), 1),
                     "reason": f"in {p['frame']}'s frame, and no board link reaches {frame}"})
    span = rec["summary"].get("coverage", {}).get("processed", {}).get("span_s") or 0
    joined_s = float(sum(p["times"][-1] - p["times"][0] for p in used if len(p["times"])))
    print(f"{S['name']}: {len(used)} piece(s) joined into one map in {frame}'s frame, {joined_s:.0f} s of the video"
          + (f" ({100 * joined_s / span:.0f}%)" if span else ""))
    for p in used:
        print(f"  map {p['map']}: {p['how']}" + (f" ({', '.join(p.get('warnings', []))})" if p.get("warnings") else ""))
    for e in left:
        print(f"  map {e['map']} ({e['seconds']} s) left out: {e['reason']}")
    out_dir = HERE / "runs" / "one_map"
    traj = out_dir / f"{S['name']}_trajectory.txt"
    write_tum(traj, t, T)
    fixed = out_dir / f"{S['name']}_fixed.json"
    fixed.write_text(json.dumps({"frame": frame, "sim3": {"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]},
                                 "by": f"one map of {len(used)} piece(s)"}, indent=1))
    info = {"frame": frame, "trajectory": str(traj), "joined_s": round(joined_s, 1),
            "share_of_video": round(joined_s / span, 3) if span else None, "links_from": link_runs if len(frames) > 1 else [],
            "refined": bool(a.refine), "window_fits": cache,
            "pieces": [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in p.items() if k not in ("times", "poses", "located")}
                       | {"first_t": round(float(p["times"][0]), 3), "last_t": round(float(p["times"][-1]), 3)} for p in used],
            "left_out": left, "attempts": attempts}
    if a.dry_run:
        print(f"dry run: trajectory {traj}; nothing reconstructed, summary unchanged")
        return info
    rid = reconstruct(rec, traj, fixed, a)
    if not rid:
        sys.exit("the reconstruction failed; the summary is unchanged")
    st = HERE / "runs" / rid / "recon" / "stats.json"
    stats = json.loads(st.read_text()) if st.is_file() else {}
    info.update(recon_run=rid, keyframes=stats.get("keyframes"), valid_keyframe_share=stats.get("valid_keyframe_share"))
    S["one_map"] = info
    S["recon_runs"] = [rid]
    shutil.copy2(rec["path"], str(rec["path"]) + ".bak")
    rec["path"].write_text(json.dumps(S, indent=1, default=str))
    print(f"{S['name']}: one map {rid} -> {rec['path']}")
    return info


if __name__ == "__main__":
    main()
