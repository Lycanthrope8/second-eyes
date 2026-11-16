#!/usr/bin/env python3
"""Quest-only reconstruction: desktop evaluation harness.

  selfcheck                         synthetic room through the whole pipeline (no models, no data needed)
  reference --seq NAME              run A: reference poses + measured depth; also the reference surface
  dense --seq NAME --arm ARM --reference RUN [--poses reference|estimated --trajectory FILE --align MODE]
  evaluate --run RUN --reference RUN
  compare --baseline RUN --challenger RUN
  evaluate-cloud --cloud PLY --reference RUN --seq NAME [--trajectory FILE --align MODE]
                                    score a cloud made on the headset
  freeze                            record hashes of the spec and configs; validation runs need this
"""
import argparse, json, os, sys, tempfile
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from qrecon import config as C, registry as REG, pipeline as P, metrics as M, anchor as A, depth_models as DM  # noqa: E402
from qrecon.sources import TumSource, read_trajectory_tum  # noqa: E402
from qrecon.fusion import read_ply_xyz  # noqa: E402

CFG = os.path.join(REPO, "configs", "qrecon.yaml")
SEQ = os.path.join(REPO, "configs", "sequences.yaml")
SPEC = os.path.join(REPO, "docs", "evaluation-spec.md")
RUNS = os.path.join(REPO, "runs")


def _seq(name):
    s = C.load(SEQ)["sequences"][name]
    root = s["dir"] if os.path.isabs(s["dir"]) else os.path.join(REPO, s["dir"])
    return s, TumSource(root, s)


def _guard(seq_cfg, arm):
    """Validation sequences run only after the freeze. The oracle arm is plumbing, so it is exempt."""
    if seq_cfg["role"] == "validation" and arm != "oracle":
        ok, why = REG.check_frozen(REPO, [CFG, SEQ], SPEC)
        if not ok:
            sys.exit(f"REFUSED: '{seq_cfg['role']}' sequence and the procedure is not frozen ({why}). Run: qrecon.py freeze")


def cmd_reference(a):
    cfg = C.load(CFG); s, src = _seq(a.seq)
    kf = P.choose_keyframes(src, cfg)
    run_id, path = REG.new_run(RUNS, "reference", {"sequence": a.seq, "config_sha256": C.digest(cfg)})
    REG.save_json(path, "summary.json", P.run_reference(src, kf, cfg, path))
    print(f"{run_id}: {len(kf)} keyframes of {len(src)} frames -> {path}")


def _poses_for(a, src, kf, cfg):
    ref = [src.frame(i).T_wc for i in kf]
    if a.poses == "reference":
        return ref, {"poses": "reference"}
    t_est, T_est = read_trajectory_tum(a.trajectory)
    times = np.array(src.times())[kf]
    idx = [int(np.argmin(np.abs(np.array(t_est) - t))) for t in times]
    gap = max(abs(t_est[i] - t) for i, t in zip(idx, times))
    if gap > 0.05:
        sys.exit(f"REFUSED: the trajectory has no pose within 0.05 s of some keyframes (largest gap {gap:.3f} s). "
                 "Tracking may have been lost or the trajectory belongs to another sequence.")
    est = [T_est[i] for i in idx]
    first_k = cfg["anchor"]["diagnostic_first_k"] if a.align == "diagnostic_start" else None
    s, R, t = A.diagnostic_alignment(est, ref, first_k)
    aligned = A.to_anchor_frame(est, s, R, t)
    traj = M.trajectory_errors(np.array([T[:3, 3] for T in aligned]), np.array([T[:3, 3] for T in ref]))
    return aligned, {"poses": "estimated", "trajectory": a.trajectory, "alignment": a.align,
                     "DIAGNOSTIC": "scale and frame were fitted to reference positions; scale error is not measured by this run",
                     "fitted_scale": s, "trajectory_error": traj}


def cmd_dense(a):
    cfg = C.load(CFG); s, src = _seq(a.seq); _guard(s, a.arm)
    ref = REG.load_json(os.path.join(RUNS, a.reference), "summary.json")
    kf = ref["keyframes"]
    poses, pose_info = _poses_for(a, src, kf, cfg)
    provider = DM.make(a.arm, cfg["arms"])
    run_id, path = REG.new_run(RUNS, "dense", {"sequence": a.seq, "arm": a.arm, "reference_run": a.reference,
                                               "config_sha256": C.digest(cfg), "model": provider.info(), **pose_info})
    summary, per_kf = P.run_dense(src, kf, poses, provider, cfg, path)
    summary.update(pose_info)
    REG.save_json(path, "summary.json", summary); REG.save_json(path, "keyframes.json", per_kf)
    print(f"{run_id}: {summary['valid_keyframes']}/{summary['keyframes']} keyframes calibrated -> {path}")
    return run_id


def cmd_evaluate(a):
    cfg = C.load(CFG)
    rp, pp = os.path.join(RUNS, a.reference), os.path.join(RUNS, a.run)
    summary = REG.load_json(pp, "summary.json")
    cloud = M.cloud_metrics(read_ply_xyz(os.path.join(pp, "predicted.ply")), read_ply_xyz(os.path.join(rp, "reference.ply")),
                            cfg["metrics"]["taus_m"], cfg["metrics"]["cloud_voxel_m"])
    verdict = P.judge(summary, cloud, cfg)
    if summary.get("poses") == "estimated":
        verdict["note"] = "diagnostic alignment: the scale check does not count as evidence of metric scale"
    REG.save_json(pp, "evaluation.json", {"summary": summary, "cloud": cloud, "screening": verdict})
    print(json.dumps({"run": a.run, "arm": summary["arm"], "scale_error": summary.get("scale_error"), **cloud, **verdict}, indent=2))
    return cloud, verdict


def cmd_compare(a):
    cfg = C.load(CFG); g = cfg["gates"]
    b = REG.load_json(os.path.join(RUNS, a.baseline), "evaluation.json")
    c = REG.load_json(os.path.join(RUNS, a.challenger), "evaluation.json")
    gain = c["cloud"]["fscore_5cm"] - b["cloud"]["fscore_5cm"]
    tail_ok = c["cloud"]["accuracy_p95_m"] <= b["cloud"]["accuracy_p95_m"] * (1 + g["max_tail_worsening"])
    out = {"baseline": a.baseline, "challenger": a.challenger, "fscore_5cm_gain": gain, "tail_not_worse": bool(tail_ok),
           "challenger_passes_screening": c["screening"]["passes_screening"],
           "challenger_improves": bool(gain >= g["da3_min_fscore5_gain"] and tail_ok)}
    print(json.dumps(out, indent=2))


def cmd_evaluate_cloud(a):
    """A cloud made elsewhere (the headset program) against a reference run. With a tracker trajectory from a
    sequence without a metric anchor, a similarity is fitted to the reference positions first: DIAGNOSTIC."""
    from qrecon.fusion import read_ply_xyz as rp
    cfg = C.load(CFG); s, src = _seq(a.seq)
    ref = REG.load_json(os.path.join(RUNS, a.reference), "summary.json")
    cloud = rp(a.cloud)
    note = "cloud scored as produced"
    if a.align != "none":
        t_est, T_est = read_trajectory_tum(a.trajectory)
        times = np.array(src.times())[ref["keyframes"]]
        idx = [int(np.argmin(np.abs(np.array(t_est) - t))) for t in times]
        refp = [src.frame(i).T_wc for i in ref["keyframes"]]
        first_k = cfg["anchor"]["diagnostic_first_k"] if a.align == "diagnostic_start" else None
        sc, R, t = A.diagnostic_alignment([T_est[i] for i in idx], refp, first_k)
        cloud = (sc * (R @ cloud.T)).T + t
        note = f"DIAGNOSTIC: similarity fitted to reference positions ({a.align}); fitted scale {sc:.4f}; not evidence of metric scale"
    m = M.cloud_metrics(cloud, read_ply_xyz(os.path.join(RUNS, a.reference, "reference.ply")),
                        cfg["metrics"]["taus_m"], cfg["metrics"]["cloud_voxel_m"])
    out = {"cloud": a.cloud, "reference_run": a.reference, "alignment": a.align, "note": note, "cloud_metrics": m}
    with open(os.path.splitext(a.cloud)[0] + "_evaluation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def cmd_freeze(a):
    files = {os.path.relpath(p, REPO).replace("\\", "/"): C.file_sha256(p) for p in (CFG, SEQ, SPEC)}
    out = {"frozen": True, "files": files}
    with open(os.path.join(REPO, "configs", "freeze.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Frozen. Commit configs/freeze.json; any change to the spec or configs now blocks validation runs.")


def cmd_selfcheck(a):
    from qrecon import synthetic
    cfg = C.load(CFG)
    with tempfile.TemporaryDirectory() as tmp:
        s = synthetic.write_tum(os.path.join(tmp, "seq"))
        src = TumSource(s["dir"], s)
        kf = P.choose_keyframes(src, cfg)
        os.makedirs(os.path.join(tmp, "ref")); os.makedirs(os.path.join(tmp, "run"))
        P.run_reference(src, kf, cfg, os.path.join(tmp, "ref"))
        poses = [src.frame(i).T_wc for i in kf]
        summary, _ = P.run_dense(src, kf, poses, DM.make("oracle", cfg["arms"]), cfg, os.path.join(tmp, "run"), log=lambda *_: None)
        cloud = M.cloud_metrics(read_ply_xyz(os.path.join(tmp, "run", "predicted.ply")),
                                read_ply_xyz(os.path.join(tmp, "ref", "reference.ply")),
                                cfg["metrics"]["taus_m"], cfg["metrics"]["cloud_voxel_m"])
    ok = (summary["scale_error"] is not None and summary["scale_error"] < 0.01 and cloud["accuracy_median_m"] < 0.02
          and summary["valid_keyframe_share"] >= 0.8)
    print(json.dumps({"keyframes": summary["keyframes"], "valid": summary["valid_keyframes"],
                      "scale_error": summary["scale_error"], "accuracy_median_m": cloud["accuracy_median_m"],
                      "fscore_5cm": cloud.get("fscore_5cm")}, indent=2))
    print("SELFCHECK PASS" if ok else "SELFCHECK FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck").set_defaults(fn=cmd_selfcheck)
    p = sub.add_parser("reference"); p.add_argument("--seq", required=True); p.set_defaults(fn=cmd_reference)
    p = sub.add_parser("dense"); p.add_argument("--seq", required=True); p.add_argument("--arm", required=True)
    p.add_argument("--reference", required=True); p.add_argument("--poses", choices=["reference", "estimated"], default="reference")
    p.add_argument("--trajectory"); p.add_argument("--align", choices=["diagnostic_full", "diagnostic_start"], default="diagnostic_start")
    p.set_defaults(fn=cmd_dense)
    p = sub.add_parser("evaluate"); p.add_argument("--run", required=True); p.add_argument("--reference", required=True); p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("compare"); p.add_argument("--baseline", required=True); p.add_argument("--challenger", required=True); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("evaluate-cloud"); p.add_argument("--cloud", required=True); p.add_argument("--reference", required=True)
    p.add_argument("--seq", required=True); p.add_argument("--trajectory")
    p.add_argument("--align", choices=["none", "diagnostic_full", "diagnostic_start"], default="none"); p.set_defaults(fn=cmd_evaluate_cloud)
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
