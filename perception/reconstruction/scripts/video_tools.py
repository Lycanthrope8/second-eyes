#!/usr/bin/env python3
"""Tools for replayed video (a phone clip now, the CM4 camera later).

  calibrate --video FILE --cols 9 --rows 6 --square-m 0.025 --out calib.json
        Intrinsics from a checkerboard video recorded in the SAME video mode, focus locked.
  tags --video FILE --calib calib.json --tag-id 0 --out tags.json
        Corner pixels of the room tag in every frame where it is found.
  anchor --trajectory traj.txt --tags tags.json --calib calib.json --tag-size-m 0.16 --out anchored.txt
        Scale and room frame for a tracker trajectory (TUM format), from the tag seen in translated views.
  frames            frames and a timestamp list for the tracker
  charuco-inspect   which ArUco dictionary and marker IDs the boards in a video use
  charuco-calibrate intrinsics from the board views inside a recording
  charuco-anchor    metric scale and frame from board 01; link and scale drift from board 02
  stella-config     stella_vslam camera file from a calibration
"""
import argparse, json, os, sys
import numpy as np
import cv2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from qrecon import anchor as A, config as C  # noqa: E402
from qrecon.sources import VideoFileSource, read_trajectory_tum  # noqa: E402


def cmd_calibrate(a):
    cap = cv2.VideoCapture(a.video)
    pattern = (a.cols, a.rows)
    obj = np.zeros((a.cols * a.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square_m
    objp, imgp, size, i = [], [], None, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % a.every == 0:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(gray, pattern)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
                objp.append(obj); imgp.append(corners)
        i += 1
    if len(objp) < a.min_views:
        sys.exit(f"only {len(objp)} usable views of the board; need {a.min_views}. Move the board through the whole image.")
    rms, K, dist, _, _ = cv2.calibrateCamera(objp, imgp, size, None, None)
    out = {"K": K.tolist(), "dist": dist.ravel().tolist(), "size": list(size), "rms_px": float(rms),
           "views": len(objp), "video": os.path.basename(a.video), "board": [a.cols, a.rows, a.square_m]}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{len(objp)} views, reprojection RMS {rms:.3f} px, size {size} -> {a.out}")


def cmd_tags(a):
    calib = json.load(open(a.calib))
    src = VideoFileSource(a.video, calib, every=a.every)
    found = []
    for f in src.frames():
        c = A.detect_tag(f.bgr, a.tag_id)
        if c is not None:
            found.append({"index": f.index, "t": f.t, "corners": c.tolist()})
    json.dump({"tag_id": a.tag_id, "video": os.path.basename(a.video), "observations": found}, open(a.out, "w"), indent=2)
    print(f"tag {a.tag_id} found in {len(found)} frames -> {a.out}")


def cmd_anchor(a):
    from scipy.spatial.transform import Rotation
    cfg = C.load(os.path.join(REPO, "configs", "qrecon.yaml"))["anchor"]
    calib = json.load(open(a.calib)); K = np.array(calib["K"])
    times, poses = read_trajectory_tum(a.trajectory)
    times = np.array(times)
    obs, pose_at = {}, {}
    for k, o in enumerate(json.load(open(a.tags))["observations"]):
        j = int(np.argmin(np.abs(times - o["t"])))
        if abs(times[j] - o["t"]) <= a.max_dt:
            obs[k], pose_at[k] = np.array(o["corners"]), poses[j]
    out = A.anchor_from_tag(obs, pose_at, K, a.tag_size_m, cfg)
    report = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    json.dump(report, open(a.out + ".json", "w"), indent=2)
    if not out["ok"]:
        sys.exit(f"anchor refused: {out['reason']}")
    with open(a.out, "w") as f:
        for t, T in zip(times, A.to_anchor_frame(poses, out["scale"], out["R"], out["t"])):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{t:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")
    print(f"scale {out['scale']:.4f}, corner fit {out['corner_fit_rms_m']*1000:.1f} mm, {out['views']} views -> {a.out}")

def _segments(video, start_s, end_s, every):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * start_s)
    i = int(round(start_s * fps))
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = i / fps
        if end_s is not None and t > end_s:
            break
        if i % every == 0:
            yield i, t, bgr
        i += 1
    cap.release()


def _windows_of(recording):
    ref = C.load(os.path.join(REPO, "configs", "references.yaml"))
    win = ref["recordings"][recording]["windows_s"]
    out = []
    for v in win.values():
        out += v if isinstance(v[0], list) else [v]
    return sorted(out)


def cmd_frames(a):
    """Frames and a timestamp list (TUM rgb.txt layout: 'time relative_path'). With --recording, only the frames
    inside that recording's board windows, at full resolution unless --scale says otherwise."""
    os.makedirs(os.path.join(a.out, "rgb"), exist_ok=True)
    lines, size = [], None
    spans = _windows_of(a.recording) if a.recording else [[a.start, a.end]]
    def frames_in_spans():
        for s0, s1 in spans:
            yield from _segments(a.video, s0, s1, a.every)
    for i, t, bgr in frames_in_spans():
        if a.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        size = (bgr.shape[1], bgr.shape[0])
        name = f"rgb/{i:06d}.jpg"
        cv2.imwrite(os.path.join(a.out, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        lines.append(f"{t:.6f} {name}")
    with open(os.path.join(a.out, "rgb.txt"), "w") as f:
        f.write(f"# {os.path.basename(a.video)} every={a.every} scale={a.scale} size={size}\n" + "\n".join(lines) + "\n")
    print(f"{len(lines)} frames at {size} -> {a.out}")


def cmd_charuco_inspect(a):
    """Which ArUco dictionary and marker IDs the boards use, so references.yaml can be filled in."""
    names = [n for n in dir(cv2.aruco) if n.startswith("DICT_") and not n.endswith("_ORIGINAL")]
    hits = {}
    for i, t, bgr in _segments(a.video, a.start, a.end, a.every):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for n in names:
            det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, n)))
            _, ids, _ = det.detectMarkers(gray)
            if ids is not None and len(ids) >= 3:
                h = hits.setdefault(n, {"frames": 0, "ids": set()})
                h["frames"] += 1; h["ids"].update(int(x) for x in ids.ravel())
    best = sorted(hits.items(), key=lambda kv: -kv[1]["frames"])
    for n, h in best[:5]:
        ids = sorted(h["ids"])
        print(f"{n:24s} frames {h['frames']:4d}  marker IDs {ids[0]}..{ids[-1]} ({len(ids)} distinct)")
    if not best:
        print("no markers found in this time range")


def _boards():
    from qrecon import charuco as CH
    ref = C.load(os.path.join(REPO, "configs", "references.yaml"))
    for k, b in ref["boards"].items():
        if not b["squares_x"] or not b["square_length_m"] or str(b["dictionary"]).startswith("CHECK"):
            sys.exit(f"configs/references.yaml: board {k} is not filled in (dictionary, squares, measured lengths)")
    return CH, ref


def cmd_charuco_calibrate(a):
    CH, ref = _boards()
    board = CH.make_board(ref["boards"][a.board])
    dets, size = [], None
    for i, t, bgr in _segments(a.video, a.start, a.end, a.every):
        size = (bgr.shape[1], bgr.shape[0])
        d = CH.detect(bgr, board, ref["calibration"]["min_corners"])
        if d is not None:
            dets.append(d)
    out = CH.calibrate(dets, board, size, ref["calibration"])
    out.update(video=os.path.basename(a.video), board=a.board, window_s=[a.start, a.end])
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out.get(k) for k in ("ok", "reason", "warning", "rms_px", "views", "image_coverage", "board_distance_m")}, indent=2))


def cmd_charuco_anchor(a):
    """Anchor a tracker trajectory with board 01; then board 02 (scale drift, room link) and board 01 seen again at the
    end (closure). Boards are detected on the original video frames with the original calibration: the poses don't
    depend on the resolution the tracker used."""
    from scipy.spatial.transform import Rotation
    CH, ref = _boards()
    calib = json.load(open(a.calib)); K = np.array(calib["K"]); dist = np.array(calib["dist"])
    win = ref["recordings"][a.recording]["windows_s"] if a.recording else {}
    w01 = [[a.ref01_start, a.ref01_end]] if a.ref01_end is not None else [win["ref01_start"]]
    w01_end = win.get("ref01_end")
    w02 = [[a.ref02_start, a.ref02_end]] if a.ref02_end is not None else win.get("ref02", [])
    times, poses = read_trajectory_tum(a.trajectory)
    times = np.array(times)

    def collect(board, windows):
        obs, idx = {}, {}
        for start, end in windows:
            for i, t, bgr in _segments(a.video, start, end, a.every):
                d = CH.detect(bgr, board, ref["anchor"]["min_corners"])
                j = int(np.argmin(np.abs(times - t)))
                if d is not None and abs(times[j] - t) <= a.max_dt:
                    obs[i], idx[i] = d, j
        return obs, idx

    def clean(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    b1 = CH.make_board(ref["boards"]["ref01"])
    obs1, idx1 = collect(b1, w01)
    out = CH.anchor(obs1, {k: poses[j] for k, j in idx1.items()}, K, dist, b1, ref["anchor"])
    report = {"windows": {"ref01": w01, "ref01_end": w01_end, "ref02": w02}, "ref01": clean(out)}
    if not out["ok"]:
        json.dump(report, open(a.out + ".json", "w"), indent=2)
        sys.exit(f"anchor refused: {out['reason']}")
    anchored = []
    for T in poses:
        Tm = T.copy(); Tm[:3, :3] = out["R"] @ T[:3, :3]; Tm[:3, 3] = out["scale"] * out["R"] @ T[:3, 3] + out["t"]
        anchored.append(Tm)
    if w02:
        b2 = CH.make_board(ref["boards"]["ref02"])
        obs2, idx2 = collect(b2, w02)
        report["ref02"] = clean(CH.second_board(obs2, {k: anchored[j] for k, j in idx2.items()}, K, dist, b2, ref["anchor"]))
        report["ref02"]["note"] = "assumes board 02 printed at the same scale as board 01 (not measured)"
    if w01_end:
        obs_e, idx_e = collect(b1, [w01_end])
        c = CH.second_board(obs_e, {k: anchored[j] for k, j in idx_e.items()}, K, dist, b1, ref["anchor"])
        if c.get("ok"):
            T = np.array(c["board02_in_board01"])
            c = {"ok": True, "scale_drift": c["scale_drift"], "position_error_m": float(np.linalg.norm(T[:3, 3])),
                 "rotation_error_deg": float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))),
                 "triangulated_corners": c["triangulated_corners"]}
        report["closure_ref01"] = clean(c)
    json.dump(report, open(a.out + ".json", "w"), indent=2)
    with open(a.out, "w") as f:
        for t, T in zip(times, anchored):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{t:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")
    msg = f"scale {out['scale']:.4f}, corner fit {out['corner_fit_rms_m']*1000:.1f} mm"
    if report.get("ref02", {}).get("ok"):
        msg += f"; board 02 scale drift {100 * report['ref02']['scale_drift']:+.1f}%"
    if report.get("closure_ref01", {}).get("ok"):
        msg += f"; closure at board 01: {100 * report['closure_ref01']['position_error_m']:.1f} cm"
    print(msg + f" -> {a.out}")


def cmd_stella_config(a):
    """stella_vslam camera file from a calibration, for frames resized by --scale."""
    from qrecon.questconfig import stella_camera_yaml
    calib = json.load(open(a.calib))
    open(a.out, "w").write(stella_camera_yaml(calib, a.scale, a.fps, a.name))
    w, h = calib["size"]
    print(f"stella camera file for {int(round(w*a.scale))}x{int(round(h*a.scale))} -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate"); p.add_argument("--video", required=True); p.add_argument("--cols", type=int, required=True)
    p.add_argument("--rows", type=int, required=True); p.add_argument("--square-m", type=float, required=True)
    p.add_argument("--every", type=int, default=10); p.add_argument("--min-views", type=int, default=15)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("tags"); p.add_argument("--video", required=True); p.add_argument("--calib", required=True)
    p.add_argument("--tag-id", type=int, required=True); p.add_argument("--every", type=int, default=1)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_tags)
    p = sub.add_parser("anchor"); p.add_argument("--trajectory", required=True); p.add_argument("--tags", required=True)
    p.add_argument("--calib", required=True); p.add_argument("--tag-size-m", type=float, required=True)
    p.add_argument("--max-dt", type=float, default=0.05); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_anchor)
    p = sub.add_parser("frames"); p.add_argument("--video", required=True); p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=1); p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--start", type=float, default=0.0); p.add_argument("--end", type=float); p.add_argument("--quality", type=int, default=95)
    p.add_argument("--recording", help="only the board windows of this recording (configs/references.yaml)")
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("charuco-inspect"); p.add_argument("--video", required=True); p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=20.0); p.add_argument("--every", type=int, default=15); p.set_defaults(fn=cmd_charuco_inspect)
    p = sub.add_parser("charuco-calibrate"); p.add_argument("--video", required=True); p.add_argument("--board", default="ref01")
    p.add_argument("--start", type=float, default=0.0); p.add_argument("--end", type=float, required=True)
    p.add_argument("--every", type=int, default=3); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_charuco_calibrate)
    p = sub.add_parser("charuco-anchor"); p.add_argument("--video", required=True); p.add_argument("--trajectory", required=True)
    p.add_argument("--calib", required=True); p.add_argument("--recording", help="name under recordings: in references.yaml")
    p.add_argument("--ref01-start", type=float, default=0.0); p.add_argument("--ref01-end", type=float)
    p.add_argument("--ref02-start", type=float); p.add_argument("--ref02-end", type=float)
    p.add_argument("--every", type=int, default=3); p.add_argument("--max-dt", type=float, default=0.05)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_charuco_anchor)
    p = sub.add_parser("stella-config"); p.add_argument("--calib", required=True); p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=30.0); p.add_argument("--name", default="replayed video"); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_stella_config)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
