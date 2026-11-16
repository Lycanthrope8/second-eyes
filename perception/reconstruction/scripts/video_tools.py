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
  report            a recording's fitness for tracking and anchoring, before any tracker runs
  check-frames      whether extracted frames are the video frames their names say
  focal-from-lines  the focal length from the room's straight lines, to check a board calibration
  charuco-survey    calibrations from every board window of a recording, one by one and together
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

def _spans(video, spans, every):
    """Frames whose time lies inside any of the spans, as (index, time, image), in one pass. The video is decoded in
    order from its first frame, so every index is the frame's true position and time = index / fps, exactly as for
    the tracker's frames. It does not seek: on variable-rate video (phones in dim light) OpenCV's seek can land a
    frame or more away from the index it implies, which pairs board views with the wrong tracker poses."""
    if not os.path.isfile(video):
        sys.exit(f"video not found: {os.path.abspath(video)}\nPut it in this folder or pass its full path with --video.")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"OpenCV cannot open {video}; send me the output of: python -c \"import cv2; print(cv2.getBuildInformation())\"")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    spans = sorted((float(s0), float("inf") if s1 is None else float(s1)) for s0, s1 in spans)
    last = max(s1 for _, s1 in spans)
    i = 0
    while i / fps <= last:
        t = i / fps
        if i % every == 0 and any(s0 <= t <= s1 for s0, s1 in spans):
            ok, bgr = cap.read()
            if not ok:
                break
            yield i, t, bgr
        elif not cap.grab():
            break
        i += 1
    cap.release()


def _segments(video, start_s, end_s, every):
    yield from _spans(video, [(start_s, end_s)], every)


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
    for i, t, bgr in _spans(a.video, spans, a.every):
        if a.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        size = (bgr.shape[1], bgr.shape[0])
        name = f"rgb/{i:06d}.jpg"
        cv2.imwrite(os.path.join(a.out, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        lines.append(f"{t:.6f} {name}")
    if not lines:
        sys.exit("no frames in that time range: check --start and --end against the video's length")
    import datetime
    stamp = datetime.datetime.now().isoformat(timespec="seconds")      # lets the headset tools see a re-extraction
    with open(os.path.join(a.out, "rgb.txt"), "w") as f:
        f.write(f"# {os.path.basename(a.video)} every={a.every} scale={a.scale} size={size} decoded=in-order "
                f"extracted={stamp}\n" + "\n".join(lines) + "\n")
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
    out_print = {k: out.get(k) for k in ("ok", "reason", "warning", "rms_px", "views", "image_coverage", "board_distance_m",
                                          "focal_std_fraction")}
    if out.get("K"):
        out_print["focal_px"] = out["K"][0][0]
        out_print["k1"] = out["dist"][0]
    print(json.dumps(out_print, indent=2))


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
        for i, t, bgr in _spans(a.video, windows, a.every):
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


def cmd_report(a):
    """A recording's fitness for tracking and anchoring, checked before any tracker runs. Every frame on the tracking
    grid (--every, --scale) is measured as the tracker would see it: ORB features (stella's pyramid and FAST
    threshold), how much of the image they cover, matches to the previous frame within stella's 100 px search area
    (ratio-tested, so repeating patterns count as ambiguous), sharpness, brightness and glare. Stretches that are
    likely to stop the tracker are listed, boards are found at full resolution every --board-every frames, and a
    contact sheet shows the content. The limits are screening heuristics, not the tracker's own rules: run them on a
    recording whose tracking result is known to see how well they agree."""
    CH, ref = _boards()
    boards = {k: CH.make_board(b) for k, b in ref["boards"].items()}
    min_corners = ref["anchor"]["min_corners"]
    stem = os.path.splitext(os.path.basename(a.video))[0]
    out = a.out or os.path.join(REPO, "reports", stem)
    os.makedirs(out, exist_ok=True)
    if not os.path.isfile(a.video):
        sys.exit(f"video not found: {os.path.abspath(a.video)}")
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"OpenCV cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else None
    orb = cv2.ORB_create(nfeatures=a.features, scaleFactor=1.2, nlevels=8, fastThreshold=a.fast)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    fast = cv2.FastFeatureDetector_create(a.fast)
    window_px = 100.0 * a.scale / 0.5                      # stella's area-matching radius at 960x540
    rows, hits, thumbs = [], {k: [] for k in boards}, []
    prev, drift, size, i, next_thumb, last_note = None, 0.0, None, 0, 0.0, 0.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = i / fps
        drift = max(drift, abs(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - t))
        size = (bgr.shape[1], bgr.shape[0])
        if i % a.every == 0:
            small = cv2.resize(bgr, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA) if a.scale != 1.0 else bgr
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(g, None)
            h, w = g.shape
            cells = np.zeros((6, 8), int)                  # texture coverage: uncapped FAST corners per cell
            for k in fast.detect(g, None):
                cells[min(int(k.pt[1] * 6 / h), 5), min(int(k.pt[0] * 8 / w), 7)] += 1
            matches = 0
            if prev is not None and des is not None and prev[1] is not None and len(des) >= 2 and len(prev[1]) >= 2:
                for pair in bf.knnMatch(prev[1], des, k=2):
                    if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
                        p0, p1 = prev[0][pair[0].queryIdx].pt, kp[pair[0].trainIdx].pt
                        matches += (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 < window_px ** 2
            rows.append((t, len(kp), float((cells >= 5).mean()), int(matches) if prev is not None else -1,
                         float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean()), float((g >= 250).mean())))
            prev = (kp, des)
        if i % a.board_every == 0:
            for name, board in boards.items():
                d = CH.detect(bgr, board, min_corners)
                if d is not None:
                    hits[name].append((t, len(d[0])))
        if t >= next_thumb:
            thumbs.append((t, cv2.resize(bgr, (320, int(round(320 * bgr.shape[0] / bgr.shape[1]))), interpolation=cv2.INTER_AREA)))
            next_thumb += a.sheet_every
        if t - last_note >= 30.0:
            print(f"  {t:6.1f} s read", flush=True); last_note = t
        i += 1
    cap.release()
    if not rows:
        sys.exit("no frames could be read")
    T = np.array(rows)
    with open(os.path.join(out, "timeline.csv"), "w") as f:
        f.write("t,keypoints,cells_share,matches,sharpness,brightness,glare_share\n")
        for r in rows:
            f.write(f"{r[0]:.4f},{r[1]},{r[2]:.3f},{r[3]},{r[4]:.1f},{r[5]:.1f},{r[6]:.4f}\n")
    dt = a.every / fps
    risky = ((T[:, 3] >= 0) & (T[:, 3] < a.min_matches)) | (T[:, 1] < a.min_keypoints)
    spans, s0 = [], None
    for k in range(len(T) + 1):
        r = k < len(T) and risky[k]
        if r and s0 is None:
            s0 = k
        if not r and s0 is not None:
            if T[k - 1, 0] - T[s0, 0] + dt >= a.min_span_s:
                spans.append([round(float(T[s0, 0]), 2), round(float(T[k - 1, 0] + dt), 2)])
            s0 = None
    need = max(1, int(round(1.0 / dt)))
    good = ~risky
    first_good = next((round(float(T[k, 0]), 2) for k in range(len(T) - need + 1) if good[k:k + need].all()), None)
    # Starting a map needs texture across the image, not just many features in one place: on the house video stella
    # made its first attempt exactly when texture first covered 30% of the image for a second (5.2 s).
    wide = T[:, 2] >= a.min_start_cells
    map_start = next((round(float(T[k, 0]), 2) for k in range(len(T) - need + 1) if wide[k:k + need].all()), None)

    def board_spans(h):
        out_s, cur = [], None
        for t, n in h:
            if cur and t - cur["end_s"] <= 1.5 * a.board_every / fps + 1e-6:
                cur["end_s"] = round(t, 2); cur["samples"] += 1; cur["max_corners"] = max(cur["max_corners"], n)
            else:
                cur = {"start_s": round(t, 2), "end_s": round(t, 2), "samples": 1, "max_corners": n}; out_s.append(cur)
        return out_s
    found = {k: board_spans(h) for k, h in hits.items()}
    duration = (i - 1) / fps
    pad = lambda s: [round(max(0.0, s["start_s"] - 1.0), 1), round(min(duration, s["end_s"] + 1.0), 1)]
    windows, notes = {}, []
    r1, r2 = found.get("ref01", []), found.get("ref02", [])
    if r1:
        windows["calibration"] = pad(r1[0])
        # The anchor needs board 01 while the tracker has a map: views before the map can start give no poses.
        usable = [sp for sp in r1 if map_start is not None and sp["end_s"] >= map_start + 0.5]
        groups = []                                       # the first usable views and any others within 20 s of them
        if usable:
            near = [sp for sp in usable if sp["start_s"] <= usable[0]["start_s"] + 20.0]
            groups.append({"start_s": usable[0]["start_s"], "end_s": near[-1]["end_s"]})
        if groups:
            windows["ref01_start"] = pad(groups[0])
            if r1[0]["end_s"] < (map_start or 0) + 0.5:
                notes.append(f"board 01 at {r1[0]['start_s']}-{r1[0]['end_s']} s comes before the map can start "
                             f"(about {map_start} s), so the anchor window uses its later views")
        else:
            windows["ref01_start"] = pad(r1[0])
            notes.append("board 01 is only seen before the map can start: the anchor will find no tracked views")
        last = r1[-1]
        if last["start_s"] > duration / 2 and (not groups or last["start_s"] > groups[0]["end_s"]):
            windows["ref01_end"] = pad(last)
    if r2:
        windows["ref02"] = [pad(s) for s in r2]

    cols = 6
    th, tw = thumbs[0][1].shape[:2]
    sheet = np.full((th * ((len(thumbs) + cols - 1) // cols), tw * cols, 3), 255, np.uint8)
    for k, (t, im) in enumerate(thumbs):
        im = im.copy()
        j = min(int(np.searchsorted(T[:, 0], t)), len(T) - 1)
        bad = any(s <= t <= e for s, e in spans)
        label = f"{int(t // 60)}:{t % 60:04.1f}  m{int(T[j, 3])} k{int(T[j, 1])}"
        cv2.putText(im, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(im, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if bad:
            cv2.rectangle(im, (0, 0), (tw - 1, th - 1), (0, 0, 255), 6)
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
    cv2.imwrite(os.path.join(out, "contact.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])

    med = lambda c: float(np.median(T[T[:, 3] >= 0, c])) if c == 3 else float(np.median(T[:, c]))
    rep = {"video": os.path.basename(a.video), "size": list(size), "orientation_meta_deg": rotation, "fps": fps,
           "frames": i, "duration_s": round(duration, 2), "timestamp_drift_max_s": round(drift, 4),
           "constant_rate": drift < 1.5 / fps,
           "tracking_screen": {"every": a.every, "scale": a.scale, "fast_threshold": a.fast, "features": a.features,
                               "min_matches": a.min_matches, "min_keypoints": a.min_keypoints, "min_span_s": a.min_span_s,
                               "frames": len(rows), "risky_share": round(float(risky.mean()), 3),
                               "min_start_cells": a.min_start_cells, "map_start_estimate_s": map_start,
                               "risky_spans_s": spans, "first_good_second_s": first_good,
                               "median": {"keypoints": med(1), "cells_share": med(2), "matches": med(3), "sharpness": med(4)}},
           "boards": found, "suggested_windows_s": windows, "notes": notes,
           "outputs": {"timeline": os.path.join(out, "timeline.csv"), "contact_sheet": os.path.join(out, "contact.jpg")}}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("video", "size", "fps", "duration_s", "constant_rate")}))
    ts = rep["tracking_screen"]
    print(f"tracking screen: map can start at about {map_start} s; {ts['risky_share']:.0%} of frames risky; "
          f"{len(spans)} risky stretches: {spans[:12]}{' ...' if len(spans) > 12 else ''}")
    for k, sp in found.items():
        print(f"{k}: " + (", ".join(f"{s['start_s']}-{s['end_s']} s (up to {s['max_corners']} corners)" for s in sp) or "not found"))
    print(f"suggested windows_s: {json.dumps(windows)}")
    for n in notes:
        print(f"note: {n}")
    print(f"-> {out}")


def cmd_check_frames(a):
    """Checks that extracted frames are the video frames their names say: a sample of them is compared with the video
    decoded in order, frame by frame around each one's index. An offset other than 0 means those frames are paired
    with the wrong tracker poses (frames extracted by seeking in a variable-rate video can be)."""
    items = [l.split() for l in open(os.path.join(a.frames, "rgb.txt")) if l.strip() and not l.startswith("#")]
    if not items:
        sys.exit(f"no frames listed in {a.frames}/rgb.txt")
    names = [it[1] for it in items]
    index = [int(os.path.splitext(os.path.basename(n))[0]) for n in names]
    pick = sorted(set(np.linspace(0, len(names) - 1, min(a.samples, len(names))).round().astype(int).tolist()))
    targets = {index[k]: cv2.imread(os.path.join(a.frames, names[k]), cv2.IMREAD_GRAYSCALE) for k in pick}
    missing = [names[k] for k in pick if targets[index[k]] is None]
    if missing:
        sys.exit(f"cannot read {len(missing)} listed frame(s) in {a.frames}, for example {missing[0]}")
    want = {}
    for n in targets:
        for d in range(-a.radius, a.radius + 1):
            want.setdefault(n + d, []).append(n)
    diffs = {n: {} for n in targets}
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"OpenCV cannot open {a.video}")
    i, last = 0, max(want)
    while i <= last:
        if i in want:
            ok, bgr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for n in want[i]:
                ref = targets[n]
                h = g if g.shape == ref.shape else cv2.resize(g, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
                diffs[n][i - n] = float(np.mean(cv2.absdiff(h, ref)))
        elif not cap.grab():
            break
        i += 1
    cap.release()
    rows, off = [], []
    for n in sorted(targets):
        d = diffs[n]
        if 0 not in d:
            continue
        best = min(d, key=d.get)
        # Only call it an offset when the frame named matches clearly worse than another one (a still camera makes
        # neighbouring frames look alike, and that is not an error).
        clear = best != 0 and d[0] - d[best] > max(0.5, 0.2 * d[0])
        rows.append({"frame": n, "best_offset": best if clear else 0, "diff_named": round(d[0], 2),
                     "diff_best": round(d[best], 2)})
        if clear:
            off.append((n, best))
    for r in rows:
        print(f"  frame {r['frame']:6d}: offset {r['best_offset']:+d}  (difference {r['diff_named']} named, {r['diff_best']} best)")
    if off:
        print(f"{len(off)} of {len(rows)} sampled frames are not the frames their names say: {off}")
    else:
        print(f"all {len(rows)} sampled frames are the frames their names say")


def cmd_focal_from_lines(a):
    """The camera's focal length from the room's straight lines (doors, floors, furniture), independent of any board:
    frames every --every-s seconds, the dominant vanishing points of each, and the focal length each perpendicular
    pair implies. Use it to check a board calibration: they should agree within a few percent."""
    from qrecon import vanishing as VP
    if not os.path.isfile(a.video):
        sys.exit(f"video not found: {os.path.abspath(a.video)}")
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(a.every_s * fps)))
    est, frames, i = [], 0, 0
    while True:
        if i % step == 0:
            ok, bgr = cap.read()
            if not ok:
                break
            small = cv2.resize(bgr, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA) if a.scale != 1.0 else bgr
            fs = VP.frame_focals(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), seed=i)
            est += [f / a.scale for f in fs]
            frames += 1
        elif not cap.grab():
            break
        i += 1
    cap.release()
    est = np.array(est)
    out = {"video": os.path.basename(a.video), "frames": frames, "candidates": int(len(est))}
    if len(est) >= 5:
        q = np.percentile(est, [25, 50, 75])
        out.update(focal_px_median=float(q[1]), focal_px_iqr=[float(q[0]), float(q[2])],
                   histogram={f"{lo}-{lo + 100}": int(((est >= lo) & (est < lo + 100)).sum()) for lo in range(800, 2200, 100)})
    if a.out:
        json.dump(dict(out, all=[float(x) for x in est]), open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_charuco_survey(a):
    """Calibrations from every board window of a recording: each window alone, each board, and all views together,
    each with the lens distortion (k1) free and held at zero. One calibration from a far, nearly face-on board can
    trade focal length against distance; if the windows disagree with each other, the focal length is not settled."""
    CH, ref = _boards()
    boards = {k: CH.make_board(b) for k, b in ref["boards"].items()}
    groups = []                                            # (label, board, start, end)
    if a.windows:
        for wspec in a.windows:
            name, s0, s1 = wspec.split(":")
            groups.append((f"{name} {s0}-{s1}", name, float(s0), float(s1)))
    elif a.recording:
        win = ref["recordings"][a.recording]["windows_s"]
        for key, spans in win.items():
            board = "ref02" if key.startswith("ref02") else "ref01"
            for s0, s1 in (spans if isinstance(spans[0], list) else [spans]):
                groups.append((f"{board} {s0}-{s1}", board, float(s0), float(s1)))
    else:
        rep = json.load(open(a.report or os.path.join(REPO, "reports", os.path.splitext(os.path.basename(a.video))[0], "report.json")))
        for board, spans in rep["boards"].items():
            for sp in spans:
                s0, s1 = max(0.0, sp["start_s"] - 1.0), sp["end_s"] + 1.0
                groups.append((f"{board} {s0:.1f}-{s1:.1f}", board, s0, s1))
    if not groups:
        sys.exit("no board windows: pass --recording, --report or --windows")
    found = {g[0]: [] for g in groups}
    size = None
    for i, t, bgr in _spans(a.video, [(g[2], g[3]) for g in groups], a.every):
        size = (bgr.shape[1], bgr.shape[0])
        for label, board, s0, s1 in groups:
            if s0 <= t <= s1:
                d = CH.detect(bgr, boards[board], ref["calibration"]["min_corners"])
                if d is not None:
                    found[label].append((boards[board], d[0], d[1]))
    cfg = ref["calibration"]
    rows = [(label, found[label]) for label, _, _, _ in groups]
    for board in sorted({g[1] for g in groups}):
        rows.append((f"{board} all windows", [v for label, b, _, _ in groups if b == board for v in found[label]]))
    rows.append(("all boards", [v for label in found for v in found[label]]))
    results = []
    print(f"{'views from':24s} {'views':>5s} {'cover':>6s} {'distance m':>11s} | {'focal':>7s} {'k1':>7s} {'rms':>5s} {'std':>5s} | {'focal, k1=0':>11s} {'rms':>5s}")
    for label, views in rows:
        r = CH.calibrate_views(views, size, cfg) if size else {"ok": False, "reason": "no frames"}
        z = CH.calibrate_views(views, size, cfg, fix_k1=True) if size else r
        results.append({"views_from": label, "free_k1": r, "k1_zero": z})
        if "K" not in r:
            print(f"{label:24s} {len([v for v in views if len(v[1]) >= cfg['min_corners']]):5d}  {r.get('reason')}")
            continue
        d = r["board_distance_m"]
        print(f"{label:24s} {r['views']:5d} {r['image_coverage']:6.0%} {d[0]:5.2f}-{d[1]:<5.2f} | {r['K'][0][0]:7.1f} "
              f"{r['dist'][0]:+7.3f} {r['rms_px']:5.2f} {100 * r['focal_std_fraction']:4.1f}% | {z['K'][0][0]:11.1f} {z['rms_px']:5.2f}")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1, default=float)


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
    p = sub.add_parser("report", help="a recording's fitness for tracking and anchoring")
    p.add_argument("--video", required=True); p.add_argument("--out", help="default: reports/<video name>")
    p.add_argument("--every", type=int, default=2); p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--features", type=int, default=2000); p.add_argument("--fast", type=int, default=20)
    p.add_argument("--min-matches", type=int, default=50, help="stella needs 50 to try a map; below it is risky")
    p.add_argument("--min-keypoints", type=int, default=300); p.add_argument("--min-span-s", type=float, default=0.5)
    p.add_argument("--min-start-cells", type=float, default=0.3, help="texture cover needed to start a map")
    p.add_argument("--board-every", type=int, default=15); p.add_argument("--sheet-every", type=float, default=10.0)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("check-frames", help="are extracted frames the video frames their names say?")
    p.add_argument("--video", required=True); p.add_argument("--frames", required=True)
    p.add_argument("--samples", type=int, default=12); p.add_argument("--radius", type=int, default=4)
    p.set_defaults(fn=cmd_check_frames)
    p = sub.add_parser("focal-from-lines", help="the focal length from the room's straight lines, without a board")
    p.add_argument("--video", required=True); p.add_argument("--every-s", type=float, default=2.0)
    p.add_argument("--scale", type=float, default=0.5); p.add_argument("--out")
    p.set_defaults(fn=cmd_focal_from_lines)
    p = sub.add_parser("charuco-survey", help="calibrations from every board window of a recording")
    p.add_argument("--video", required=True); p.add_argument("--recording", help="name under recordings: in references.yaml")
    p.add_argument("--report", help="a report.json whose board spans give the windows (default: reports/<video name>)")
    p.add_argument("--windows", nargs="*", help="board:start:end, e.g. ref01:0:6 ref02:150:160")
    p.add_argument("--every", type=int, default=3); p.add_argument("--out")
    p.set_defaults(fn=cmd_charuco_survey)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
