import csv
"""End-to-end check of the video tools on a synthetic recording: a board segment, then a later second board."""
import json, os, shutil, subprocess, sys
import numpy as np
import cv2
import pytest
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qrecon import charuco as CH, geometry as G
from test_charuco import SPEC, _view, _look_at

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = G.K_matrix(900.0, 900.0, 640.0, 360.0)


def test_video_tools_end_to_end(tmp_path, monkeypatch):
    b1, b2 = CH.make_board(SPEC), CH.make_board(dict(SPEC, first_marker_id=20))
    T_b2 = np.eye(4); T_b2[:3, 3] = [2.5, 0.0, 1.0]                    # board 02 in board 01's frame
    fps, frames, poses = 10.0, [], []
    centre = np.array([0.14, 0.10, 0.0])
    for k in range(60):                                                 # 0-6 s: board 01, translated arc
        a = np.radians(-35 + 70 * k / 59)
        T = _look_at(centre + [0.5 * np.sin(a), 0.04 * np.cos(2 * a), -0.85 + 0.15 * np.cos(a)], centre)
        frames.append(_view(b1, T, K)); poses.append(T)
    for k in range(40):                                                 # 6-10 s: board 02
        a = np.radians(-30 + 60 * k / 39)
        Tl = _look_at(centre + [0.45 * np.sin(a), 0.03 * np.cos(a), -0.8], centre)
        frames.append(_view(b2, Tl, K)); poses.append(T_b2 @ Tl)
    video = str(tmp_path / "house.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))
    for f in frames:
        vw.write(f)
    vw.release()
    ref = yaml.safe_load(open(os.path.join(REPO, "configs", "references.yaml")))
    ref["boards"]["ref01"] = dict(SPEC, mounting="upright_level")
    ref["boards"]["ref02"] = dict(SPEC, first_marker_id=20, mounting="upright_level")
    ref_path = tmp_path / "references.yaml"
    yaml.safe_dump(ref, open(ref_path, "w"))
    calib = {"K": K.tolist(), "dist": [0, 0, 0, 0, 0], "size": [1280, 720]}
    json.dump(calib, open(tmp_path / "calib.json", "w"))
    scale = 2.2                                                          # tracker trajectory in arbitrary units
    with open(tmp_path / "traj.txt", "w") as f:
        from scipy.spatial.transform import Rotation
        for k, T in enumerate(poses):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            p = T[:3, 3] / scale
            f.write(f"{k / fps:.6f} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    env = dict(os.environ)
    # point the tool at the test's references.yaml
    cfg_dir = os.path.join(REPO, "configs")
    orig = open(os.path.join(cfg_dir, "references.yaml")).read()
    try:
        open(os.path.join(cfg_dir, "references.yaml"), "w").write(open(ref_path).read())
        out = subprocess.run([sys.executable, tool, "frames", "--video", video, "--out", str(tmp_path / "fr"), "--every", "2", "--scale", "0.5"],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        lines = [l for l in open(tmp_path / "fr" / "rgb.txt") if not l.startswith("#")]
        assert len(lines) == 50 and os.path.exists(tmp_path / "fr" / lines[0].split()[1])
        out = subprocess.run([sys.executable, tool, "charuco-inspect", "--video", video, "--end", "9", "--every", "20"],
                             capture_output=True, text=True, env=env)
        assert "DICT_5X5_100" in out.stdout, out.stdout + out.stderr
        out = subprocess.run([sys.executable, tool, "charuco-anchor", "--video", video, "--trajectory", str(tmp_path / "traj.txt"),
                              "--calib", str(tmp_path / "calib.json"), "--ref01-end", "5.9", "--ref02-start", "6.0", "--ref02-end", "10",
                              "--every", "1", "--out", str(tmp_path / "anchored.txt")], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stdout + out.stderr
        rep = json.load(open(str(tmp_path / "anchored.txt") + ".json"))
        assert abs(rep["ref01"]["scale"] - scale) / scale < 0.02
        assert rep["ref02"]["ok"] and abs(rep["ref02"]["scale_drift"]) < 0.02
        link = np.array(rep["ref02"]["board02_in_board01"])
        assert np.allclose(link[:3, 3], [2.5, 0.0, 1.0], atol=0.05)
    finally:
        open(os.path.join(cfg_dir, "references.yaml"), "w").write(orig)


def test_missing_video_is_an_error_not_zero_frames(tmp_path):
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    out = subprocess.run([sys.executable, tool, "frames", "--video", str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "fr")],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "video not found" in (out.stderr + out.stdout)


def test_report_flags_a_blank_stretch_and_finds_the_board(tmp_path):
    """Synthetic recording: 0-3 s textured scene with board 01, 3-5 s a bare wall, 5-8 s textured again."""
    ref = yaml.safe_load(open(os.path.join(REPO, "configs", "references.yaml")))
    board = CH.make_board(ref["boards"]["ref01"])
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.random((900, 3000)) * 255).astype(np.uint8), (0, 0), 3)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    btex = cv2.cvtColor(board.generateImage((500, 300), marginSize=20), cv2.COLOR_GRAY2BGR)
    fps, video = 30.0, str(tmp_path / "walk.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))
    for k in range(240):
        t = k / fps
        if 3.0 <= t < 5.0:
            f = np.full((720, 1280, 3), 180, np.uint8)                       # bare wall
            f = cv2.add(f, (rng.normal(0, 1.5, f.shape)).clip(0, 255).astype(np.uint8))
        else:
            x0 = 4 * k
            f = tex[90:810, x0:x0 + 1280].copy()
            if t < 3.0:
                f[200:500, 390:890] = btex
        vw.write(f)
    vw.release()
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    out = subprocess.run([sys.executable, tool, "report", "--video", video, "--out", str(tmp_path / "rep"), "--sheet-every", "2"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    rep = json.load(open(tmp_path / "rep" / "report.json"))
    assert rep["frames"] == 240 and abs(rep["fps"] - 30) < 0.01 and rep["constant_rate"]
    spans = rep["tracking_screen"]["risky_spans_s"]
    assert any(s <= 3.2 and e >= 4.8 for s, e in spans), spans                # the bare wall is flagged
    assert not any(e <= 3.0 or s >= 5.2 for s, e in spans), spans             # and nothing else
    b = rep["boards"]["ref01"]
    assert b and b[0]["start_s"] <= 0.5 and 2.0 <= b[0]["end_s"] < 3.0 and not rep["boards"]["ref02"]
    assert rep["tracking_screen"]["map_start_estimate_s"] == 0.0
    assert rep["suggested_windows_s"]["ref01_start"][0] == 0.0 and rep["suggested_windows_s"]["calibration"][0] == 0.0
    assert os.path.exists(tmp_path / "rep" / "contact.jpg")
    lines = open(tmp_path / "rep" / "timeline.csv").read().splitlines()
    assert len(lines) == 1 + 120


def _barcode_frames(n, w=640, h=360):
    """Frames that carry their own index as a 10-bit barcode along the top, over a moving texture."""
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur((rng.random((h, w + 3 * n)) * 255).astype(np.uint8), (0, 0), 2)
    for k in range(n):
        f = cv2.cvtColor(tex[:, 3 * k:3 * k + w].copy(), cv2.COLOR_GRAY2BGR)
        for b in range(10):
            f[0:40, b * 64:(b + 1) * 64] = 255 if (k >> b) & 1 else 0
        yield f


def _barcode(img):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = g.shape[1] / 640.0
    return sum(1 << b for b in range(10) if g[int(8 * s):int(32 * s), int((b * 64 + 12) * s):int((b * 64 + 52) * s)].mean() > 127)


def _extracted_codes(folder):
    rows = [l.split() for l in open(os.path.join(folder, "rgb.txt")) if l.strip() and not l.startswith("#")]
    return [(int(os.path.basename(n)[:6]), _barcode(cv2.imread(os.path.join(folder, n)))) for _, n in rows]


def test_frames_carry_their_true_index_and_check_frames_finds_a_mislabel(tmp_path):
    video = str(tmp_path / "coded.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 360))
    for f in _barcode_frames(240):
        vw.write(f)
    vw.release()
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    out = subprocess.run([sys.executable, tool, "frames", "--video", video, "--out", str(tmp_path / "w"), "--every", "2",
                          "--start", "3.0", "--end", "5.0"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    codes = _extracted_codes(tmp_path / "w")
    assert codes and all(name == code for name, code in codes), codes
    assert codes[0][0] == 90 and codes[-1][0] == 150
    assert "decoded=in-order" in open(tmp_path / "w" / "rgb.txt").readline()
    out = subprocess.run([sys.executable, tool, "check-frames", "--video", video, "--frames", str(tmp_path / "w")],
                         capture_output=True, text=True)
    assert "are the frames their names say" in out.stdout, out.stdout + out.stderr
    # mislabel one frame: the image of frame 120 named as frame 122
    rgb = open(tmp_path / "w" / "rgb.txt").read()
    os.remove(tmp_path / "w" / "rgb" / "000122.jpg")
    os.rename(tmp_path / "w" / "rgb" / "000120.jpg", tmp_path / "w" / "rgb" / "000122.jpg")
    open(tmp_path / "w" / "rgb.txt", "w").write("".join(l for l in rgb.splitlines(True) if not l.rstrip().endswith("rgb/000120.jpg")))
    out = subprocess.run([sys.executable, tool, "check-frames", "--video", video, "--frames", str(tmp_path / "w"),
                          "--samples", "40"], capture_output=True, text=True)
    assert "(122, -2)" in out.stdout, out.stdout + out.stderr


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to make a variable-rate video")
def test_frames_are_exact_on_variable_rate_video(tmp_path):
    for k, f in enumerate(_barcode_frames(300)):
        cv2.imwrite(str(tmp_path / f"f{k:05d}.png"), f)
    cfr, vfr = str(tmp_path / "cfr.mp4"), str(tmp_path / "vfr.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(tmp_path / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", cfr], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", cfr, "-vf", "setpts=(N/30+0.08*sin(2*PI*N/200))/TB",
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", vfr], check=True)
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    for start in (2.3, 4.6, 7.3):
        out_dir = tmp_path / f"w{start}"
        out = subprocess.run([sys.executable, tool, "frames", "--video", vfr, "--out", str(out_dir), "--every", "2",
                              "--start", str(start), "--end", str(start + 1.0)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        codes = _extracted_codes(out_dir)
        assert codes and all(name == code for name, code in codes), (start, codes)


def test_frame_lists_differ_on_a_new_extraction_with_the_same_count():
    from qrecon import framelist as FL
    old = "# v.mp4 every=2 scale=1.0 size=(1920, 1080)\n1.0 rgb/000030.jpg\n2.0 rgb/000060.jpg\n"
    new = "# v.mp4 every=2 scale=1.0 size=(1920, 1080) decoded=in-order extracted=2026-11-16T14:00:00\n" + old.split("\n", 1)[1]
    assert FL.count(old) == FL.count(new) == 2
    assert FL.differs(new, old) and not FL.differs(old, old.replace("\n", "\r\n"))


def _render_board(board, spec, T_wc, K, size=(1280, 720), px_per_m=3000):
    W = int(spec["squares_x"] * spec["square_length_m"] * px_per_m)
    H = int(spec["squares_y"] * spec["square_length_m"] * px_per_m)
    tex = board.generateImage((W, H), marginSize=0)
    corners = np.array([[0, 0, 0], [W / px_per_m, 0, 0], [W / px_per_m, H / px_per_m, 0], [0, H / px_per_m, 0]])
    Tcw = G.inv(T_wc)
    rvec, _ = cv2.Rodrigues(Tcw[:3, :3])
    uv, _ = cv2.projectPoints(corners, rvec, Tcw[:3, 3], K, np.zeros(5))
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), uv.reshape(4, 2).astype(np.float32))
    return cv2.cvtColor(cv2.warpPerspective(tex, Hm, size, borderValue=200), cv2.COLOR_GRAY2BGR)


def test_charuco_survey_calibrates_each_window_and_all_together(tmp_path):
    ref = yaml.safe_load(open(os.path.join(REPO, "configs", "references.yaml")))
    specs = {k: ref["boards"][k] for k in ("ref01", "ref02")}
    boards = {k: CH.make_board(v) for k, v in specs.items()}
    video = str(tmp_path / "boards.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720))
    for k in range(60):                                  # 0-3 s board 01, 3-6 s board 02, varied tilted views
        name = "ref01" if k < 30 else "ref02"
        sp = specs[name]
        centre = np.array([sp["squares_x"] * sp["square_length_m"] / 2, sp["squares_y"] * sp["square_length_m"] / 2, 0.0])
        a = np.radians(-30 + 60 * (k % 30) / 29)
        cam = centre + [0.25 * np.sin(a), 0.06 * np.cos(3 * a), -0.32 + 0.05 * np.cos(a)]
        vw.write(_render_board(boards[name], sp, _look_at(cam, centre + [0.02 * np.cos(k), 0.015 * np.sin(k), 0]), K))
    vw.release()
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "video_tools.py"), "charuco-survey", "--video", video,
                          "--windows", "ref01:0:2.9", "ref02:3:5.9", "--every", "1", "--out", str(tmp_path / "s.json")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    res = {r["views_from"]: r for r in json.load(open(tmp_path / "s.json"))}
    assert set(res) == {"ref01 0-2.9", "ref02 3-5.9", "ref01 all windows", "ref02 all windows", "all boards"}
    for key in ("ref01 0-2.9", "ref02 3-5.9", "all boards"):
        assert abs(res[key]["free_k1"]["K"][0][0] - 900) / 900 < 0.03, (key, res[key]["free_k1"])
    assert "all boards" in out.stdout


def _board_video(tmp_path, fps=30.0):
    ref = yaml.safe_load(open(os.path.join(REPO, "configs", "references.yaml")))
    board = CH.make_board(ref["boards"]["ref01"])
    rng = np.random.default_rng(0)
    tex = cv2.cvtColor(cv2.normalize(cv2.GaussianBlur((rng.random((900, 3000)) * 255).astype(np.uint8), (0, 0), 3), None, 0, 255,
                                     cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    btex = cv2.cvtColor(board.generateImage((500, 300), marginSize=20), cv2.COLOR_GRAY2BGR)
    video = str(tmp_path / "walk.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))
    for k in range(240):
        f = tex[90:810, 4 * k:4 * k + 1280].copy()
        if k / fps < 3.0 or k / fps >= 6.0:
            f[200:500, 390:890] = btex
        vw.write(f)
    vw.release()
    return video


def test_board_spans_are_found_and_frames_follow_them(tmp_path):
    video = _board_video(tmp_path)
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    out = subprocess.run([sys.executable, tool, "board-spans", "--video", video, "--out", str(tmp_path / "b.json"), "--every", "15"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    b = json.load(open(tmp_path / "b.json"))
    s = b["spans"]["ref01"]
    assert len(s) == 2 and s[0]["start_s"] == 0.0 and 2.4 <= s[0]["end_s"] < 3.0 and 6.0 <= s[1]["start_s"] < 6.6
    assert not b["windows_s"]["ref02"] and b["windows_s"]["ref01"][1][0] == round(s[1]["start_s"] - 1.0, 2)
    out = subprocess.run([sys.executable, tool, "frames", "--video", video, "--every", "10", "--board-spans", str(tmp_path / "b.json"),
                          "--out", str(tmp_path / "bf")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    ts = [float(l.split()[0]) for l in open(tmp_path / "bf" / "rgb.txt") if not l.startswith("#")]
    assert ts and all(t <= 3.6 or t >= 5.0 for t in ts)                       # nothing from the stretch without the board
    rows = list(csv.DictReader(open(tmp_path / "bf" / "source_times.csv")))
    assert len(rows) == len(ts) and all(abs(float(r["source_s"]) - float(r["nominal_s"])) < 1e-3 for r in rows)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg and ffprobe")
def test_source_times_are_the_videos_own_timestamps(tmp_path):
    for k, f in enumerate(_barcode_frames(200)):
        cv2.imwrite(str(tmp_path / f"f{k:05d}.png"), f)
    cfr, vfr = str(tmp_path / "cfr.mp4"), str(tmp_path / "vfr.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(tmp_path / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", cfr], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", cfr, "-vf", "setpts=(N/30+0.08*sin(2*PI*N/200))/TB",
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", vfr], check=True)
    pts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0", vfr],
                         capture_output=True, text=True).stdout
    truth = [float(x.strip().strip(",")) for x in pts.splitlines() if x.strip().strip(",")]
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    out_dir = tmp_path / "fr"
    assert subprocess.run([sys.executable, tool, "frames", "--video", vfr, "--every", "2", "--out", str(out_dir)]).returncode == 0
    (out_dir / "source_times.csv").unlink()                                  # an extraction from before R3.14
    out = subprocess.run([sys.executable, tool, "source-times", "--video", vfr, "--frames", str(out_dir)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    rows = list(csv.DictReader(open(out_dir / "source_times.csv")))
    assert len(rows) == 100
    assert max(abs(float(r["source_s"]) - (truth[int(r["index"])] - truth[0])) for r in rows) < 1e-3
    assert max(abs(float(r["source_s"]) - float(r["nominal_s"])) for r in rows) > 0.03     # nominal times differ here
