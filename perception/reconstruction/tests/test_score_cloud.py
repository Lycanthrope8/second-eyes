import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import score_cloud as SC  # noqa: E402


def _ply(path, P, C):
    v = np.zeros(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = P.T
    v["red"], v["green"], v["blue"] = C.T
    path.write_bytes(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
                     f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode() + v.tobytes())


def _room(n, rng):
    """Floor and two walls of a 4 x 3 m room, 2.5 m high."""
    f = np.c_[rng.uniform(0, 4, n), np.zeros(n), rng.uniform(0, 3, n)]
    w1 = np.c_[rng.uniform(0, 4, n), rng.uniform(-2.5, 0, n), np.zeros(n)]
    w2 = np.c_[np.zeros(n), rng.uniform(-2.5, 0, n), rng.uniform(0, 3, n)]
    return np.concatenate([f, w1, w2])


def test_scores_reward_points_on_the_reference_and_alignment_recovers_a_small_shift(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    ref = _room(40000, rng)
    _ply(tmp_path / "ref.ply", ref, np.full((len(ref), 3), 128))
    good = _room(8000, rng) + rng.normal(0, 0.01, (24000, 3))
    bad = np.c_[rng.uniform(0, 4, 4000), rng.uniform(-2, 0, 4000), rng.uniform(0.5, 2.5, 4000)]       # floating inside the room
    runs = tmp_path / "runs"; runs.mkdir()
    P = np.concatenate([good, bad])
    _ply(runs / "m.ply", P + [0.04, 0, 0], np.full((len(P), 3), 90))
    _ply(runs / "m_by_run.ply", P + [0.04, 0, 0], np.r_[np.tile([230, 25, 75], (len(good), 1)), np.tile([60, 180, 75], (len(bad), 1))])
    (runs / "m.json").write_text(json.dumps({"run_colors": {"good": [230, 25, 75], "bad": [60, 180, 75]}}))
    monkeypatch.setattr(SC, "HERE", tmp_path)
    sys.argv = ["score_cloud.py", "--reference", str(tmp_path / "ref.ply"), "--cloud", "m", "--align", "--by-run"]
    SC.main()
    r = json.load(open(runs / "m_score.json"))
    assert abs(r["alignment_correction"]["translation_m"] - 0.04) < 0.01
    assert r["runs"]["good"]["within_5cm"] > 0.95 and r["runs"]["bad"]["within_10cm"] < 0.2
    assert 0.6 < r["whole"]["precision_10cm"] < 0.95 and r["whole"]["recall_10cm"] > 0.9
