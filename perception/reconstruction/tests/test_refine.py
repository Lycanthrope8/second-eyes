import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qrecon import refine as RF  # noqa: E402


def _drifting_piece():
    """A 60 s walk whose tracker scale drifts: the tracker's units grow 50% longer by the end."""
    t = np.arange(0.0, 60.0, 0.2)
    vu = np.c_[0.4 * np.ones_like(t), 0.2 * np.cos(t / 3), 0.05 * np.sin(t / 5)]       # tracker velocity, tracker units
    s = 1.0 + 0.5 * t / 60.0                                                              # metres per tracker unit
    cu = np.cumsum(vu * 0.2, axis=0)
    cw = np.cumsum(vu * s[:, None] * 0.2, axis=0) + [2.0, 0.0, -1.0]
    Tu = np.tile(np.eye(4), (len(t), 1, 1)); Tu[:, :3, 3] = cu
    return t, Tu, cw


def _fit(t, Tu, cw, lo, hi):
    ix = np.flatnonzero((t >= lo) & (t <= hi))
    return ix, RF.umeyama(Tu[ix, :3, 3], cw[ix])


def test_board_windows_at_both_ends_correct_the_drift_between_them():
    t, Tu, cw = _drifting_piece()
    ia, A = _fit(t, Tu, cw, 0.0, 5.0)
    ib, B = _fit(t, Tu, cw, 55.0, 60.0)
    one = RF.apply(RF._params(*A), Tu)[:, :3, 3]                                          # one similarity, from the first board
    W, info = RF.refine_piece(t, Tu, A, windows=[(ia[::5], A), (ib[::5], B)], max_len_s=10.0)
    e_one, e_ref = np.linalg.norm(one - cw, axis=1), np.linalg.norm(W[:, :3, 3] - cw, axis=1)
    assert np.median(e_one) > 1.0 and np.median(e_ref) < 0.3 * np.median(e_one) and e_ref[-1] < 0.1
    assert info["stretches"] == 6 and info["scale_change"][1] > 1.2


def test_located_frames_along_the_piece_correct_it_and_a_piece_without_evidence_is_unchanged():
    t, Tu, cw = _drifting_piece()
    ia, A = _fit(t, Tu, cw, 0.0, 5.0)
    rng = np.random.default_rng(1)
    loc = []
    for i in range(10, len(t), 25):
        Tw = np.eye(4); Tw[:3, 3] = cw[i] + rng.normal(0, 0.03, 3)
        loc.append((i, Tw))
    W, _ = RF.refine_piece(t, Tu, A, windows=[(ia[::5], A)], located=loc, max_len_s=10.0)
    one = RF.apply(RF._params(*A), Tu)[:, :3, 3]
    assert np.median(np.linalg.norm(W[:, :3, 3] - cw, axis=1)) < 0.3 * np.median(np.linalg.norm(one - cw, axis=1))
    W0, info0 = RF.refine_piece(t, Tu, A)
    assert np.allclose(W0[:, :3, 3], one) and info0["windows"] == 0


def test_a_piece_held_by_its_board_stays_put_there_when_located_frames_are_biased():
    t, Tu, cw = _drifting_piece()
    ia, A = _fit(t, Tu, cw, 0.0, 5.0)
    start = RF.apply(RF._params(*A), Tu)
    loc = []
    for i in range(150, len(t), 20):                                  # frames located 0.5 m off, far from the board
        Tw = start[i].copy(); Tw[0, 3] += 0.5
        loc.append((i, Tw))
    W, info = RF.refine_piece(t, Tu, A, windows=[(ia, A)], located=loc, max_len_s=10.0)
    near = np.linalg.norm(W[ia, :3, 3] - start[ia, :3, 3], axis=1)
    assert near.max() < 0.05                                          # where the board was seen, the piece does not move
    assert 1 / 1.6 - 1e-6 <= info["scale_change"][0] and info["scale_change"][1] <= 1.6 + 1e-6
    assert info["evidence_rms_after_m"] < info["evidence_rms_before_m"]
