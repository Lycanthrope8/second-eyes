"""Drift-aware refinement of a joined piece (D142): one similarity per stretch of the piece instead of one for all of it.

A camera-only tracker's scale drifts along a long piece, so a single similarity, fitted where the piece was anchored
(at a board, or by the frames that shared views located), places the rest of the piece too far or too near. Here the
piece is cut into stretches of at most max_len_s seconds, each with its own similarity (log scale, rotation, offset),
and all are solved together by least squares from:
  - board windows inside the piece, each with its own local fit (strong);
  - frames located by shared views that agreed with one placement, each a camera pose in the target frame (weaker);
  - continuity: neighbouring stretches must place the frame at their boundary alike, and their scales may change
    only gradually (the drift model);
  - a weak pull towards the starting similarity, so a stretch with no evidence follows its neighbours.
Near a boundary the two similarities are blended over blend_s seconds, so the path has no jump.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

SIGMA = {"window_m": 0.02, "window_rad": 0.01, "located_m": 0.10, "located_rad": 0.05,
         "continuity_m": 0.01, "continuity_rad": 0.005, "scale_step": 0.05}
PRIOR = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0])        # log scale, rotation (rad), offset (m)


def umeyama(A, B):
    """Similarity (s, R, t) with B ~ s R A + t, by least squares."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / A0.var(0).sum())
    return s, R, mb - s * R @ ma


def stretches(times, max_len_s):
    """Index ranges [(i0, i1), ...] cutting a piece into stretches of at most max_len_s seconds."""
    n = len(times)
    if n < 2:
        return [(0, n)]
    k = max(1, int(np.ceil((times[-1] - times[0]) / max_len_s - 1e-9)))
    cuts = np.searchsorted(times, np.linspace(times[0], times[-1], k + 1)[1:-1])
    b = sorted(set([0] + [int(c) for c in cuts] + [n]))
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] > b[i]]


def _params(s, R, t):
    return np.r_[np.log(s), Rot.from_matrix(R).as_rotvec(), np.asarray(t, float)]


def _sim3(x):
    return float(np.exp(x[0])), Rot.from_rotvec(x[1:4]).as_matrix(), x[4:7]


def apply(x, T):
    s, R, t = _sim3(x)
    out = T.copy()
    out[..., :3, 3] = s * T[..., :3, 3] @ R.T + t
    out[..., :3, :3] = np.einsum("ij,...jk->...ik", R, T[..., :3, :3])
    return out


def refine_piece(times, T_u, sim3_0, windows=(), located=(), max_len_s=15.0, blend_s=1.0, scale_bound=1.6):
    """times (N,), T_u (N, 4, 4) the piece's camera poses in tracker units, sim3_0 = (s, R, t) its starting similarity;
    windows: [(frame indices, (s, R, t))] board fits (the piece's own anchor among them); located: [(frame index,
    T_world (4, 4))] frames located by shared views, only those that agreed with one placement. Plain least squares, so
    continuity is never traded away; each stretch's scale stays within scale_bound of the starting scale.
    Returns (world poses (N, 4, 4), info with the evidence's rms position error before and after)."""
    st = stretches(times, max_len_s)
    K = len(st)
    seg = np.zeros(len(times), int)
    for k, (a, b) in enumerate(st):
        seg[a:b] = k
    x0 = np.tile(_params(*sim3_0), K)
    C, Rc = T_u[:, :3, 3], T_u[:, :3, :3]
    win = [(np.asarray(ix, int), (float(s), np.asarray(R), np.asarray(t))) for ix, (s, R, t) in windows if len(ix)]
    loc = [(int(i), np.asarray(Tw)) for i, Tw in located]

    def residuals(x):
        X = x.reshape(K, 7)
        S = [_sim3(X[k]) for k in range(K)]
        out = []
        for ix, (sw, Rw, tw) in win:
            for i in ix:
                s, R, t = S[seg[i]]
                out.append((s * R @ C[i] + t - (sw * Rw @ C[i] + tw)) / SIGMA["window_m"])
                out.append(Rot.from_matrix(R @ Rw.T).as_rotvec() / SIGMA["window_rad"])
        for i, Tw in loc:
            s, R, t = S[seg[i]]
            out.append((s * R @ C[i] + t - Tw[:3, 3]) / SIGMA["located_m"])
            out.append(Rot.from_matrix((R @ Rc[i]) @ Tw[:3, :3].T).as_rotvec() / SIGMA["located_rad"])
        for k in range(K - 1):
            b = st[k + 1][0]
            (s1, R1, t1), (s2, R2, t2) = S[k], S[k + 1]
            out.append((s1 * R1 @ C[b] + t1 - (s2 * R2 @ C[b] + t2)) / SIGMA["continuity_m"])
            out.append(Rot.from_matrix(R1 @ R2.T).as_rotvec() / SIGMA["continuity_rad"])
            out.append(np.atleast_1d((X[k, 0] - X[k + 1, 0]) / SIGMA["scale_step"]))
        out.append(((X - x0.reshape(K, 7)) / PRIOR).ravel())
        return np.concatenate([np.ravel(o) for o in out])

    def evidence_rms(x):
        X = x.reshape(K, 7)
        S = [_sim3(X[k]) for k in range(K)]
        e = [np.linalg.norm(S[seg[i]][0] * S[seg[i]][1] @ C[i] + S[seg[i]][2] - (sw * Rw @ C[i] + tw)) for ix, (sw, Rw, tw) in win for i in ix]
        e += [np.linalg.norm(S[seg[i]][0] * S[seg[i]][1] @ C[i] + S[seg[i]][2] - Tw[:3, 3]) for i, Tw in loc]
        return float(np.sqrt(np.mean(np.square(e)))) if e else 0.0

    if win or loc:
        lb, ub = np.full(7 * K, -np.inf), np.full(7 * K, np.inf)
        lb[0::7], ub[0::7] = x0[0] - np.log(scale_bound), x0[0] + np.log(scale_bound)
        X = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=100 * 7 * K).x.reshape(K, 7)
    else:
        X = x0.reshape(K, 7)
    W = np.empty((len(times), 4, 4))
    for i in range(len(times)):
        k = seg[i]
        x = X[k]
        if blend_s > 0:                         # blend with a neighbour near the boundary between them
            if k + 1 < K and times[st[k + 1][0]] - times[i] < blend_s:
                w = 0.5 * (1 - (times[st[k + 1][0]] - times[i]) / blend_s)
                x = (1 - w) * X[k] + w * X[k + 1]
            elif k > 0 and times[i] - times[st[k][0]] < blend_s:
                w = 0.5 * (1 - (times[i] - times[st[k][0]]) / blend_s)
                x = (1 - w) * X[k] + w * X[k - 1]
        W[i] = apply(x, T_u[i])
    scales = np.exp(X[:, 0]) / sim3_0[0]
    info = {"stretches": K, "windows": len(win), "located": len(loc),
            "scale_change": [round(float(scales.min()), 3), round(float(scales.max()), 3)],
            "evidence_rms_before_m": round(evidence_rms(x0), 3), "evidence_rms_after_m": round(evidence_rms(X.ravel()), 3)}
    return W, info
