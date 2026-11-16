"""Shared geometry. Poses are camera-to-world 4x4 (T_wc), OpenCV camera axes: x right, y down, z forward."""
import numpy as np


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def quat_to_T(tx, ty, tz, qx, qy, qz, qw):
    """TUM trajectory line -> T_wc."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def K_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def backproject(depth, K, T_wc=None, stride=1, mask=None):
    """Depth image (metres, 0 = invalid) -> points (N,3), plus the pixel coordinates used (N,2: u,v)."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    ok = z > 0
    if mask is not None:
        ok &= mask[::stride, ::stride]
    u, v, z = u[ok].astype(np.float64), v[ok].astype(np.float64), z[ok].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1)
    if T_wc is not None:
        pts = pts @ T_wc[:3, :3].T + T_wc[:3, 3]
    return pts, np.stack([u, v], axis=1)


def project(points_w, K, T_wc):
    """World points -> (u, v, z_cam). z <= 0 means behind the camera."""
    Tcw = inv(T_wc)
    pc = points_w @ Tcw[:3, :3].T + Tcw[:3, 3]
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
    return u, v, z


def rotation_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def umeyama(src, dst, with_scale=True):
    """Similarity (s, R, t) minimising ||dst - (s R src + t)||. src, dst: (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (xs ** 2).sum() * len(src)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3_to_pose(T_wc, s, R, t):
    """Map a pose from an arbitrary-scale frame into the target frame."""
    T = np.eye(4)
    T[:3, :3] = R @ T_wc[:3, :3]
    T[:3, 3] = s * R @ T_wc[:3, 3] + t
    return T
