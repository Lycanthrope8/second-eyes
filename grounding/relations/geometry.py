"""Box geometry for the relation library, v1 (A2.1b, D67).

A box comes from a validated scene record: its centre and full side lengths in metres, and a quaternion
[x, y, z, w] from the box's local axes to the scene axes (right-handed, z up). This module computes what the
relations need: corners, vertical extents, the footprint (the box's projection onto the XY plane), footprint
overlap, the exact shortest distance between two boxes, and how far one box's corners reach outside another.
All of it is exact for axis-aligned, yaw-only and fully rotated boxes. Pure Python, so the headset port can follow
it line by line; its cost on the Quest is not measured here.
"""
from __future__ import annotations

import math

EPS = 1e-9  # numerical guard only (parallel axes, empty polygons, zero-length edges); no semantic meaning

# The 12 edges of a box, as pairs of corner indices; a corner's index has bit 2, 1, 0 set for +x, +y, +z.
EDGES = tuple((i, i | bit) for i in range(8) for bit in (1, 2, 4) if not i & bit)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def rotation_matrix(q):
    """Rotation matrix of a quaternion [x, y, z, w], normalized first (validation already bounds its norm)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


class Box:
    """An oriented box with its derived geometry, computed once."""

    def __init__(self, centre, size, quaternion):
        self.centre = tuple(float(c) for c in centre)
        self.half = tuple(float(s) / 2.0 for s in size)
        r = rotation_matrix(quaternion)
        self.axes = tuple((r[0][i], r[1][i], r[2][i]) for i in range(3))  # local axes in scene coordinates
        corners = []
        for i in range(8):
            local = tuple(self.half[k] if i & (4 >> k) else -self.half[k] for k in range(3))
            offset = (r[0][0] * local[0] + r[0][1] * local[1] + r[0][2] * local[2],
                      r[1][0] * local[0] + r[1][1] * local[1] + r[1][2] * local[2],
                      r[2][0] * local[0] + r[2][1] * local[1] + r[2][2] * local[2])
            corners.append(add(self.centre, offset))
        self.corners = tuple(corners)
        zs = [c[2] for c in corners]
        self.z_min, self.z_max = min(zs), max(zs)
        self.footprint = convex_hull([(c[0], c[1]) for c in corners])
        self.footprint_area = polygon_area(self.footprint)
        up = self.axes[2]
        self.tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))  # local +z against scene +z

    def local(self, p):
        """A scene point in this box's local coordinates."""
        d = sub(p, self.centre)
        return (dot(d, self.axes[0]), dot(d, self.axes[1]), dot(d, self.axes[2]))


def convex_hull(points):
    """Counter-clockwise convex hull of 2D points (monotone chain), without repeated or collinear points."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= EPS:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= EPS:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def clip_polygon(subject, clip):
    """Sutherland-Hodgman: the part of a polygon inside a convex counter-clockwise polygon."""
    output = list(subject)
    for i in range(len(clip)):
        if not output:
            break
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % len(clip)]
        ex, ey = bx - ax, by - ay

        def inside(p):
            return ex * (p[1] - ay) - ey * (p[0] - ax) >= -EPS

        def crossing(p, q):
            dx, dy = q[0] - p[0], q[1] - p[1]
            denom = dx * ey - dy * ex
            if abs(denom) < 1e-18:
                return q
            t = ((ax - p[0]) * ey - (ay - p[1]) * ex) / denom
            return (p[0] + t * dx, p[1] + t * dy)

        points, output = output, []
        prev = points[-1]
        for cur in points:
            if inside(cur):
                if not inside(prev):
                    output.append(crossing(prev, cur))
                output.append(cur)
            elif inside(prev):
                output.append(crossing(prev, cur))
            prev = cur
    return output


def intersection_area(poly_a, poly_b):
    """Area of the intersection of two convex counter-clockwise polygons."""
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    return polygon_area(clip_polygon(poly_a, poly_b))


def boxes_intersect(a: Box, b: Box) -> bool:
    """Separating-axis test with the 15 candidate axes; touching boxes count as intersecting."""
    t = sub(b.centre, a.centre)
    axes = list(a.axes) + list(b.axes)
    for u in a.axes:
        for v in b.axes:
            c = cross(u, v)
            if norm(c) > EPS:
                axes.append(c)
    for axis in axes:
        ra = sum(a.half[i] * abs(dot(a.axes[i], axis)) for i in range(3))
        rb = sum(b.half[i] * abs(dot(b.axes[i], axis)) for i in range(3))
        if abs(dot(t, axis)) > ra + rb:
            return False
    return True


def point_box_distance(p, box: Box) -> float:
    loc = box.local(p)
    d2 = 0.0
    for i in range(3):
        excess = abs(loc[i]) - box.half[i]
        if excess > 0.0:
            d2 += excess * excess
    return math.sqrt(d2)


def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def segment_distance(p1, q1, p2, q2) -> float:
    """Shortest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection, 5.1.9)."""
    d1, d2, r = sub(q1, p1), sub(q2, p2), sub(p1, p2)
    a, e, f = dot(d1, d1), dot(d2, d2), dot(d2, r)
    if a <= EPS and e <= EPS:
        s = t = 0.0
    elif a <= EPS:
        s, t = 0.0, _clamp01(f / e)
    else:
        c = dot(d1, r)
        if e <= EPS:
            t, s = 0.0, _clamp01(-c / a)
        else:
            b = dot(d1, d2)
            denom = a * e - b * b
            s = _clamp01((b * f - c * e) / denom) if denom > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp01(-c / a)
            elif t > 1.0:
                t, s = 1.0, _clamp01((b - c) / a)
    return norm(sub(add(p1, scale(d1, s)), add(p2, scale(d2, t))))


def box_distance(a: Box, b: Box) -> float:
    """Exact shortest distance between two boxes; 0 if they intersect or touch.

    For disjoint convex boxes the closest pair involves a corner of one box or two edges, so the minimum over
    corner-to-box and edge-to-edge distances is exact.
    """
    if boxes_intersect(a, b):
        return 0.0
    best = min(point_box_distance(c, b) for c in a.corners)
    best = min(best, min(point_box_distance(c, a) for c in b.corners))
    for i, j in EDGES:
        for k, m in EDGES:
            best = min(best, segment_distance(a.corners[i], a.corners[j], b.corners[k], b.corners[m]))
    return best


def footprint_overlap(t: Box, a: Box):
    """Footprint intersection area over the smaller footprint area; None if a footprint is degenerate."""
    smaller = min(t.footprint_area, a.footprint_area)
    if smaller < EPS:
        return None
    return intersection_area(t.footprint, a.footprint) / smaller


def vertical_overlap(t: Box, a: Box) -> float:
    """Length (m) shared by two boxes' vertical extents; 0 or negative when they don't overlap."""
    return min(t.z_max, a.z_max) - max(t.z_min, a.z_min)


def containment_excess(target: Box, container: Box) -> float:
    """Largest amount (m) by which a target corner lies outside the container along a container axis."""
    worst = -math.inf
    for c in target.corners:
        loc = container.local(c)
        for i in range(3):
            worst = max(worst, abs(loc[i]) - container.half[i])
    return worst
