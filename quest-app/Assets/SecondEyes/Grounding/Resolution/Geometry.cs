using System;
using System.Collections.Generic;

namespace SecondEyes.Grounding.Resolution
{
    // A2.7 delivery 2 (D115): grounding/relations/geometry.py (A2.1b, D67), line by line, free of Unity. A box comes from
    // a validated scene record: its centre, full side lengths and a quaternion [x, y, z, w] from local to scene axes
    // (right-handed, z up). Every expression keeps the Python's operand order, and Py's helpers stand in for the
    // builtins, so the values match the PC's bit for bit where the platform's sqrt and acos do.
    public static class Geo
    {
        public const double EPS = 1e-9;   // a numerical guard only

        /// <summary>The 12 edges, as pairs of corner indices (bit 2, 1, 0 set for +x, +y, +z), in the Python's order.</summary>
        public static readonly int[][] Edges = BuildEdges();

        private static int[][] BuildEdges()
        {
            var e = new List<int[]>();
            for (int i = 0; i < 8; i++)
                foreach (int bit in new[] { 1, 2, 4 })
                    if ((i & bit) == 0) e.Add(new[] { i, i | bit });
            return e.ToArray();
        }

        public static double[] Sub(double[] a, double[] b) { return new[] { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; }
        public static double[] Add(double[] a, double[] b) { return new[] { a[0] + b[0], a[1] + b[1], a[2] + b[2] }; }
        public static double[] Scale(double[] a, double s) { return new[] { a[0] * s, a[1] * s, a[2] * s }; }
        public static double Dot(double[] a, double[] b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
        public static double[] Cross(double[] a, double[] b)
        {
            return new[] { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
        }
        public static double Norm(double[] a) { return Math.Sqrt(Dot(a, a)); }

        /// <summary>The rotation matrix of a quaternion [x, y, z, w], normalized first.</summary>
        public static double[][] RotationMatrix(double[] q)
        {
            double x = q[0], y = q[1], z = q[2], w = q[3];
            double n = Math.Sqrt(x * x + y * y + z * z + w * w);
            x = x / n; y = y / n; z = z / n; w = w / n;
            return new[]
            {
                new[] { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w) },
                new[] { 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
                new[] { 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y) }
            };
        }

        /// <summary>The counter-clockwise convex hull (monotone chain), without repeated or collinear points. Repeats
        /// are dropped as Python's set drops them (the first of equal points stays); the rest sort as tuples.</summary>
        public static List<double[]> ConvexHull(IList<double[]> points)
        {
            var pts = new List<double[]>();
            foreach (var p in points)
            {
                bool seen = false;
                foreach (var q in pts) if (q[0] == p[0] && q[1] == p[1]) { seen = true; break; }
                if (!seen) pts.Add(p);
            }
            pts.Sort((a, b) => a[0] == b[0] ? (a[1] == b[1] ? 0 : (a[1] < b[1] ? -1 : 1)) : (a[0] < b[0] ? -1 : 1));
            if (pts.Count <= 2) return pts;
            var lower = new List<double[]>();
            var upper = new List<double[]>();
            foreach (var p in pts)
            {
                while (lower.Count >= 2 && Turn(lower[lower.Count - 2], lower[lower.Count - 1], p) <= EPS) lower.RemoveAt(lower.Count - 1);
                lower.Add(p);
            }
            for (int i = pts.Count - 1; i >= 0; i--)
            {
                var p = pts[i];
                while (upper.Count >= 2 && Turn(upper[upper.Count - 2], upper[upper.Count - 1], p) <= EPS) upper.RemoveAt(upper.Count - 1);
                upper.Add(p);
            }
            var hull = new List<double[]>();
            for (int i = 0; i < lower.Count - 1; i++) hull.Add(lower[i]);
            for (int i = 0; i < upper.Count - 1; i++) hull.Add(upper[i]);
            return hull;
        }

        private static double Turn(double[] o, double[] a, double[] b)
        {
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
        }

        public static double PolygonArea(IList<double[]> poly)
        {
            if (poly.Count < 3) return 0.0;
            double s = 0.0;
            for (int i = 0; i < poly.Count; i++)
            {
                double x1 = poly[i][0], y1 = poly[i][1];
                double x2 = poly[(i + 1) % poly.Count][0], y2 = poly[(i + 1) % poly.Count][1];
                s += x1 * y2 - x2 * y1;
            }
            return Math.Abs(s) / 2.0;
        }

        /// <summary>Sutherland-Hodgman: the part of a polygon inside a convex counter-clockwise polygon.</summary>
        public static List<double[]> ClipPolygon(IList<double[]> subject, IList<double[]> clip)
        {
            var output = new List<double[]>(subject);
            for (int i = 0; i < clip.Count; i++)
            {
                if (output.Count == 0) break;
                double ax = clip[i][0], ay = clip[i][1];
                double bx = clip[(i + 1) % clip.Count][0], by = clip[(i + 1) % clip.Count][1];
                double ex = bx - ax, ey = by - ay;
                Func<double[], bool> inside = p => ex * (p[1] - ay) - ey * (p[0] - ax) >= -EPS;
                Func<double[], double[], double[]> crossing = (p, q) =>
                {
                    double dx = q[0] - p[0], dy = q[1] - p[1];
                    double denom = dx * ey - dy * ex;
                    if (Math.Abs(denom) < 1e-18) return q;
                    double t = ((ax - p[0]) * ey - (ay - p[1]) * ex) / denom;
                    return new[] { p[0] + t * dx, p[1] + t * dy };
                };
                var points = output;
                output = new List<double[]>();
                var prev = points[points.Count - 1];
                foreach (var cur in points)
                {
                    if (inside(cur))
                    {
                        if (!inside(prev)) output.Add(crossing(prev, cur));
                        output.Add(cur);
                    }
                    else if (inside(prev))
                    {
                        output.Add(crossing(prev, cur));
                    }
                    prev = cur;
                }
            }
            return output;
        }

        public static double IntersectionArea(IList<double[]> a, IList<double[]> b)
        {
            if (a.Count < 3 || b.Count < 3) return 0.0;
            return PolygonArea(ClipPolygon(a, b));
        }

        /// <summary>The separating-axis test with the 15 candidate axes; touching boxes count as intersecting.</summary>
        public static bool BoxesIntersect(Box a, Box b)
        {
            var t = Sub(b.Centre, a.Centre);
            var axes = new List<double[]>(a.Axes);
            axes.AddRange(b.Axes);
            foreach (var u in a.Axes)
                foreach (var v in b.Axes)
                {
                    var c = Cross(u, v);
                    if (Norm(c) > EPS) axes.Add(c);
                }
            foreach (var axis in axes)
            {
                double ra = Py.Sum(new[] { a.Half[0] * Math.Abs(Dot(a.Axes[0], axis)), a.Half[1] * Math.Abs(Dot(a.Axes[1], axis)),
                                           a.Half[2] * Math.Abs(Dot(a.Axes[2], axis)) });
                double rb = Py.Sum(new[] { b.Half[0] * Math.Abs(Dot(b.Axes[0], axis)), b.Half[1] * Math.Abs(Dot(b.Axes[1], axis)),
                                           b.Half[2] * Math.Abs(Dot(b.Axes[2], axis)) });
                if (Math.Abs(Dot(t, axis)) > ra + rb) return false;
            }
            return true;
        }

        public static double PointBoxDistance(double[] p, Box box)
        {
            var loc = box.Local(p);
            double d2 = 0.0;
            for (int i = 0; i < 3; i++)
            {
                double excess = Math.Abs(loc[i]) - box.Half[i];
                if (excess > 0.0) d2 += excess * excess;
            }
            return Math.Sqrt(d2);
        }

        private static double Clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

        /// <summary>The shortest distance between segments p1-q1 and p2-q2 (Ericson 5.1.9).</summary>
        public static double SegmentDistance(double[] p1, double[] q1, double[] p2, double[] q2)
        {
            var d1 = Sub(q1, p1); var d2 = Sub(q2, p2); var r = Sub(p1, p2);
            double a = Dot(d1, d1), e = Dot(d2, d2), f = Dot(d2, r);
            double s, t;
            if (a <= EPS && e <= EPS) { s = 0.0; t = 0.0; }
            else if (a <= EPS) { s = 0.0; t = Clamp01(f / e); }
            else
            {
                double c = Dot(d1, r);
                if (e <= EPS) { t = 0.0; s = Clamp01(-c / a); }
                else
                {
                    double b = Dot(d1, d2);
                    double denom = a * e - b * b;
                    s = denom > EPS ? Clamp01((b * f - c * e) / denom) : 0.0;
                    t = (b * s + f) / e;
                    if (t < 0.0) { t = 0.0; s = Clamp01(-c / a); }
                    else if (t > 1.0) { t = 1.0; s = Clamp01((b - c) / a); }
                }
            }
            return Norm(Sub(Add(p1, Scale(d1, s)), Add(p2, Scale(d2, t))));
        }

        /// <summary>The exact shortest distance between two boxes; 0 if they intersect or touch.</summary>
        public static double BoxDistance(Box a, Box b)
        {
            if (BoxesIntersect(a, b)) return 0.0;
            var first = new List<double>();
            foreach (var c in a.Corners) first.Add(PointBoxDistance(c, b));
            double best = Py.Min(first);
            var second = new List<double>();
            foreach (var c in b.Corners) second.Add(PointBoxDistance(c, a));
            best = Py.Min(best, Py.Min(second));
            foreach (var ij in Edges)
                foreach (var km in Edges)
                    best = Py.Min(best, SegmentDistance(a.Corners[ij[0]], a.Corners[ij[1]], b.Corners[km[0]], b.Corners[km[1]]));
            return best;
        }

        /// <summary>The footprint intersection over the smaller footprint's area; null if a footprint is degenerate.</summary>
        public static double? FootprintOverlap(Box t, Box a)
        {
            double smaller = Py.Min(t.FootprintArea, a.FootprintArea);
            if (smaller < EPS) return null;
            return IntersectionArea(t.Footprint, a.Footprint) / smaller;
        }

        public static double VerticalOverlap(Box t, Box a)
        {
            return Py.Min(t.ZMax, a.ZMax) - Py.Max(t.ZMin, a.ZMin);
        }

        public static double ContainmentExcess(Box target, Box container)
        {
            double worst = double.NegativeInfinity;
            foreach (var c in target.Corners)
            {
                var loc = container.Local(c);
                for (int i = 0; i < 3; i++) worst = Py.Max(worst, Math.Abs(loc[i]) - container.Half[i]);
            }
            return worst;
        }
    }

    /// <summary>An oriented box with its derived geometry, computed once.</summary>
    public sealed class Box
    {
        public readonly double[] Centre, Half;
        public readonly double[][] Axes, Corners;
        public readonly double ZMin, ZMax, FootprintArea, TiltDeg;
        public readonly List<double[]> Footprint;

        public Box(double[] centre, double[] size, double[] quaternion)
        {
            Centre = new[] { centre[0], centre[1], centre[2] };
            Half = new[] { size[0] / 2.0, size[1] / 2.0, size[2] / 2.0 };
            var r = Geo.RotationMatrix(quaternion);
            Axes = new double[3][];
            for (int i = 0; i < 3; i++) Axes[i] = new[] { r[0][i], r[1][i], r[2][i] };
            Corners = new double[8][];
            for (int i = 0; i < 8; i++)
            {
                var local = new double[3];
                for (int k = 0; k < 3; k++) local[k] = (i & (4 >> k)) != 0 ? Half[k] : -Half[k];
                var offset = new[]
                {
                    r[0][0] * local[0] + r[0][1] * local[1] + r[0][2] * local[2],
                    r[1][0] * local[0] + r[1][1] * local[1] + r[1][2] * local[2],
                    r[2][0] * local[0] + r[2][1] * local[1] + r[2][2] * local[2]
                };
                Corners[i] = Geo.Add(Centre, offset);
            }
            var zs = new List<double>();
            foreach (var c in Corners) zs.Add(c[2]);
            ZMin = Py.Min(zs);
            ZMax = Py.Max(zs);
            var flat = new List<double[]>();
            foreach (var c in Corners) flat.Add(new[] { c[0], c[1] });
            Footprint = Geo.ConvexHull(flat);
            FootprintArea = Geo.PolygonArea(Footprint);
            double up = Axes[2][2];
            TiltDeg = Py.Degrees(Math.Acos(Py.Max(-1.0, Py.Min(1.0, up))));
        }

        /// <summary>A scene point in this box's local coordinates.</summary>
        public double[] Local(double[] p)
        {
            var d = Geo.Sub(p, Centre);
            return new[] { Geo.Dot(d, Axes[0]), Geo.Dot(d, Axes[1]), Geo.Dot(d, Axes[2]) };
        }
    }
}
