// A4: 3D boxes for the automatic scan (docs/auto-scan.md), in SpatialLM's form: a centre, a turn about the vertical and
// three side lengths. Every object of the scan is one: Meta's boxes as Space Setup made them, and the detector's objects
// fitted to the points their depth samples hit, their hidden side completed to the category's size prior. Free of Unity,
// so the PC's replay fits the same boxes.
using System;
using System.Collections.Generic;
using System.Linq;
using SecondEyes.Perception.Frames;

namespace SecondEyes.Perception.AutoScan
{
    /// <summary>An upright box in quest_tracking (Unity axes, y up): its centre, its turn about +y in degrees (Unity's
    /// sense: +90 takes +x to -z), and its side lengths along its own x, y (up) and z.</summary>
    public sealed class Box3
    {
        public Vec3 Centre, Size;
        public double YawDeg;
        /// <summary>A side was grown to the category's size prior, away from where the object was seen.</summary>
        public bool Completed;

        public double Volume => Size.X * Size.Y * Size.Z;
        public Vec3 Top => new Vec3(Centre.X, Centre.Y + Size.Y / 2, Centre.Z);

        /// <summary>p in the box's own axes, relative to its centre.</summary>
        public Vec3 Local(Vec3 p)
        {
            double a = YawDeg * Math.PI / 180, c = Math.Cos(a), s = Math.Sin(a);
            double dx = p.X - Centre.X, dz = p.Z - Centre.Z;
            return new Vec3(dx * c - dz * s, p.Y - Centre.Y, dx * s + dz * c);
        }

        /// <summary>A point in the box's own axes, relative to its centre, in quest_tracking.</summary>
        public Vec3 World(Vec3 l)
        {
            double a = YawDeg * Math.PI / 180, c = Math.Cos(a), s = Math.Sin(a);
            return new Vec3(Centre.X + l.X * c + l.Z * s, Centre.Y + l.Y, Centre.Z - l.X * s + l.Z * c);
        }

        /// <summary>How far p lies outside the box; a point above its top counts as inside up to topAllowance higher
        /// (something resting on it, or a pile on a bed).</summary>
        public double Outside(Vec3 p, double topAllowance)
        {
            Vec3 l = Local(p);
            double dx = Math.Max(0, Math.Abs(l.X) - Size.X / 2), dz = Math.Max(0, Math.Abs(l.Z) - Size.Z / 2);
            double dy = Math.Max(0, Math.Abs(l.Y) - Size.Y / 2);
            if (l.Y > 0) dy = Math.Max(0, dy - topAllowance);
            return Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        /// <summary>The turn as a quaternion [x, y, z, w] in quest_tracking.</summary>
        public double[] Rotation()
        {
            double h = YawDeg * Math.PI / 360;
            return new[] { 0, Math.Sin(h), 0, Math.Cos(h) };
        }
    }

    public static class Boxes
    {
        /// <summary>The value at fraction f of the sorted values (0 the smallest, 1 the largest), the nearer index.</summary>
        private static double At(double[] sorted, double f)
        {
            int i = (int)Math.Round(f * (sorted.Length - 1), MidpointRounding.AwayFromZero);
            return sorted[Math.Max(0, Math.Min(sorted.Length - 1, i))];
        }

        private const double CloseFloorM = 0.01;   // closeness counts a point at most this close to an edge

        /// <summary>How closely the points hug the edges of the footprint at this turn (the closeness criterion of
        /// L-shape fitting, as used for vehicles in laser scans): each point scores 1 over its distance to the nearest
        /// edge, floored at 1 cm. The footprint's sides drop the trim fraction of points at either end.</summary>
        private static double Closeness(List<Vec3> pts, double yawDeg, double trim, out double uLo, out double uHi, out double wLo, out double wHi)
        {
            double a = yawDeg * Math.PI / 180, c = Math.Cos(a), s = Math.Sin(a);
            var u = new double[pts.Count];
            var w = new double[pts.Count];
            for (int i = 0; i < pts.Count; i++) { u[i] = pts[i].X * c - pts[i].Z * s; w[i] = pts[i].X * s + pts[i].Z * c; }
            var su = (double[])u.Clone();
            var sw = (double[])w.Clone();
            Array.Sort(su);
            Array.Sort(sw);
            uLo = At(su, trim); uHi = At(su, 1 - trim); wLo = At(sw, trim); wHi = At(sw, 1 - trim);
            double score = 0;
            for (int i = 0; i < pts.Count; i++)
            {
                double du = Math.Min(Math.Abs(u[i] - uLo), Math.Abs(uHi - u[i])), dw = Math.Min(Math.Abs(w[i] - wLo), Math.Abs(wHi - w[i]));
                score += 1 / Math.Max(Math.Min(du, dw), CloseFloorM);
            }
            return score;
        }

        /// <summary>The upright box around the points (quest_tracking): the turn in [0, 90) degrees at which the points
        /// hug the footprint's edges most closely (10-degree steps, then 1-degree steps around the best; ties to the
        /// smaller turn). Closeness, not the smallest area: a camera mostly sees an object's front and one side, an L of
        /// points, whose smallest rectangle is ambiguous (aligned with the object or with the L's diagonal, the same
        /// area). Each side drops the trim fraction of points at either end, so a stray ray doesn't stretch it. A box
        /// standing on the floor reaches down to floorY. Every side is at least 2 cm.</summary>
        public static Box3 Fit(List<Vec3> pts, double trim, bool onFloor, double floorY)
        {
            double best = 0, bestScore = double.MinValue, uLo, uHi, wLo, wHi;
            for (int d = 0; d < 90; d += 10)
            {
                double score = Closeness(pts, d, trim, out uLo, out uHi, out wLo, out wHi);
                if (score > bestScore * (1 + 1e-12) + 1e-12) { bestScore = score; best = d; }
            }
            double coarse = best;
            for (int d = -9; d <= 9; d++)
            {
                double yaw = ((coarse + d) % 90 + 90) % 90;
                double score = Closeness(pts, yaw, trim, out uLo, out uHi, out wLo, out wHi);
                if (score > bestScore * (1 + 1e-12) + 1e-12 || (Math.Abs(score - bestScore) <= 1e-12 * Math.Max(1, bestScore) && yaw < best)) { bestScore = score; best = yaw; }
            }
            Closeness(pts, best, trim, out uLo, out uHi, out wLo, out wHi);
            double[] ys = pts.Select(p => p.Y).OrderBy(y => y).ToArray();
            double top = At(ys, 1 - trim), bottom = onFloor ? floorY : At(ys, trim);
            if (top < bottom + 0.02) top = bottom + 0.02;
            double su = Math.Max(uHi - uLo, 0.02), sw = Math.Max(wHi - wLo, 0.02);
            Vec3 mid = new Box3 { YawDeg = best, Centre = new Vec3(0, 0, 0) }.World(new Vec3((uLo + uHi) / 2, 0, (wLo + wHi) / 2));
            return new Box3 { YawDeg = best, Size = new Vec3(su, top - bottom, sw), Centre = new Vec3(mid.X, (top + bottom) / 2, mid.Z) };
        }

        /// <summary>The box brought to the size prior closest to it (priors as [width, depth, height]): each side shorter
        /// than the prior's grows away from the mean of the viewpoints that saw it, since the hidden side is the far one;
        /// the height grows upward from the bottom. A side longer than maxOver times the prior's (a small thing's samples
        /// landing on the desk around it) shrinks to the prior's, centred on the median of the points, given in the
        /// box's own axes; a height so tall keeps its bottom.</summary>
        public static Box3 Complete(Box3 b, IList<double[]> priors, IList<Vec3> viewpoints, double maxOver, Vec3 medianLocal)
        {
            if (priors == null || priors.Count == 0) return b;
            double big = Math.Max(b.Size.X, b.Size.Z), small = Math.Min(b.Size.X, b.Size.Z);
            double[] p = priors.OrderBy(q =>
            {
                double pa = Math.Max(q[0], q[1]), pb = Math.Min(q[0], q[1]);
                return (pa - big) * (pa - big) + (pb - small) * (pb - small) + (q[2] - b.Size.Y) * (q[2] - b.Size.Y);
            }).First();
            double pBig = Math.Max(p[0], p[1]), pSmall = Math.Min(p[0], p[1]);
            double tx = b.Size.X >= b.Size.Z ? pBig : pSmall, tz = b.Size.X >= b.Size.Z ? pSmall : pBig;
            Vec3 view = new Vec3(viewpoints.Average(v => v.X), viewpoints.Average(v => v.Y), viewpoints.Average(v => v.Z));
            Vec3 toView = b.Local(view);
            double sx = b.Size.X, sy = b.Size.Y, sz = b.Size.Z, lx = 0, lz = 0, bottom = b.Centre.Y - b.Size.Y / 2;
            bool grew = false;
            if (sx < tx) { lx = -(toView.X >= 0 ? 1 : -1) * (tx - sx) / 2; sx = tx; grew = true; }
            else if (sx > maxOver * tx) { lx = medianLocal.X; sx = tx; grew = true; }
            if (sz < tz) { lz = -(toView.Z >= 0 ? 1 : -1) * (tz - sz) / 2; sz = tz; grew = true; }
            else if (sz > maxOver * tz) { lz = medianLocal.Z; sz = tz; grew = true; }
            if (sy < p[2] || sy > maxOver * p[2]) { sy = p[2]; grew = true; }
            if (!grew) return b;
            Vec3 c = b.World(new Vec3(lx, 0, lz));
            return new Box3 { YawDeg = b.YawDeg, Size = new Vec3(sx, sy, sz), Centre = new Vec3(c.X, bottom + sy / 2, c.Z), Completed = true };
        }

        /// <summary>One of Meta's boxes in the same upright form, whichever of its own axes points up (Space Setup's
        /// volumes stand on their z axis). The turn is brought into [0, 90) by swapping the horizontal sides.</summary>
        public static Box3 Upright(RoomBox m)
        {
            Vec3[] axes = { RoomModel.Rotate(m.Rotation, new Vec3(1, 0, 0)), RoomModel.Rotate(m.Rotation, new Vec3(0, 1, 0)), RoomModel.Rotate(m.Rotation, new Vec3(0, 0, 1)) };
            double[] sides = { m.Size.X, m.Size.Y, m.Size.Z };
            int up = Enumerable.Range(0, 3).OrderByDescending(k => Math.Abs(axes[k].Y)).First();
            int h1 = up == 0 ? 1 : 0, h2 = up == 2 ? 1 : 2;
            Vec3 ax = axes[h1];
            double yaw = Math.Atan2(-ax.Z, ax.X) * 180 / Math.PI;
            double sx = sides[h1], sz = sides[h2];
            yaw = ((yaw % 360) + 360) % 360;
            while (yaw >= 90) { yaw -= 90; double t = sx; sx = sz; sz = t; }
            if (yaw > 89.9999) yaw = 0;
            return new Box3 { Centre = m.Centre, Size = new Vec3(sx, sides[up], sz), YawDeg = yaw };
        }

        /// <summary>Points kept one per voxel of the given size, in the order they came; past max points the voxel
        /// doubles and the points are kept again. Deterministic, so the replay keeps the same points.</summary>
        public static double Thin(List<Vec3> points, double voxel, int max)
        {
            while (true)
            {
                var seen = new HashSet<(long, long, long)>();
                var kept = new List<Vec3>();
                foreach (Vec3 p in points)
                    if (seen.Add(((long)Math.Floor(p.X / voxel), (long)Math.Floor(p.Y / voxel), (long)Math.Floor(p.Z / voxel)))) kept.Add(p);
                points.Clear();
                points.AddRange(kept);
                if (points.Count <= max) return voxel;
                voxel *= 2;
            }
        }
    }
}
