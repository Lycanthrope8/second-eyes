// A4: Meta's room for the automatic scan (docs/auto-scan.md): the boxes Space Setup puts around obstacles, the floor's
// outline, and the floor's and ceiling's heights, in quest_tracking (Unity axes: y up). Free of Unity, so the PC's replay
// reads the same room from frames.jsonl and places every sighting exactly as the headset did.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Perception.Frames;
using SecondEyes.Perception.Localization;

namespace SecondEyes.Perception.AutoScan
{
    /// <summary>One of Meta's boxes (a Space Setup volume), in quest_tracking.</summary>
    public sealed class RoomBox
    {
        /// <summary>Meta's anchor ID, and Meta's label for it ("BED", "TABLE", "OTHER" and so on).</summary>
        public string Id, MetaLabel;
        /// <summary>The box's centre in quest_tracking, and its full side lengths along its own x, y and z axes.</summary>
        public Vec3 Centre, Size;
        /// <summary>[x, y, z, w]: from the box's own axes to quest_tracking's.</summary>
        public double[] Rotation = { 0, 0, 0, 1 };

        public double Volume => Size.X * Size.Y * Size.Z;

        /// <summary>How far p lies outside the box, measured along the box's own axes; 0 inside it.</summary>
        public double Outside(Vec3 p)
        {
            Vec3 local = RoomModel.Rotate(RoomModel.Conjugate(Rotation), p - Centre);
            double dx = Math.Max(0, Math.Abs(local.X) - Size.X / 2), dy = Math.Max(0, Math.Abs(local.Y) - Size.Y / 2),
                   dz = Math.Max(0, Math.Abs(local.Z) - Size.Z / 2);
            return Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        /// <summary>The middle of the box's top face, in quest_tracking: where a label floats over it.</summary>
        public Vec3 Top => Centre + RoomModel.Rotate(Rotation, new Vec3(0, Size.Y / 2, 0));

        public JNode ToJson()
        {
            return JNode.Obj().Add("id", JNode.Str(Id)).Add("meta_label", JNode.Str(MetaLabel)).Add("centre", Localizer.Vec(Centre))
                .Add("size", Localizer.Vec(Size)).Add("rotation_xyzw", JNode.Arr(Rotation.Select(x => Localizer.Num(x)).ToList()));
        }

        public static RoomBox FromJson(JNode n)
        {
            return new RoomBox { Id = n["id"].Text, MetaLabel = n["meta_label"].Text, Centre = RoomModel.V(n["centre"]), Size = RoomModel.V(n["size"]),
                                 Rotation = n["rotation_xyzw"].Items.Select(x => x.Num).ToArray() };
        }
    }

    /// <summary>Meta's room: its boxes, its floor's outline, its floor's and ceiling's heights. Not loaded: no room is used.</summary>
    public sealed class RoomModel
    {
        public const string RecordType = "auto_scan_room";

        /// <summary>"loaded", or "none" with why (Space Setup declined or failed, no room found, or space_setup never).</summary>
        public string Status = "none", Why = "";
        public readonly List<RoomBox> Boxes = new List<RoomBox>();
        /// <summary>The floor's outline, each corner (x, z) in quest_tracking.</summary>
        public readonly List<double[]> FloorXz = new List<double[]>();
        public double FloorY, CeilingY;

        public bool Loaded => Status == "loaded";

        /// <summary>Whether p is in the room, within margin: between the floor and the ceiling, and inside the floor's
        /// outline or within margin of it. With no room, or an outline of fewer than three corners, everything is in it.</summary>
        public bool InRoom(Vec3 p, double margin)
        {
            if (!Loaded) return true;
            if (p.Y < FloorY - margin || p.Y > CeilingY + margin) return false;
            if (FloorXz.Count < 3) return true;
            return InOutline(p.X, p.Z) || EdgeDistance(p.X, p.Z) <= margin;
        }

        /// <summary>The smallest of the boxes p is in, within margin; ties go to the box whose ID sorts first. None: null.</summary>
        public RoomBox BoxAt(Vec3 p, double margin)
        {
            if (!Loaded) return null;
            return Boxes.Where(b => b.Outside(p) <= margin).OrderBy(b => b.Volume).ThenBy(b => b.Id, StringComparer.Ordinal).FirstOrDefault();
        }

        private bool InOutline(double x, double z)
        {
            bool inside = false;
            for (int i = 0, j = FloorXz.Count - 1; i < FloorXz.Count; j = i++)
            {
                double xi = FloorXz[i][0], zi = FloorXz[i][1], xj = FloorXz[j][0], zj = FloorXz[j][1];
                if ((zi > z) != (zj > z) && x < (xj - xi) * (z - zi) / (zj - zi) + xi) inside = !inside;
            }
            return inside;
        }

        private double EdgeDistance(double x, double z)
        {
            double best = double.MaxValue;
            for (int i = 0, j = FloorXz.Count - 1; i < FloorXz.Count; j = i++)
            {
                double ax = FloorXz[j][0], az = FloorXz[j][1], bx = FloorXz[i][0], bz = FloorXz[i][1];
                double dx = bx - ax, dz = bz - az, len2 = dx * dx + dz * dz;
                double t = len2 > 0 ? Math.Max(0, Math.Min(1, ((x - ax) * dx + (z - az) * dz) / len2)) : 0;
                double ex = ax + t * dx - x, ez = az + t * dz - z;
                best = Math.Min(best, Math.Sqrt(ex * ex + ez * ez));
            }
            return best;
        }

        /// <summary>v rotated by the unit quaternion q = [x, y, z, w].</summary>
        public static Vec3 Rotate(double[] q, Vec3 v)
        {
            double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
            double tx = 2 * (qy * v.Z - qz * v.Y), ty = 2 * (qz * v.X - qx * v.Z), tz = 2 * (qx * v.Y - qy * v.X);
            return new Vec3(v.X + qw * tx + (qy * tz - qz * ty), v.Y + qw * ty + (qz * tx - qx * tz), v.Z + qw * tz + (qx * ty - qy * tx));
        }

        public static double[] Conjugate(double[] q) { return new[] { -q[0], -q[1], -q[2], q[3] }; }

        /// <summary>A rotation in quest_tracking (Unity axes) re-expressed in the scene's axes (z up, right-handed): the
        /// basis change swaps y and z, a reflection, so the axis is swapped too and the angle reverses.</summary>
        public static double[] RotationToScene(double[] q) { return new[] { -q[0], -q[2], -q[1], q[3] }; }

        internal static Vec3 V(JNode n) { return new Vec3(n.Items[0].Num, n.Items[1].Num, n.Items[2].Num); }

        public JNode ToJson()
        {
            return JNode.Obj().Add("record_type", JNode.Str(RecordType)).Add("status", JNode.Str(Status)).Add("why", JNode.Str(Why))
                .Add("floor_y", Localizer.Num(FloorY)).Add("ceiling_y", Localizer.Num(CeilingY))
                .Add("floor_xz", JNode.Arr(FloorXz.Select(c => JNode.Arr(new List<JNode> { Localizer.Num(c[0]), Localizer.Num(c[1]) })).ToList()))
                .Add("boxes", JNode.Arr(Boxes.Select(b => b.ToJson()).ToList()));
        }

        public static RoomModel FromJson(JNode n)
        {
            var r = new RoomModel { Status = n["status"].Text, Why = n["why"].Text, FloorY = n["floor_y"].Num, CeilingY = n["ceiling_y"].Num };
            foreach (JNode c in n["floor_xz"].Items) r.FloorXz.Add(new[] { c.Items[0].Num, c.Items[1].Num });
            foreach (JNode b in n["boxes"].Items) r.Boxes.Add(RoomBox.FromJson(b));
            return r;
        }

        /// <summary>No room, and why.</summary>
        public static RoomModel None(string why) { return new RoomModel { Status = "none", Why = why }; }
    }
}
