using System;
using System.Collections.Generic;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Perception.Frames;

namespace SecondEyes.Perception.Scan
{
    /// <summary>The automatic scan's settings (schemas/scan-config.v1.json; docs/scan.md).</summary>
    public sealed class ScanSettings
    {
        public string ConfigId;
        public double StepDeg, StepMarginDeg, SteadyS, SteadyDeg, SteadyM, NewPositionM;
        public int MaxKeyframesPerPosition, MaxPositions;

        public static ScanSettings From(JNode n)
        {
            return new ScanSettings
            {
                ConfigId = n["config_id"].Text, StepDeg = n["step_deg"].Num, StepMarginDeg = n["step_margin_deg"].Num,
                SteadyS = n["steady_s"].Num, SteadyDeg = n["steady_deg"].Num, SteadyM = n["steady_m"].Num,
                NewPositionM = n["new_position_m"].Num, MaxKeyframesPerPosition = (int)n["max_keyframes_per_position"].Num,
                MaxPositions = (int)n["max_positions"].Num
            };
        }
    }

    /// <summary>
    /// When the automatic scan takes a keyframe (A4 Delivery 1, stage 3; docs/scan.md), Unity-free. Fed the head's pose
    /// once per rendered frame (position in quest_tracking, y up; yaw in degrees, 0 facing +z, positive turning right), it
    /// says when to take one:
    /// - only while tracking holds, no inference is running, and the head has held still for steady_s: over that time its
    ///   yaw varied by at most steady_deg (largest minus smallest) and its position stayed within steady_m of where it
    ///   is now, so the image is sharp and the pose settled;
    /// - the first as soon as the head is still;
    /// - at the same spot, when the head faces a direction at least step_deg - step_margin_deg from every keyframe already
    ///   taken there, so turning back adds nothing;
    /// - at a new spot, when the head has moved new_position_m or more across the floor from the spot's first keyframe;
    ///   that keyframe starts the next position.
    /// The caller takes the keyframe, then calls Taken(); a keyframe that couldn't be taken leaves the rule as it was.
    /// </summary>
    public sealed class KeyframeRule
    {
        private struct Sample { public double T, Yaw; public Vec3 P; }

        private readonly ScanSettings s;
        private readonly LinkedList<Sample> window = new LinkedList<Sample>();
        private readonly List<double> yaws = new List<double>();   // the keyframes' yaws at the current position
        private Vec3 origin;
        private bool pending, pendingNewPosition;
        private double pendingYaw;
        private Vec3 pendingAt;

        public KeyframeRule(ScanSettings settings) { s = settings; }

        public int Position { get; private set; } = -1;
        public int Step { get; private set; } = -1;
        public int Keyframes { get; private set; }
        public bool Full => Position + 1 >= s.MaxPositions && Step + 1 >= s.MaxKeyframesPerPosition;
        public string Hint { get; private set; } = "hold still";
        /// <summary>The yaws of the keyframes taken at the current position, in degrees, in the order taken.</summary>
        public IReadOnlyList<double> SpotYaws => yaws;
        public double StepDeg => s.StepDeg;
        public int MaxPositions => s.MaxPositions;
        /// <summary>After Update asked for a keyframe: the position it will have once taken; -1 when none is due.</summary>
        public int PendingPosition => pending ? (pendingNewPosition ? Position + 1 : Position) : -1;

        /// <summary>One head sample; true when a keyframe should be taken now.</summary>
        public bool Update(double t, Vec3 head, double yawDeg, bool tracked, bool busy)
        {
            pending = false;
            if (!tracked)
            {
                window.Clear();
                Hint = "tracking lost: look around until it returns";
                return false;
            }
            window.AddLast(new Sample { T = t, Yaw = yawDeg, P = head });
            while (window.Count > 2 && t - window.First.Next.Value.T >= s.SteadyS) window.RemoveFirst();
            if (!Steady(t, head, yawDeg)) { Hint = "hold still"; return false; }

            bool newPosition;
            if (Position < 0) newPosition = true;
            else if (Across(head, origin) >= s.NewPositionM) newPosition = Position + 1 < s.MaxPositions;
            else newPosition = false;
            bool newDirection = !newPosition && Position >= 0 && Step + 1 < s.MaxKeyframesPerPosition
                                && Nearest(yawDeg) >= s.StepDeg - s.StepMarginDeg;
            if (!newPosition && !newDirection)
            {
                Hint = Position + 1 >= s.MaxPositions && (Step + 1 >= s.MaxKeyframesPerPosition || Across(head, origin) >= s.NewPositionM)
                    ? "the scan is complete: press Finish"
                    : Step + 1 >= s.MaxKeyframesPerPosition ? "move about " + s.NewPositionM.ToString("0.#", System.Globalization.CultureInfo.InvariantCulture) + " m to a new spot and hold still"
                    : "turn about " + s.StepDeg.ToString("0", System.Globalization.CultureInfo.InvariantCulture) + "° and hold still, or move to a new spot";
                return false;
            }
            if (busy) { Hint = "hold still: the last keyframe is still being read"; return false; }
            pending = true;
            pendingNewPosition = newPosition;
            pendingYaw = yawDeg;
            pendingAt = head;
            Hint = "keyframe";
            return true;
        }

        /// <summary>Records the keyframe that Update just asked for.</summary>
        public void Taken()
        {
            if (!pending) throw new InvalidOperationException("no keyframe is due");
            if (pendingNewPosition)
            {
                Position++;
                Step = 0;
                origin = pendingAt;
                yaws.Clear();
            }
            else Step++;
            yaws.Add(pendingYaw);
            Keyframes++;
            pending = false;
        }

        private bool Steady(double t, Vec3 head, double yaw)
        {
            if (t - window.First.Value.T < s.SteadyS) return false;
            double low = 0, high = 0;   // the yaws relative to the newest, so the range is measured across the +-180 seam
            foreach (Sample x in window)
            {
                double d = Wrap(x.Yaw - yaw);
                low = Math.Min(low, d); high = Math.Max(high, d);
                if ((x.P - head).Norm > s.SteadyM) return false;
            }
            return high - low <= s.SteadyDeg;
        }

        private double Nearest(double yaw)
        {
            double best = double.MaxValue;
            foreach (double y in yaws) best = Math.Min(best, Math.Abs(Wrap(yaw - y)));
            return best;
        }

        private static double Across(Vec3 a, Vec3 b) { return Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Z - b.Z) * (a.Z - b.Z)); }

        /// <summary>An angle in degrees brought into (-180, 180].</summary>
        public static double Wrap(double deg)
        {
            double r = deg % 360.0;
            if (r > 180) r -= 360;
            if (r <= -180) r += 360;
            return r;
        }
    }
}
