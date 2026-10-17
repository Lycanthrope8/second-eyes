using System;
using System.Collections.Generic;
using System.Text;

namespace SecondEyes.Perception.Scan
{
    /// <summary>
    /// The scan's guidance in view (A4 Delivery 1, stage 3; docs/scan.md), free of Unity: the lines the head-locked
    /// overlay shows while scanning, made from the keyframe rule's state. A big hint first, then the spot and keyframe
    /// counts, a bar of the directions already taken at this spot (one slot per step, counted from the spot's first
    /// keyframe), and how to finish.
    /// </summary>
    public static class ScanGuide
    {
        public static string Text(KeyframeRule rule, int kept, int failed)
        {
            var sb = new StringBuilder();
            sb.Append("SCAN: ").Append(Hint(rule)).Append('\n');
            int spot = Math.Max(rule.Position, 0) + 1;
            sb.Append("spot ").Append(spot).Append(" of up to ").Append(rule.MaxPositions).Append(", ").Append(kept)
              .Append(kept == 1 ? " keyframe" : " keyframes").Append(failed > 0 ? " (" + failed + " left out)" : "").Append('\n');
            if (rule.HasRows)
            {
                sb.Append("level ").Append(Directions(rule)).Append('\n');
                sb.Append("down  ").Append(Directions(rule, true)).Append('\n');
                sb.Append("Look down at desks and tables too; skip walls right beside you\n");
            }
            else sb.Append(Directions(rule)).Append("  directions done here\n");
            sb.Append("Done? Right thumbstick brings the panel: press Finish");
            return sb.ToString();
        }

        /// <summary>The rule's hint, as words for someone turning around a room.</summary>
        public static string Hint(KeyframeRule rule)
        {
            string h = rule.Hint ?? "";
            if (h.StartsWith("tracking lost", StringComparison.Ordinal)) return "tracking lost: look around slowly";
            if (h.StartsWith("hold still: the last keyframe", StringComparison.Ordinal)) return "hold still, reading the last view";
            if (h == "hold still") return rule.Keyframes == 0 ? "face part of the room and hold still" : "hold still";
            if (h == "keyframe") return "got it";
            if (h.StartsWith("the scan is complete", StringComparison.Ordinal)) return "complete: press Finish on the panel";
            if (h.StartsWith("move about", StringComparison.Ordinal)) return "walk 2 to 3 m to another side of the room, then hold still";
            if (h.StartsWith("turn about", StringComparison.Ordinal))
                return "turn about " + rule.StepDeg.ToString("0", System.Globalization.CultureInfo.InvariantCulture)
                       + " degrees and hold still" + (rule.SpotYaws.Count >= Slots(rule) / 2 ? "; or walk 2 to 3 m to a new spot" : "");
            return h;
        }

        /// <summary>One slot per step around the spot, counted from the spot's first keyframe: # where a keyframe was taken
        /// (in the look-down row, when down), . where none yet.</summary>
        public static string Directions(KeyframeRule rule, bool down = false)
        {
            int n = Slots(rule);
            var done = new bool[n];
            IReadOnlyList<double> row = down ? rule.SpotDownYaws : rule.SpotYaws;
            if (row.Count > 0)
            {
                double first = rule.SpotFirstYaw;
                foreach (double y in row)
                {
                    int k = (int)Math.Round(KeyframeRule.Wrap(y - first) / rule.StepDeg);
                    done[((k % n) + n) % n] = true;
                }
            }
            var sb = new StringBuilder("[");
            foreach (bool d in done) sb.Append(d ? '#' : '.');
            return sb.Append(']').ToString();
        }

        private static int Slots(KeyframeRule rule) { return Math.Max(1, (int)Math.Round(360.0 / rule.StepDeg)); }
    }
}
