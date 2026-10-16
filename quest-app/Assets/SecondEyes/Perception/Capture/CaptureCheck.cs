using System;
using System.Collections.Generic;
using System.Globalization;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;

namespace SecondEyes.Perception.Capture
{
    /// <summary>
    /// The capture record's check (A4 Delivery 1; docs/capture-record.md), Unity-free, run on the headset where the record
    /// is written: the record type and version, its JSON Schema (schemas/capture.v1.json), finite numbers, then the rules
    /// a schema can't state. A layer runs only if the one before it passed, as in the scene contract's check, and issues
    /// use the contract's Issue with its codes where a rule is shared (E_RECORD_TYPE, E_SCHEMA_*, E_NONFINITE, E_QUAT_NORM).
    /// </summary>
    public static class CaptureCheck
    {
        public const string SchemaFile = "capture.v1.json";
        public const double Tolerance = 1e-6;   // the contract's tolerance for a unit quaternion (D66)

        public static List<Issue> Check(string name, JNode record, Schema schema)
        {
            var issues = new List<Issue>();
            Action<string, string, string> add = (path, code, message) => issues.Add(new Issue { File = name, Path = path, Code = code, Message = message });
            JNode rt = record != null ? Contract.Get(record, "record_type") : JNode.Null;
            JNode ver = record != null ? Contract.Get(record, "schema_version") : JNode.Null;
            if (record == null || record.Kind != JKind.Object || rt.Kind != JKind.String || rt.Text != "capture"
                || ver.Kind != JKind.Number || !ver.IsInteger || ver.Num != 1)
            {
                add("$", "E_RECORD_TYPE", "not a capture record of schema_version 1");
                return issues;
            }
            foreach (var e in schema.Errors(record)) add(e.Path, Contract.CodeForKeyword(e.Keyword), e.Keyword);
            if (issues.Count > 0) return issues;
            Numbers(record, "$", add);
            if (issues.Count > 0) return issues;

            JNode camera = record["camera"], intr = camera["intrinsics"];
            double width = camera["width_px"].Num, height = camera["height_px"].Num;
            if (intr["cx"].Num < 0 || intr["cx"].Num > width || intr["cy"].Num < 0 || intr["cy"].Num > height)
                add("$.camera.intrinsics", "E_CAPTURE_INTRINSICS", "the principal point is outside the image");

            string unit = record["clock"]["sdk_unit"].Text;
            long offset;
            bool offsetKnown = Whole(record["clock"]["offset_us"], out offset);
            List<JNode> frames = record["keyframes"].Items;
            double epoch = frames[0]["tracking"]["epoch"].Num;
            var frameIds = new HashSet<string>();
            for (int i = 0; i < frames.Count; i++)
            {
                JNode k = frames[i];
                string b = "$.keyframes[" + i + "]";
                if (k["index"].Num != i) add(b + ".index", "E_CAPTURE_ORDER", "keyframes are numbered 0, 1, 2... in order");
                if (!frameIds.Add(k["frame_id"].Text)) add(b + ".frame_id", "E_CAPTURE_FRAME_ID", "a frame ID is used once in a capture");
                if (i > 0 && k["capture_time"]["headset_mono_us"].Num <= frames[i - 1]["capture_time"]["headset_mono_us"].Num)
                    add(b + ".capture_time.headset_mono_us", "E_CAPTURE_ORDER", "capture times rise strictly");
                long sdk, mono;
                if (offsetKnown && Whole(k["capture_time"]["sdk_value"], out sdk) && Whole(k["capture_time"]["headset_mono_us"], out mono)
                    && mono != Microseconds(sdk, unit) + offset)
                    add(b + ".capture_time.headset_mono_us", "E_CAPTURE_TIME", "not the SDK value in microseconds (rounded down) plus clock.offset_us");
                double pos = k["position_index"].Num, step = k["step_index"].Num;
                double prevPos = i == 0 ? -1 : frames[i - 1]["position_index"].Num, prevStep = i == 0 ? -1 : frames[i - 1]["step_index"].Num;
                if (pos != prevPos && pos != prevPos + 1)
                    add(b + ".position_index", "E_CAPTURE_ORDER", "positions start at 0 and go up one at a time");
                else if (pos == prevPos ? step != prevStep + 1 : step != 0)
                    add(b + ".step_index", "E_CAPTURE_ORDER", "steps start at 0 at each position and go up one at a time");
                if (k["tracking"]["epoch"].Num != epoch)
                    add(b + ".tracking.epoch", "E_CAPTURE_EPOCH", "every keyframe of a scan is in its first keyframe's tracking epoch");
                if (k["tracking"]["state"].Text != "tracked")
                    add(b + ".tracking.state", "E_CAPTURE_NOT_TRACKED", "a capture keeps only tracked keyframes: without tracking the pose isn't known");
                double n = 0;
                foreach (JNode x in k["camera_pose"]["rotation_xyzw"].Items) n += x.Num * x.Num;
                if (Math.Abs(Math.Sqrt(n) - 1.0) > Tolerance)
                    add(b + ".camera_pose.rotation_xyzw", "E_QUAT_NORM", "norm is not 1 within the tolerance");
                List<JNode> dets = k["detections"].Items;
                for (int j = 0; j < dets.Count; j++)
                {
                    List<JNode> box = dets[j]["box_px"].Items;
                    double x1 = box[0].Num, y1 = box[1].Num, x2 = box[2].Num, y2 = box[3].Num;
                    // Boxes aren't clipped (detector.boxes), so one may run past the edge; it must still be a box on the image.
                    if (!(x1 < x2 && y1 < y2 && x2 > 0 && x1 < width && y2 > 0 && y1 < height))
                        add(b + ".detections[" + j + "].box_px", "E_CAPTURE_BOX", "a box has x1 < x2 and y1 < y2 and overlaps the image");
                }
            }
            return issues;
        }

        /// <summary>A whole number read exactly from its source text: a nanosecond count can exceed what a double holds exactly.</summary>
        private static bool Whole(JNode n, out long value)
        {
            value = 0;
            return n.Kind == JKind.Number && n.IsInteger
                   && long.TryParse(n.Text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out value);
        }

        /// <summary>An SDK value in microseconds, rounded down (values are non-negative, by the schema).</summary>
        public static long Microseconds(long value, string unit)
        {
            switch (unit)
            {
                case "ns": return value / 1000;
                case "us": return value;
                case "ms": return value * 1000;
                default: return value / 10;   // "100ns"
            }
        }

        private static void Numbers(JNode v, string path, Action<string, string, string> add)
        {
            switch (v.Kind)
            {
                case JKind.Number:
                    if (double.IsInfinity(v.Num) || double.IsNaN(v.Num)) add(path, "E_NONFINITE", "number is too large for a double");
                    break;
                case JKind.Object:
                    foreach (var f in v.Fields) Numbers(f.Value, path + "." + f.Key, add);
                    break;
                case JKind.Array:
                    for (int i = 0; i < v.Items.Count; i++) Numbers(v.Items[i], path + "[" + i + "]", add);
                    break;
            }
        }
    }
}
