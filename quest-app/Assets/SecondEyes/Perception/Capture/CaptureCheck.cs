using System;
using System.Collections.Generic;
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

            List<JNode> frames = record["keyframes"].Items;
            double epoch = frames[0]["tracking"]["epoch"].Num;
            for (int i = 0; i < frames.Count; i++)
            {
                JNode k = frames[i];
                string b = "$.keyframes[" + i + "]";
                if (k["index"].Num != i) add(b + ".index", "E_CAPTURE_ORDER", "keyframes are numbered 0, 1, 2... in order");
                if (i > 0 && k["capture_time"]["headset_mono_us"].Num <= frames[i - 1]["capture_time"]["headset_mono_us"].Num)
                    add(b + ".capture_time.headset_mono_us", "E_CAPTURE_ORDER", "capture times rise strictly");
                double pos = k["position_index"].Num, step = k["step_index"].Num;
                double prevPos = i == 0 ? -1 : frames[i - 1]["position_index"].Num, prevStep = i == 0 ? -1 : frames[i - 1]["step_index"].Num;
                if (pos != prevPos && pos != prevPos + 1)
                    add(b + ".position_index", "E_CAPTURE_ORDER", "positions start at 0 and go up one at a time");
                else if (pos == prevPos ? step != prevStep + 1 : step != 0)
                    add(b + ".step_index", "E_CAPTURE_ORDER", "steps start at 0 at each position and go up one at a time");
                if (k["tracking"]["epoch"].Num != epoch)
                    add(b + ".tracking.epoch", "E_CAPTURE_EPOCH", "every keyframe of a scan is in its first keyframe's tracking epoch");
                if (k["tracking"]["state"].Text != "tracked" && k["detections"].Items.Count > 0)
                    add(b + ".detections", "E_CAPTURE_NOT_TRACKED", "a keyframe without tracking has no known pose, so it carries no detections");
                double n = 0;
                foreach (JNode x in k["camera_pose"]["rotation_xyzw"].Items) n += x.Num * x.Num;
                if (Math.Abs(Math.Sqrt(n) - 1.0) > Tolerance)
                    add(b + ".camera_pose.rotation_xyzw", "E_QUAT_NORM", "norm is not 1 within the tolerance");
                List<JNode> dets = k["detections"].Items;
                for (int j = 0; j < dets.Count; j++)
                {
                    List<JNode> box = dets[j]["box_px"].Items;
                    double x1 = box[0].Num, y1 = box[1].Num, x2 = box[2].Num, y2 = box[3].Num;
                    if (!(0 <= x1 && x1 < x2 && x2 <= width && 0 <= y1 && y1 < y2 && y2 <= height))
                        add(b + ".detections[" + j + "].box_px", "E_CAPTURE_BOX", "a box is x1 < x2 and y1 < y2, inside the image");
                }
            }
            return issues;
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
