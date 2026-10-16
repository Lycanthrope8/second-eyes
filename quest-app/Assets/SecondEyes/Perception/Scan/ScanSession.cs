using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
using SecondEyes.Perception.Capture;
using SecondEyes.Perception.Frames;
using SecondEyes.Perception.Localization;

namespace SecondEyes.Perception.Scan
{
    /// <summary>A camera timestamp as the capture record states it: a whole number in one unit (docs/capture-record.md).</summary>
    public static class SdkStamp
    {
        /// <summary>A DateTime or TimeSpan becomes its 100 ns ticks; a whole number is read as nanoseconds; a floating number
        /// as seconds, converted to microseconds. False for anything else, or a negative value.</summary>
        public static bool TryConvert(object raw, out long value, out string unit, out string kind)
        {
            value = 0; unit = null; kind = raw == null ? "none" : raw.GetType().Name;
            switch (raw)
            {
                case DateTime d: value = d.Ticks; unit = "100ns"; return true;
                case TimeSpan t: value = t.Ticks; unit = "100ns"; return value >= 0;
                case long l: value = l; unit = "ns"; return l >= 0;
                case ulong u: if (u > long.MaxValue) return false; value = (long)u; unit = "ns"; return true;
                case int i: value = i; unit = "ns"; return i >= 0;
                case uint ui: value = ui; unit = "ns"; return true;
                case double x: if (double.IsNaN(x) || double.IsInfinity(x) || x < 0) return false; value = (long)Math.Round(x * 1e6); unit = "us"; return true;
                case float x: if (float.IsNaN(x) || float.IsInfinity(x) || x < 0) return false; value = (long)Math.Round(x * 1e6); unit = "us"; return true;
                default: return false;
            }
        }
    }

    /// <summary>One detection as the detector returned it, before its class becomes a label.</summary>
    public sealed class RawDetection { public int ClassId; public double Score, X1, Y1, X2, Y2; }

    /// <summary>What one scan keyframe's inference produced: the frame's own timestamp and pose, read as its image was copied.</summary>
    public sealed class ScanFrame
    {
        public bool Completed;
        public string Error, Snapshot;
        public object RawStamp;
        public long CopyMonoUs, TrackingEpoch;
        public bool Tracked;
        public double[] TranslationM, RotationXyzw;   // the camera's pose in quest_tracking, as Unity reports it
        public List<RawDetection> Detections = new List<RawDetection>();
    }

    /// <summary>A finished scan: the capture record, and when it passes its check, the perceived scene and the trace.</summary>
    public sealed class ScanResult
    {
        public string Error;
        public JNode Capture, Scene, Trace, Facts;
        public List<Issue> CaptureIssues = new List<Issue>(), SceneIssues = new List<Issue>();
        public bool Ok => Error == null && CaptureIssues.Count == 0 && Scene != null && !SceneIssues.Any(i => i.IsError);
    }

    /// <summary>
    /// One automatic scan on the headset (A4 Delivery 1, stage 3; docs/scan.md), free of Unity so every rule is checked on
    /// the PC. The scanner feeds it the head's pose each frame (Update), starts a keyframe when it says so (Started), and
    /// hands back each keyframe's result (Add); Finish checks the capture, runs the localization and checks the scene.
    /// - Positions and steps are numbered over the keyframes kept, so a failed inference leaves no gap in the record.
    /// - The SDK timestamp becomes a whole number and unit (SdkStamp); the clock offset is set at the first keyframe kept,
    ///   taking that frame's age as zero, and every later frame's age against it is kept as a fact.
    /// - A keyframe not tracked at capture is left out; a keyframe in another tracking epoch, or with a timestamp of
    ///   another kind, ends the scan with an error, since the capture would fail its check.
    /// </summary>
    public sealed class ScanSession
    {
        public readonly string CaptureId;
        public readonly KeyframeRule Rule;
        public readonly List<JNode> KeyframeFacts = new List<JNode>();
        public string Error { get; private set; }
        public int Kept => builder == null ? 0 : builder.Count;
        public int Failed { get; private set; }

        private readonly PerceptionFolder folder;
        private readonly JNode manifest;
        private readonly string packageSha256, appBuild;
        private CaptureCamera camera;
        private JNode cameraFacts = JNode.Null;
        private CaptureBuilder builder;
        private string unit, stampKind;
        private long offsetUs, epoch;
        private int lastRulePosition = -1, position = -1, step = -1;

        public ScanSession(string captureId, PerceptionFolder folder, JNode detectorManifest, string packageSha256, string appBuild)
        {
            CaptureId = captureId; this.folder = folder; manifest = detectorManifest; this.packageSha256 = packageSha256; this.appBuild = appBuild;
            Rule = new KeyframeRule(ScanSettings.From(folder.ScanConfig));
        }

        public bool Update(double t, Vec3 head, double yawDeg, bool tracked, bool busy)
        {
            return Error == null && Rule.Update(t, head, yawDeg, tracked, busy);
        }

        /// <summary>The keyframe the scanner just started; returns the rule's position for it.</summary>
        public int Started() { Rule.Taken(); return Rule.Position; }

        /// <summary>The camera as found when the scan began: intrinsics already in the texture's pixels, and how they were read.</summary>
        public void SetCamera(CaptureCamera c, JNode facts) { if (camera == null) { camera = c; cameraFacts = facts ?? JNode.Null; } }

        /// <summary>One keyframe's result, started at the rule's position rulePosition. False when it isn't kept.</summary>
        public bool Add(int rulePosition, ScanFrame f)
        {
            if (Error != null) return false;
            JNode fact = JNode.Obj().Add("rule_position", Localizer.Int(rulePosition)).Add("snapshot", f.Snapshot == null ? JNode.Null : JNode.Str(f.Snapshot));
            if (!f.Completed || !f.Tracked || f.TranslationM == null || f.RotationXyzw == null)
            {
                Failed++;
                KeyframeFacts.Add(fact.Add("kept", Flag(false)).Add("why", JNode.Str(!f.Completed ? "inference: " + (f.Error ?? "not completed")
                    : !f.Tracked ? "not tracked at capture" : "no pose")));
                return false;
            }
            long value; string u, kind;
            if (!SdkStamp.TryConvert(f.RawStamp, out value, out u, out kind)) { Error = "the camera's timestamp (" + kind + ") isn't a time the capture can record"; return false; }
            if (camera == null) { Error = "the camera's intrinsics weren't read before the first keyframe"; return false; }
            if (builder == null)
            {
                unit = u; stampKind = kind; epoch = f.TrackingEpoch;
                offsetUs = f.CopyMonoUs - CaptureCheck.Microseconds(value, unit);
                builder = new CaptureBuilder(CaptureId, camera, appBuild, manifest, packageSha256, folder.ManifestSha256, folder.CategoryMapId,
                                             folder.SizePriorsId, unit, offsetUs, ScanSettings.From(folder.ScanConfig).StepDeg);
            }
            else if (u != unit) { Error = "the camera's timestamp changed kind mid-scan (" + stampKind + " to " + kind + ")"; return false; }
            if (f.TrackingEpoch != epoch) { Error = "tracking was reset during the scan: start it again"; return false; }
            if (rulePosition != lastRulePosition) { position++; step = 0; lastRulePosition = rulePosition; } else step++;
            List<JNode> classes = manifest["classes"].Items;
            var dets = f.Detections.Where(d => d.ClassId >= 0 && d.ClassId < classes.Count)
                .Select(d => new CaptureDetection { Label = classes[d.ClassId].Text, Score = d.Score, X1 = d.X1, Y1 = d.Y1, X2 = d.X2, Y2 = d.Y2 }).ToList();
            builder.Add(position, step, value, f.TrackingEpoch, true, f.TranslationM, f.RotationXyzw, dets);
            long derived = CaptureCheck.Microseconds(value, unit) + offsetUs;
            KeyframeFacts.Add(fact.Add("kept", Flag(true)).Add("index", Localizer.Int(builder.Count - 1)).Add("position", Localizer.Int(position))
                .Add("step", Localizer.Int(step)).Add("stamp_kind", JNode.Str(kind))
                .Add("stamp_text", JNode.Str(Convert.ToString(f.RawStamp, CultureInfo.InvariantCulture)))
                .Add("sdk_value", Localizer.Int(value)).Add("copy_mono_us", Localizer.Int(f.CopyMonoUs)).Add("derived_mono_us", Localizer.Int(derived))
                .Add("frame_age_us", Localizer.Int(f.CopyMonoUs - derived)).Add("detections", Localizer.Int(dets.Count)));
            return true;
        }

        private static JNode Flag(bool b) { return new JNode { Kind = JKind.Bool, Bool = b }; }

        public ScanResult Finish()
        {
            var r = new ScanResult();
            r.Facts = JNode.Obj().Add("capture_id", JNode.Str(CaptureId)).Add("camera", cameraFacts)
                .Add("stamp_kind", stampKind == null ? JNode.Null : JNode.Str(stampKind)).Add("sdk_unit", unit == null ? JNode.Null : JNode.Str(unit))
                .Add("offset_us", builder == null ? JNode.Null : Localizer.Int(offsetUs)).Add("perception_folder_sha256", JNode.Str(folder.ManifestSha256))
                .Add("kept", Localizer.Int(Kept)).Add("failed", Localizer.Int(Failed)).Add("keyframes", JNode.Arr(new List<JNode>(KeyframeFacts)));
            if (Error != null) { r.Error = Error; return r; }
            if (Kept == 0) { r.Error = "no keyframe was kept"; return r; }
            r.Capture = builder.Build();
            r.CaptureIssues = CaptureCheck.Check(CaptureId, r.Capture, folder.Schema(CaptureCheck.SchemaFile));
            if (r.CaptureIssues.Count > 0) return r;
            LocalizationOutput o = Localizer.Run(r.Capture, folder.CategoryMap, folder.SizePriors, folder.LocalizationConfig);
            r.Scene = o.Scene; r.Trace = o.Trace;
            r.SceneIssues = new Contract(folder.Schema).Validate(new List<KeyValuePair<string, JNode>> {
                new KeyValuePair<string, JNode>("scene", o.Scene), new KeyValuePair<string, JNode>("category_map", folder.CategoryMap) });
            return r;
        }
    }
}
