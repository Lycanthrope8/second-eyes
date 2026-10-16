using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Perception.Capture
{
    /// <summary>The camera as a capture record states it: OpenCV intrinsics in the texture's pixels (docs/frames.md).</summary>
    public sealed class CaptureCamera
    {
        public string CameraId, IntrinsicsSource, Distortion;
        public int WidthPx, HeightPx;
        public double Fx, Fy, Cx, Cy;
    }

    /// <summary>One detection as the detector reported it: a box in source pixels, not clipped.</summary>
    public sealed class CaptureDetection
    {
        public string Label;
        public double Score, X1, Y1, X2, Y2;
    }

    /// <summary>
    /// Builds a capture record (schemas/capture.v1.json; docs/capture-record.md) keyframe by keyframe, Unity-free, so the
    /// headset and the PC's checks build records the same way. The detector block is copied from the package's manifest,
    /// never restated, and a package whose boxes are anything but source pixels, not clipped, is refused. Each keyframe's
    /// headset_mono_us is derived from its SDK timestamp and the scan's one clock offset, so the capture check's clock rule
    /// holds by construction. Keyframes are numbered in the order added, with frame IDs kf-0000, kf-0001, and so on.
    /// </summary>
    public sealed class CaptureBuilder
    {
        private readonly JNode head;
        private readonly string sdkUnit;
        private readonly long offsetUs;
        private readonly List<JNode> frames = new List<JNode>();

        public CaptureBuilder(string captureId, CaptureCamera camera, string appBuild, JNode detectorManifest, string packageSha256,
                              string perceptionFolderSha256, string categoryMapId, string sizePriorsId, string sdkUnit, long offsetUs,
                              double nominalStepDeg)
        {
            JNode input = detectorManifest["input"], post = detectorManifest["postprocess"];
            string coordinates = post["coordinates"].Text ?? "";
            if (!coordinates.Contains("source-image pixels") || !coordinates.Contains("not clipped"))
                throw new ArgumentException("the detector's boxes are not source-image pixels, not clipped: " + coordinates);
            this.sdkUnit = sdkUnit;
            this.offsetUs = offsetUs;
            head = JNode.Obj()
                .Add("schema_version", I(1)).Add("record_type", S("capture")).Add("capture_id", S(captureId))
                .Add("camera", JNode.Obj().Add("camera_id", S(camera.CameraId)).Add("width_px", I(camera.WidthPx)).Add("height_px", I(camera.HeightPx))
                    .Add("intrinsics", JNode.Obj().Add("fx", R(camera.Fx)).Add("fy", R(camera.Fy)).Add("cx", R(camera.Cx)).Add("cy", R(camera.Cy))
                        .Add("source", S(camera.IntrinsicsSource)))
                    .Add("distortion", S(camera.Distortion)))
                .Add("app", JNode.Obj().Add("build", S(appBuild)))
                .Add("detector", JNode.Obj().Add("package_id", S(detectorManifest["package_id"].Text)).Add("package_sha256", S(packageSha256))
                    .Add("input_px", I((long)input["size"].Num)).Add("letterbox", S(input["letterbox"].Text))
                    .Add("score_threshold", R(post["score_threshold"].Num)).Add("nms", S(post["nms"].Text))
                    .Add("nms_iou_threshold", R(post["nms_iou_threshold"].Num)).Add("boxes", S("source_pixels_unclipped")))
                .Add("configuration", JNode.Obj().Add("perception_folder_sha256", S(perceptionFolderSha256))
                    .Add("category_map_id", S(categoryMapId)).Add("size_priors_id", S(sizePriorsId)))
                .Add("clock", JNode.Obj().Add("sdk_unit", S(sdkUnit)).Add("offset_us", I(offsetUs)))
                .Add("tracking_frame", S("quest_tracking")).Add("camera_pose_convention", S("unity_camera_to_quest_tracking"))
                .Add("nominal_step_deg", R(nominalStepDeg));
        }

        public int Count => frames.Count;

        /// <summary>Adds one keyframe: its place in the scan, its SDK timestamp, the tracking epoch and state, the camera's
        /// pose in quest_tracking exactly as Unity reports it, its detections, and its image (null when not saved).</summary>
        public void Add(int positionIndex, int stepIndex, long sdkValue, long trackingEpoch, bool tracked, double[] translationM,
                        double[] rotationXyzw, IList<CaptureDetection> detections, string imageFile = null, string imageSha256 = null)
        {
            int index = frames.Count;
            JNode image = imageFile == null ? JNode.Obj().Add("state", S("not_saved"))
                                            : JNode.Obj().Add("state", S("saved")).Add("file", S(imageFile)).Add("sha256", S(imageSha256));
            frames.Add(JNode.Obj()
                .Add("index", I(index)).Add("frame_id", S("kf-" + index.ToString("D4", CultureInfo.InvariantCulture)))
                .Add("position_index", I(positionIndex)).Add("step_index", I(stepIndex))
                .Add("capture_time", JNode.Obj().Add("sdk_value", I(sdkValue))
                    .Add("headset_mono_us", I(CaptureCheck.Microseconds(sdkValue, sdkUnit) + offsetUs)))
                .Add("tracking", JNode.Obj().Add("epoch", I(trackingEpoch)).Add("state", S(tracked ? "tracked" : "not_tracked")))
                .Add("camera_pose", JNode.Obj().Add("translation_m", JNode.Arr(translationM.Select(R).ToList()))
                    .Add("rotation_xyzw", JNode.Arr(rotationXyzw.Select(R).ToList())))
                .Add("detections", JNode.Arr(detections.Select(d => JNode.Obj().Add("label", S(d.Label)).Add("score", R(d.Score))
                    .Add("box_px", JNode.Arr(new List<JNode> { R(d.X1), R(d.Y1), R(d.X2), R(d.Y2) }))).ToList()))
                .Add("image", image));
        }

        /// <summary>The record so far: the head and every keyframe added.</summary>
        public JNode Build()
        {
            var o = JNode.Obj();
            foreach (var f in head.Fields) o.Add(f.Key, f.Value);
            return o.Add("keyframes", JNode.Arr(new List<JNode>(frames)));
        }

        private static JNode S(string s) { return JNode.Str(s); }
        private static JNode I(long v) { return new JNode { Kind = JKind.Number, Num = v, Text = v.ToString(CultureInfo.InvariantCulture), IsInteger = true }; }
        private static JNode R(double v)   // the shortest text that reads back as exactly this double
        {
            return new JNode { Kind = JKind.Number, Num = v, Text = v.ToString("R", CultureInfo.InvariantCulture), IsInteger = false };
        }
    }
}
