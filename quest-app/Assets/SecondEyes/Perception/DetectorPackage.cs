using System;
using UnityEngine;

namespace SecondEyes.Perception
{
    /// <summary>
    /// The manifest of a detector package (detector-package.json beside the model; A1.10b, D84), read with JsonUtility.
    /// Only the fields the headset needs are declared here; the rest (source, build, outputs) stay in the file.
    /// </summary>
    [Serializable]
    public class DetectorPackage
    {
        public int format_version;
        public string record_type;
        public string package_id;
        public string model_file;
        public string model_sha256;
        public InputSpec input;
        public PostprocessSpec postprocess;
        public string[] classes;
        public CategorySpec[] project_categories;
        public string license;

        [Serializable]
        public class InputSpec
        {
            public string name;
            public int size;
            public int[] shape;
            public string channels;
            public float[] range;
            public string letterbox;
            public int pad_value;
        }

        [Serializable]
        public class PostprocessSpec
        {
            public float score_threshold;
            public string nms;
            public float nms_iou_threshold;
            public string nms_overlap;
        }

        [Serializable]
        public class CategorySpec
        {
            public string @class;
            public string category;
        }

        /// <summary>The package, or null with the reason when the manifest is missing, malformed or describes a
        /// decoder this app does not implement.</summary>
        public static DetectorPackage Parse(TextAsset manifest, out string problem)
        {
            problem = null;
            if (manifest == null)
            {
                problem = "no detector-package manifest is assigned";
                return null;
            }
            DetectorPackage p;
            try
            {
                p = JsonUtility.FromJson<DetectorPackage>(manifest.text);
            }
            catch (Exception e)
            {
                problem = "the manifest is not valid JSON: " + e.Message;
                return null;
            }
            if (p == null || p.format_version != 1 || p.record_type != "detector_package")
                problem = "not a version 1 detector_package manifest";
            else if (p.input == null || p.input.size <= 0 || p.input.channels != "RGB" || p.input.letterbox != "top_left"
                     || p.input.shape == null || p.input.shape.Length != 4 || p.input.shape[0] != 1 || p.input.shape[1] != 3
                     || p.input.shape[2] != p.input.size || p.input.shape[3] != p.input.size
                     || p.input.range == null || p.input.range.Length != 2 || p.input.range[0] != 0f || p.input.range[1] != 1f
                     || p.input.pad_value < 0 || p.input.pad_value > 255)
                problem = "the manifest's input is not an RGB [0, 1] square, top-left letterboxed tensor";
            else if (p.postprocess == null || p.postprocess.nms != "class_agnostic_greedy" || p.postprocess.nms_overlap != "yolox_plus_one"
                     || !(p.postprocess.score_threshold > 0f && p.postprocess.score_threshold < 1f)
                     || !(p.postprocess.nms_iou_threshold > 0f && p.postprocess.nms_iou_threshold < 1f))
                problem = "the manifest's post-processing is not one this app implements";
            else if (p.classes == null || p.classes.Length == 0)
                problem = "the manifest lists no classes";
            return problem == null ? p : null;
        }
    }
}
