using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SecondEyes.Perception
{
    /// <summary>
    /// One detection: an index into the package's class list, its score, and its box in source-image pixels with the
    /// origin at the top-left corner and y pointing down, as the detector package defines them (A1.10, D84).
    /// </summary>
    [Serializable]
    public struct Detection
    {
        public int ClassId;
        public float Score;
        public float X1, Y1, X2, Y2;
    }

    /// <summary>What one inference produced and what it cost. Reused between inferences: read it before the next.</summary>
    public sealed class DetectionResult
    {
        public readonly List<Detection> Detections = new List<Detection>();
        public int SourceWidth, SourceHeight;
        public double Ratio;
        public int Candidates;
        public double ScheduleMs, LatencyMs, PostprocessMs;
        public int FramesWaited;
        public int StepsPerFrame, ScheduleSteps, ScheduleFrames;
        public float Floor;
        public bool Blocking;
        public bool Completed;
        public string Error;

        public void Clear()
        {
            Detections.Clear();
            SourceWidth = SourceHeight = Candidates = FramesWaited = StepsPerFrame = ScheduleSteps = ScheduleFrames = 0;
            Floor = 0f;
            Ratio = ScheduleMs = LatencyMs = PostprocessMs = 0;
            Completed = Blocking = false;
            Error = null;
        }
    }

    /// <summary>
    /// A replaceable object detector (A1.10, D84). Callers see only this interface and a detector package (a model
    /// file and its manifest), so a larger model or another family can replace this one without changes elsewhere.
    /// Both detection methods are coroutines: run them with StartCoroutine or yield them from a coroutine, then read
    /// the result they filled in. One inference runs at a time.
    /// </summary>
    public interface IObjectDetector : IDisposable
    {
        string PackageId { get; }

        /// <summary>"gpu_compute" or "cpu".</summary>
        string Backend { get; }

        /// <summary>The model's square input size in pixels.</summary>
        int InputSize { get; }

        IReadOnlyList<string> Classes { get; }

        /// <summary>The model's layer count after import: the number of scheduling steps one inference takes.</summary>
        int LayerCount { get; }

        /// <summary>Scheduling steps (layers) dispatched per rendered frame; 0 dispatches the whole network in one frame.
        /// Applies from the next inference (A1.10, D88).</summary>
        int StepsPerFrame { get; set; }

        /// <summary>Lowest score kept after suppression. The package's threshold still defines a detection; scores
        /// between the floor and the threshold are kept only so parity can verify threshold crossings (D88).</summary>
        float Floor { get; set; }

        /// <summary>Letterboxes a texture into the model's input on the GPU, then detects.</summary>
        /// <param name="flipVertically">True when the texture's rows run bottom to top.</param>
        /// <param name="snapshotPath">If not null, the letterboxed input is also saved there as a PNG.</param>
        /// <param name="blocking">True: read the outputs back synchronously in the same frame (a hitch, but no
        /// asynchronous step); false: read them back asynchronously over the following frames.</param>
        IEnumerator DetectTexture(Texture source, bool flipVertically, string snapshotPath, bool blocking, DetectionResult into);

        /// <summary>Detects on an exact InputSize x InputSize canvas given as RGBA pixels, rows from the top. This is the
        /// parity path: the PC reference reads the same pixels.</summary>
        IEnumerator DetectCanvas(Color32[] topDownPixels, bool blocking, DetectionResult into);
    }
}
