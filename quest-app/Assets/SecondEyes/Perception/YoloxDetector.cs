using System;
using System.Collections;
using System.Collections.Generic;
using Stopwatch = System.Diagnostics.Stopwatch;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using Object = UnityEngine.Object;

namespace SecondEyes.Perception
{
    /// <summary>
    /// The YOLOX detector package on Unity's Inference Engine (A1.10c, D84). The wrapped model takes an RGB [0, 1]
    /// tensor and returns boxes (input pixels), scores and class indices; this class letterboxes frames on the GPU,
    /// runs the model, reads the three outputs back asynchronously and applies the package's post-processing:
    /// candidates at or above the score threshold, YOLOX's greedy class-agnostic suppression (its "+1" overlap
    /// convention), then input pixels divided by the letterbox ratio. The PC reference (perception/detector) applies
    /// the same rules.
    /// </summary>
    public sealed class YoloxDetector : IObjectDetector
    {
        private static readonly int RegionId = Shader.PropertyToID("_Region");
        private static readonly int EncodeId = Shader.PropertyToID("_EncodeSRGB");
        private static readonly int FlipId = Shader.PropertyToID("_FlipSource");

        private readonly DetectorPackage package;
        private readonly BackendType backendType;
        private readonly int size;
        private Worker worker;
        private Tensor<float> input;
        private RenderTexture canvas;
        private Material letterbox;
        private Texture2D readback;
        private bool running, abandoned;

        /// <summary>Frames a readback may take before the inference is abandoned and reported (about 8 s at 72 Hz).</summary>
        public int MaxWaitFrames = 600;
        private readonly List<int> candidates = new List<int>();
        private float[] x1 = new float[0], y1 = new float[0], x2 = new float[0], y2 = new float[0], area = new float[0];
        private bool[] suppressed = new bool[0];

        public string PackageId => package.package_id;
        public string Backend => backendType == BackendType.GPUCompute ? "gpu_compute" : "cpu";
        public int InputSize => size;
        public IReadOnlyList<string> Classes => package.classes;
        public int LayerCount { get; private set; }
        public int StepsPerFrame { get; set; }
        public int[] Slices { get; set; }
        public IReadOnlyList<string> LayerTypes => layerTypes;
        private readonly List<string> layerTypes = new List<string>();
        public float Floor { get; set; }

        public YoloxDetector(DetectorPackage package, ModelAsset modelAsset, Shader letterboxShader, BackendType backend)
        {
            if (package == null || modelAsset == null || letterboxShader == null)
                throw new ArgumentException("the package, the model asset and the letterbox shader are all required");
            this.package = package;
            backendType = backend;
            size = package.input.size;
            var model = ModelLoader.Load(modelAsset);
            var shape = model.inputs[0].shape;
            if (model.inputs.Count != 1 || shape.Get(0) != 1 || shape.Get(1) != 3 || shape.Get(2) != size || shape.Get(3) != size)
                throw new ArgumentException("the model's input is not [1, 3, " + size + ", " + size + "] as the manifest says");
            LayerCount = model.layers.Count;
            foreach (var layer in model.layers) layerTypes.Add(layer.GetType().Name);
            Floor = package.postprocess.score_threshold;
            worker = new Worker(model, backend);
            input = new Tensor<float>(new TensorShape(1, 3, size, size));
            canvas = new RenderTexture(size, size, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = "DetectorCanvas"
            };
            canvas.Create();
            letterbox = new Material(letterboxShader) { name = "DetectorLetterbox" };
            readback = new Texture2D(size, size, TextureFormat.RGBA32, false, true);
        }

        public IEnumerator DetectTexture(Texture source, bool flipVertically, string snapshotPath, bool blocking, DetectionResult into)
        {
            into.Clear();
            if (running)
            {
                into.Error = "busy";
                yield break;
            }
            if (source == null)
            {
                into.Error = "no source texture";
                yield break;
            }
            running = true;
            try
            {
                var clock = Stopwatch.StartNew();
                into.StartFrame = Time.frameCount;
                int w = source.width, h = source.height;
                double ratio = Math.Min((double)size / h, (double)size / w);
                int rw = (int)(w * ratio), rh = (int)(h * ratio);
                bool encode = QualitySettings.activeColorSpace == ColorSpace.Linear
                              && GraphicsFormatUtility.IsSRGBFormat(source.graphicsFormat);
                letterbox.SetVector(RegionId, new Vector4((float)rw / size, (float)rh / size, package.input.pad_value / 255f, 0f));
                letterbox.SetFloat(EncodeId, encode ? 1f : 0f);
                letterbox.SetFloat(FlipId, flipVertically ? 1f : 0f);
                Graphics.Blit(source, canvas, letterbox);
                if (snapshotPath != null) SaveCanvas(snapshotPath);
                var transform = new TextureTransform().SetDimensions(size, size, 3).SetCoordOrigin(CoordOrigin.TopLeft);
                TextureConverter.ToTensor(canvas, input, transform);
                into.PreprocessMs = clock.Elapsed.TotalMilliseconds;
                into.SourceWidth = w;
                into.SourceHeight = h;
                into.Ratio = ratio;
                yield return Schedule(input, into, clock);
                yield return Collect(into, clock, ratio, blocking);
            }
            finally
            {
                running = false;
            }
        }

        public IEnumerator DetectCanvas(Color32[] topDownPixels, bool blocking, DetectionResult into)
        {
            into.Clear();
            if (running)
            {
                into.Error = "busy";
                yield break;
            }
            int plane = size * size;
            if (topDownPixels == null || topDownPixels.Length != plane)
            {
                into.Error = "the canvas is not " + size + " x " + size + " pixels";
                yield break;
            }
            running = true;
            try
            {
                var clock = Stopwatch.StartNew();
                into.StartFrame = Time.frameCount;
                var data = new float[3 * plane];
                for (int k = 0; k < plane; k++)
                {
                    var p = topDownPixels[k];
                    data[k] = p.r / 255f;
                    data[plane + k] = p.g / 255f;
                    data[2 * plane + k] = p.b / 255f;
                }
                using var exact = new Tensor<float>(new TensorShape(1, 3, size, size), data);
                into.PreprocessMs = clock.Elapsed.TotalMilliseconds;
                into.SourceWidth = into.SourceHeight = size;
                into.Ratio = 1.0;
                yield return Schedule(exact, into, clock);
                yield return Collect(into, clock, 1.0, blocking);
            }
            finally
            {
                running = false;
            }
        }

        /// <summary>
        /// Dispatches the network: all at once when StepsPerFrame is 0, otherwise StepsPerFrame layers per rendered
        /// frame through Worker.ScheduleIterable, so no frame carries the whole network (D88). The input stays fixed
        /// until the inference ends: one inference is in flight, and its tensor is written once, before the first step.
        /// </summary>
        private IEnumerator Schedule(Tensor<float> source, DetectionResult into, Stopwatch clock)
        {
            int perFrame = StepsPerFrame;
            into.StepsPerFrame = perFrame;
            if (Slices != null)
            {
                var plan = worker.ScheduleIterable(source);
                int done = 0, framesUsed = 0;
                foreach (int slice in Slices)
                {
                    if (framesUsed > 0) yield return null;
                    framesUsed++;
                    for (int k = 0; k < slice; k++)
                    {
                        if (!plan.MoveNext()) break;
                        done++;
                    }
                }
                while (plan.MoveNext()) done++;   // never leave the network unscheduled (a schedule is checked at load)
                into.ScheduleSteps = done;
                into.ScheduleFrames = framesUsed;
                into.ScheduleEndFrame = Time.frameCount;
                into.ScheduleMs = clock.Elapsed.TotalMilliseconds;
                yield break;
            }
            if (perFrame <= 0)
            {
                worker.Schedule(source);
                into.ScheduleSteps = LayerCount;
                into.ScheduleFrames = 1;
                into.ScheduleEndFrame = Time.frameCount;
                into.ScheduleMs = clock.Elapsed.TotalMilliseconds;
                yield break;
            }
            var steps = worker.ScheduleIterable(source);
            int count = 0, frames = 1;
            while (true)
            {
                bool more = true;
                for (int k = 0; k < perFrame; k++)
                {
                    if (!steps.MoveNext())
                    {
                        more = false;
                        break;
                    }
                    count++;
                }
                if (!more) break;
                frames++;
                yield return null;
            }
            into.ScheduleSteps = count;
            into.ScheduleFrames = frames;
            into.ScheduleEndFrame = Time.frameCount;
            into.ScheduleMs = clock.Elapsed.TotalMilliseconds;
        }

        public IEnumerator ProfileTexture(Texture source, int spacing, ProfilePass into)
        {
            into.Clear();
            into.Spacing = spacing;
            if (running || source == null)
            {
                into.Error = running ? "busy" : "no source texture";
                yield break;
            }
            running = true;
            try
            {
                int w = source.width, h = source.height;
                double ratio = Math.Min((double)size / h, (double)size / w);
                int rw = (int)(w * ratio), rh = (int)(h * ratio);
                bool encode = QualitySettings.activeColorSpace == ColorSpace.Linear
                              && GraphicsFormatUtility.IsSRGBFormat(source.graphicsFormat);
                letterbox.SetVector(RegionId, new Vector4((float)rw / size, (float)rh / size, package.input.pad_value / 255f, 0f));
                letterbox.SetFloat(EncodeId, encode ? 1f : 0f);
                letterbox.SetFloat(FlipId, 0f);
                into.PreFrame = Time.frameCount;
                Graphics.Blit(source, canvas, letterbox);
                TextureConverter.ToTensor(canvas, input, new TextureTransform().SetDimensions(size, size, 3).SetCoordOrigin(CoordOrigin.TopLeft));
                for (int k = 0; k < spacing; k++) yield return null;
                var plan = worker.ScheduleIterable(input);
                while (true)
                {
                    into.StepFrames.Add(Time.frameCount);
                    if (!plan.MoveNext())
                    {
                        into.StepFrames.RemoveAt(into.StepFrames.Count - 1);
                        break;
                    }
                    for (int k = 0; k < spacing; k++) yield return null;
                }
                into.ReadbackFrame = Time.frameCount;
                var boxesAwaiter = (worker.PeekOutput("boxes") as Tensor<float>).ReadbackAndCloneAsync().GetAwaiter();
                var scoresAwaiter = (worker.PeekOutput("scores") as Tensor<float>).ReadbackAndCloneAsync().GetAwaiter();
                var classesAwaiter = (worker.PeekOutput("classes") as Tensor<int>).ReadbackAndCloneAsync().GetAwaiter();
                int waited = 0;
                while (!boxesAwaiter.IsCompleted || !scoresAwaiter.IsCompleted || !classesAwaiter.IsCompleted)
                {
                    if (++waited > MaxWaitFrames)
                    {
                        abandoned = true;
                        into.Error = "the readbacks did not complete within " + MaxWaitFrames + " frames";
                        yield break;
                    }
                    yield return null;
                }
                into.DoneFrame = Time.frameCount;
                boxesAwaiter.GetResult().Dispose();
                scoresAwaiter.GetResult().Dispose();
                classesAwaiter.GetResult().Dispose();
                into.Completed = true;
            }
            finally
            {
                running = false;
            }
        }

        private IEnumerator Collect(DetectionResult into, Stopwatch clock, double ratio, bool blocking)
        {
            into.Floor = Floor;
            into.Blocking = blocking;
            float[] boxes, scores;
            int[] classes;
            if (blocking)
            {
                // DownloadToArray waits for the scheduled work and copies the output: the call this project's
                // A1.7 model test already used with Inference Engine 2.2.1.
                boxes = (worker.PeekOutput("boxes") as Tensor<float>).DownloadToArray();
                scores = (worker.PeekOutput("scores") as Tensor<float>).DownloadToArray();
                classes = (worker.PeekOutput("classes") as Tensor<int>).DownloadToArray();
            }
            else
            {
                // All three readbacks are requested together and complete in parallel (D87); waiting for them one
                // after another cost about ten frames per inference in r040.
                var boxesAwaiter = (worker.PeekOutput("boxes") as Tensor<float>).ReadbackAndCloneAsync().GetAwaiter();
                var scoresAwaiter = (worker.PeekOutput("scores") as Tensor<float>).ReadbackAndCloneAsync().GetAwaiter();
                var classesAwaiter = (worker.PeekOutput("classes") as Tensor<int>).ReadbackAndCloneAsync().GetAwaiter();
                while (!boxesAwaiter.IsCompleted || !scoresAwaiter.IsCompleted || !classesAwaiter.IsCompleted)
                {
                    if (++into.FramesWaited > MaxWaitFrames)
                    {
                        abandoned = true;
                        into.Error = "the readbacks did not complete within " + MaxWaitFrames + " frames";
                        yield break;
                    }
                    yield return null;
                }
                using (var t = boxesAwaiter.GetResult()) boxes = t.DownloadToArray();
                using (var t = scoresAwaiter.GetResult()) scores = t.DownloadToArray();
                using (var t = classesAwaiter.GetResult()) classes = t.DownloadToArray();
            }
            into.LatencyMs = clock.Elapsed.TotalMilliseconds;
            if (boxes == null || scores == null || classes == null || scores.Length != classes.Length || boxes.Length != 4 * scores.Length)
                throw new InvalidOperationException("the model's outputs do not have the package's shapes: boxes " +
                    (boxes?.Length ?? -1) + ", scores " + (scores?.Length ?? -1) + ", classes " + (classes?.Length ?? -1));
            var post = Stopwatch.StartNew();
            Postprocess(boxes, scores, classes, ratio, into);
            into.PostprocessMs = post.Elapsed.TotalMilliseconds;
            into.EndFrame = Time.frameCount;
            into.Completed = true;
        }

        private void Postprocess(float[] boxes, float[] scores, int[] classes, double ratio, DetectionResult into)
        {
            float threshold = Mathf.Min(Floor, package.postprocess.score_threshold);
            float iouThreshold = package.postprocess.nms_iou_threshold;
            candidates.Clear();
            for (int i = 0; i < scores.Length; i++)
                if (scores[i] >= threshold) candidates.Add(i);
            into.Candidates = candidates.Count;
            if (candidates.Count == 0) return;
            candidates.Sort((a, b) =>
            {
                int c = scores[b].CompareTo(scores[a]);
                return c != 0 ? c : a.CompareTo(b);
            });
            int n = candidates.Count;
            if (x1.Length < n)
            {
                x1 = new float[n]; y1 = new float[n]; x2 = new float[n]; y2 = new float[n];
                area = new float[n]; suppressed = new bool[n];
            }
            for (int k = 0; k < n; k++)
            {
                int i = candidates[k];
                x1[k] = boxes[4 * i]; y1[k] = boxes[4 * i + 1]; x2[k] = boxes[4 * i + 2]; y2[k] = boxes[4 * i + 3];
                area[k] = (x2[k] - x1[k] + 1f) * (y2[k] - y1[k] + 1f);
                suppressed[k] = false;
            }
            for (int k = 0; k < n; k++)
            {
                if (suppressed[k]) continue;
                int i = candidates[k];
                into.Detections.Add(new Detection
                {
                    ClassId = classes[i],
                    Score = scores[i],
                    X1 = (float)(x1[k] / ratio), Y1 = (float)(y1[k] / ratio),
                    X2 = (float)(x2[k] / ratio), Y2 = (float)(y2[k] / ratio)
                });
                for (int j = k + 1; j < n; j++)
                {
                    if (suppressed[j]) continue;
                    float w = Mathf.Max(0f, Mathf.Min(x2[k], x2[j]) - Mathf.Max(x1[k], x1[j]) + 1f);
                    float h = Mathf.Max(0f, Mathf.Min(y2[k], y2[j]) - Mathf.Max(y1[k], y1[j]) + 1f);
                    float inter = w * h;
                    float overlap = inter / (area[k] + area[j] - inter);
                    if (overlap > iouThreshold) suppressed[j] = true;
                }
            }
        }

        private void SaveCanvas(string path)
        {
            var previous = RenderTexture.active;
            RenderTexture.active = canvas;
            readback.ReadPixels(new Rect(0, 0, size, size), 0, 0, false);
            readback.Apply(false);
            RenderTexture.active = previous;
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllBytes(path, readback.EncodeToPNG());
        }

        public void Dispose()
        {
            if (worker != null)
            {
                foreach (var name in abandoned ? new string[0] : new[] { "boxes", "scores", "classes" })
                {
                    try { worker.PeekOutput(name)?.CompleteAllPendingOperations(); }
                    catch (Exception) { /* nothing was scheduled yet */ }
                }
                worker.Dispose();
                worker = null;
            }
            input?.Dispose();
            input = null;
            if (canvas != null)
            {
                canvas.Release();
                Object.Destroy(canvas);
                canvas = null;
            }
            if (letterbox != null) Object.Destroy(letterbox);
            if (readback != null) Object.Destroy(readback);
            letterbox = null;
            readback = null;
        }
    }
}
