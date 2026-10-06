using System;
using System.Collections;
using System.Collections.Generic;
using Stopwatch = System.Diagnostics.Stopwatch;
using System.IO;
using System.Text;
using SecondEyes.Logging;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace SecondEyes.Perception
{
    /// <summary>
    /// Runs the detector in the app (A1.10c, D84 to D86). Turning it on loads the package, warms it up, runs the
    /// self-checks (the exact test canvases, then the full-size test images through the GPU letterbox, as an sRGB and
    /// as a linear texture), then detects on camera frames until it is turned off, which releases the model. Every
    /// step is logged (docs/logging.md, detector.*); parity with the PC is judged offline by
    /// `python -m perception.detector parity`.
    /// </summary>
    public class DetectorRunner : MonoBehaviour
    {
        [Header("Detector package (D84)")]
        [SerializeField] private ModelAsset model;
        [SerializeField] private TextAsset packageManifest;
        [SerializeField] private Shader letterboxShader;

        [Header("Frames")]
        [SerializeField] private PassthroughFrameSource frameSource;
        [Tooltip("Set if camera frames turn out to be stored bottom row first (check with a snapshot).")]
        [SerializeField] private bool flipCamera;

        [Header("Self-checks, run each time the detector starts")]
        [SerializeField] private bool runChecks = true;
        [Tooltip("The *_416.png.bytes test canvases.")]
        [SerializeField] private TextAsset[] canvasImages;
        [Tooltip("The *_full.png.bytes test images.")]
        [SerializeField] private TextAsset[] fullImages;

        [Header("Running")]
        [SerializeField] private bool useGpu = true;
        [Tooltip("Camera inferences per second; 0 runs them back to back.")]
        [SerializeField] private float maxRateHz;
        [SerializeField] private int maxLoggedDetections = 20;
        [Tooltip("Read camera results back asynchronously (no hitch); the warm-up and self-checks always read back synchronously.")]
        [SerializeField] private bool asyncCameraReadback = true;
        [Tooltip("Frames without progress before the watchdog reports the detector as stuck.")]
        [SerializeField] private int watchdogFrames = 300;

        private IObjectDetector detector;
        private readonly DetectionResult result = new DetectionResult();
        private bool wanted, loopRunning, snapshotRequested;
        private int snapshots, cameraInferences, lastProgressFrame;
        private string stage = "idle";
        private Coroutine handle;
        private readonly Stack<IEnumerator> running = new Stack<IEnumerator>();
        private bool watchdogReported;

        /// <summary>One short line on what the detector is doing, for the debug overlay; null until first used.</summary>
        public static string StatusLine { get; private set; }

        public bool IsOn => wanted;
        public string Backend => useGpu ? "gpu_compute" : "cpu";

        public void Toggle(string source) => SetOn(!wanted, source);

        public void SetOn(bool on, string source)
        {
            if (on == wanted) return;
            wanted = on;
            var sb = new StringBuilder("{\"on\":").Append(on ? "true" : "false").Append(",\"source\":");
            Json.AppendString(sb, source);
            EventLog.Write("detector.toggle", sb.Append('}').ToString());
            if (on)
            {
                if (loopRunning) Stop();
                loopRunning = true;
                watchdogReported = false;
                lastProgressFrame = Time.frameCount;
                Stage("start", "starting");
                handle = StartCoroutine(Guarded(Run()));
            }
            else if (loopRunning && stage != "camera")
            {
                Stop();   // the camera loop ends by itself at its next frame; anything else is stopped now
            }
        }

        /// <summary>Switches between the GPU and CPU backends; only while the detector is off.</summary>
        public void SwitchBackend(string source)
        {
            if (wanted || loopRunning)
            {
                EventLog.Error("detector", "turn the detector off before switching its backend");
                return;
            }
            useGpu = !useGpu;
            var sb = new StringBuilder("{\"backend\":");
            Json.AppendString(sb, Backend);
            sb.Append(",\"source\":");
            Json.AppendString(sb, source);
            EventLog.Write("detector.backend", sb.Append('}').ToString());
        }

        /// <summary>Saves the next camera frame's letterboxed input, as the model sees it, beside the logs.</summary>
        public void RequestSnapshot(string source)
        {
            if (!wanted) return;
            snapshotRequested = true;
            EventLog.Mark("detector snapshot requested (" + source + ")");
        }

        [ContextMenu("Detector on")]
        private void OnFromMenu() => SetOn(true, "context_menu");

        [ContextMenu("Detector off")]
        private void OffFromMenu() => SetOn(false, "context_menu");

        /// <summary>
        /// Runs a coroutine and every coroutine it yields as one flat sequence, so an exception at any depth is caught:
        /// it is logged (an error event naming the stage, and Unity's console), shown on the overlay, and the detector
        /// is released and can be started again. Without this, an exception inside a nested coroutine stops the chain
        /// silently and leaves the runner stuck.
        /// </summary>
        private IEnumerator Guarded(IEnumerator root)
        {
            var stack = running;
            stack.Clear();
            stack.Push(root);
            while (stack.Count > 0)
            {
                lastProgressFrame = Time.frameCount;
                object current = null;
                bool moved = false;
                Exception failure = null;
                try
                {
                    moved = stack.Peek().MoveNext();
                    if (moved) current = stack.Peek().Current;
                }
                catch (Exception e)
                {
                    failure = e;
                }
                if (failure != null)
                {
                    Fail(stage, failure.GetType().Name + ": " + failure.Message + FirstFrame(failure));
                    Debug.LogException(failure);
                    while (stack.Count > 0)
                    {
                        try { (stack.Pop() as IDisposable)?.Dispose(); }
                        catch (Exception e) { Debug.LogException(e); }
                    }
                    loopRunning = false;
                    handle = null;
                    yield break;
                }
                if (!moved)
                {
                    stack.Pop();
                    continue;
                }
                if (current is IEnumerator nested)
                {
                    stack.Push(nested);
                    continue;
                }
                yield return current;
            }
        }

        /// <summary>Stops the coroutine wherever it is, finishes its iterators (running their cleanup) and releases
        /// the detector.</summary>
        private void Stop()
        {
            if (handle != null) StopCoroutine(handle);
            handle = null;
            while (running.Count > 0)
            {
                try { (running.Pop() as IDisposable)?.Dispose(); }
                catch (Exception e) { Debug.LogException(e); }
            }
            Unload();
            loopRunning = false;
            if (StatusLine == null || !StatusLine.StartsWith("failed")) StatusLine = "off";
        }

        /// <summary>Notes a stage: the overlay line, Unity's console and a mark in the event log.</summary>
        private void Stage(string name, string status)
        {
            stage = name;
            StatusLine = status;
            Debug.Log("[SecondEyes] Detector: " + status + " (frame " + Time.frameCount + ")");
            EventLog.Mark("detector: " + status);
        }

        private void Update()
        {
            if (!loopRunning || watchdogReported || Time.frameCount - lastProgressFrame < watchdogFrames) return;
            watchdogReported = true;
            string message = "no progress for " + (Time.frameCount - lastProgressFrame) + " frames at " + stage +
                             " (frame " + Time.frameCount + "): its coroutine is not being resumed";
            Debug.LogWarning("[SecondEyes] Detector: " + message);
            EventLog.Error("detector", message);
            StatusLine = "stuck at " + stage;
        }

        private void Fail(string where, string message)
        {
            wanted = false;
            EventLog.Error("detector", where + ": " + message);
            StatusLine = "failed at " + where + ": " + message;
            if (StatusLine.Length > 90) StatusLine = StatusLine.Substring(0, 90) + "...";
        }

        private static string FirstFrame(Exception e)
        {
            string trace = e.StackTrace;
            if (string.IsNullOrEmpty(trace)) return "";
            string first = trace.Split('\n')[0].Trim();
            return first.Length > 0 ? " (" + first + ")" : "";
        }

        private IEnumerator Run()
        {
            loopRunning = true;
            try
            {
                Stage("load", "loading");
                var clock = Stopwatch.StartNew();
                var package = DetectorPackage.Parse(packageManifest, out string problem);
                if (package == null)
                {
                    Fail(stage, problem);
                    yield break;
                }
                string failure = null;
                try
                {
                    detector = new YoloxDetector(package, model, letterboxShader, useGpu ? BackendType.GPUCompute : BackendType.CPU);
                }
                catch (Exception e)
                {
                    failure = e.GetType().Name + ": " + e.Message;
                }
                if (failure != null)
                {
                    Fail(stage, failure);
                    yield break;
                }
                double loadMs = clock.Elapsed.TotalMilliseconds;
                var gray = new Color32[detector.InputSize * detector.InputSize];
                byte pad = (byte)package.input.pad_value;
                for (int k = 0; k < gray.Length; k++) gray[k] = new Color32(pad, pad, pad, 255);
                Stage("warm-up", "warming up (" + detector.Backend + ")");
                var warm = Stopwatch.StartNew();
                yield return detector.DetectCanvas(gray, true, result);
                LogLoad(package, loadMs, warm.Elapsed.TotalMilliseconds);
                if (!result.Completed)
                {
                    Fail(stage, result.Error ?? "the first inference did not complete");
                    yield break;
                }

                Stage("warm-up", "warmed up in " + Mathf.RoundToInt((float)warm.Elapsed.TotalMilliseconds) + " ms");
                yield return null;
                if (runChecks) yield return Checks();
                Stage("camera", "on, waiting for camera frames (X)");

                bool sourceLogged = false;
                float next = 0f;
                while (wanted)
                {
                    var texture = frameSource != null ? frameSource.CurrentTexture : null;
                    if (texture == null || Time.realtimeSinceStartup < next)
                    {
                        if (texture == null && cameraInferences == 0) StatusLine = "on, waiting for camera frames (X)";
                        yield return null;
                        continue;
                    }
                    if (!sourceLogged)
                    {
                        LogSource("camera", texture, flipCamera);
                        sourceLogged = true;
                    }
                    next = maxRateHz > 0f ? Time.realtimeSinceStartup + 1f / maxRateHz : 0f;
                    string snapshot = null;
                    if (snapshotRequested)
                    {
                        snapshotRequested = false;
                        snapshots++;
                        snapshot = EventLog.SessionId + "_" + snapshots.ToString("D3");
                    }
                    yield return detector.DetectTexture(texture, flipCamera, snapshot == null ? null : SnapshotPath(snapshot),
                                                        !asyncCameraReadback, result);
                    LogResult("camera", null, snapshot);
                    cameraInferences++;
                    StatusLine = CameraStatus(snapshot != null);
                    yield return null;
                }
            }
            finally
            {
                Unload();
                loopRunning = false;
                handle = null;
                if (StatusLine == null || !StatusLine.StartsWith("failed")) StatusLine = "off";
            }
        }

        private string CameraStatus(bool snapshot)
        {
            var sb = new StringBuilder(detector.Backend == "gpu_compute" ? "GPU" : "CPU");
            sb.Append(" #").Append(cameraInferences).Append(' ');
            if (!result.Completed)
                return sb.Append("incomplete: ").Append(result.Error ?? "?").ToString();
            sb.Append(Mathf.RoundToInt((float)result.LatencyMs)).Append(" ms, ").Append(result.Detections.Count).Append(" found");
            for (int k = 0; k < Mathf.Min(2, result.Detections.Count); k++)
            {
                var d = result.Detections[k];
                var names = detector.Classes;
                string name = d.ClassId >= 0 && d.ClassId < names.Count ? names[d.ClassId] : d.ClassId.ToString();
                sb.Append(k == 0 ? ": " : ", ").Append(name).Append(' ').Append(d.Score.ToString("0.00"));
            }
            if (snapshot) sb.Append(" [snapshot]");
            return sb.ToString();
        }

        private IEnumerator Checks()
        {
            int total = (canvasImages?.Length ?? 0) + 2 * (fullImages?.Length ?? 0), done = 0;
            foreach (var asset in canvasImages ?? new TextAsset[0])
            {
                if (asset == null) continue;
                Stage("self-check " + asset.name, "self-checks " + (++done) + "/" + total);
                var pixels = DecodeTopDown(asset.bytes, out int w, out int h);
                if (pixels == null || w != detector.InputSize || h != detector.InputSize)
                {
                    EventLog.Error("detector", asset.name + " is not a " + detector.InputSize + "-pixel test canvas");
                    continue;
                }
                yield return detector.DetectCanvas(pixels, true, result);
                LogResult("tensor", ImageId(asset), null);
                yield return null;   // one blocking inference per frame
            }
            foreach (var asset in fullImages ?? new TextAsset[0])
            {
                if (asset == null) continue;
                foreach (bool linear in new[] { false, true })
                {
                    var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, linear);
                    if (!texture.LoadImage(asset.bytes, false))
                    {
                        EventLog.Error("detector", asset.name + " could not be decoded");
                        Destroy(texture);
                        continue;
                    }
                    texture.filterMode = FilterMode.Bilinear;
                    texture.wrapMode = TextureWrapMode.Clamp;
                    string path = linear ? "texture_linear" : "texture_srgb";
                    Stage("self-check " + asset.name + " " + path, "self-checks " + (++done) + "/" + total);
                    LogSource(path, texture, false);
                    yield return detector.DetectTexture(texture, false, null, true, result);
                    LogResult(path, ImageId(asset), null);
                    Destroy(texture);
                    yield return null;
                }
            }
        }

        private void Unload()
        {
            if (detector == null) return;
            string backend = detector.Backend;
            detector.Dispose();
            detector = null;
            var sb = new StringBuilder("{\"backend\":");
            Json.AppendString(sb, backend);
            EventLog.Write("detector.unload", sb.Append('}').ToString());
        }

        private void OnDestroy() => Unload();

        private static Color32[] DecodeTopDown(byte[] bytes, out int width, out int height)
        {
            var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
            width = height = 0;
            if (!texture.LoadImage(bytes, false))
            {
                Destroy(texture);
                return null;
            }
            width = texture.width;
            height = texture.height;
            var bottomUp = texture.GetPixels32();
            Destroy(texture);
            var topDown = new Color32[bottomUp.Length];
            for (int y = 0; y < height; y++)
                Array.Copy(bottomUp, (height - 1 - y) * width, topDown, y * width, width);
            return topDown;
        }

        private static string ImageId(TextAsset asset)
        {
            string name = asset.name;
            int cut = name.IndexOf("_416", StringComparison.Ordinal);
            if (cut < 0) cut = name.IndexOf("_full", StringComparison.Ordinal);
            return cut > 0 ? name.Substring(0, cut) : name;
        }

        private static string SnapshotPath(string snapshot) =>
            Path.Combine(Application.persistentDataPath, "snapshots", snapshot + ".png");

        private void LogLoad(DetectorPackage package, double loadMs, double warmupMs)
        {
            var sb = new StringBuilder("{\"package_id\":");
            Json.AppendString(sb, package.package_id);
            sb.Append(",\"model_sha256\":");
            Json.AppendString(sb, package.model_sha256);
            sb.Append(",\"backend\":");
            Json.AppendString(sb, detector.Backend);
            sb.Append(",\"input_size\":").Append(detector.InputSize).Append(",\"load_ms\":");
            Json.AppendNumber(sb, (float)loadMs);
            sb.Append(",\"warmup_ms\":");
            Json.AppendNumber(sb, (float)warmupMs);
            sb.Append(",\"warmup_completed\":").Append(result.Completed ? "true" : "false");
            sb.Append(",\"color_space\":");
            Json.AppendString(sb, QualitySettings.activeColorSpace == ColorSpace.Linear ? "linear" : "gamma");
            EventLog.Write("detector.load", sb.Append('}').ToString());
        }

        private void LogSource(string path, Texture texture, bool flip)
        {
            bool srgb = GraphicsFormatUtility.IsSRGBFormat(texture.graphicsFormat);
            bool linearSpace = QualitySettings.activeColorSpace == ColorSpace.Linear;
            var sb = new StringBuilder("{\"path\":");
            Json.AppendString(sb, path);
            sb.Append(",\"graphics_format\":");
            Json.AppendString(sb, texture.graphicsFormat.ToString());
            sb.Append(",\"srgb\":").Append(srgb ? "true" : "false")
              .Append(",\"width\":").Append(texture.width).Append(",\"height\":").Append(texture.height)
              .Append(",\"encode_srgb\":").Append(linearSpace && srgb ? "true" : "false")
              .Append(",\"flip\":").Append(flip ? "true" : "false");
            EventLog.Write("detector.source", sb.Append('}').ToString());
        }

        private void LogResult(string path, string imageId, string snapshot)
        {
            var sb = new StringBuilder("{\"path\":");
            Json.AppendString(sb, path);
            sb.Append(",\"image_id\":");
            if (imageId == null) sb.Append("null"); else Json.AppendString(sb, imageId);
            sb.Append(",\"backend\":");
            Json.AppendString(sb, detector != null ? detector.Backend : Backend);
            sb.Append(",\"readback\":").Append(result.Blocking ? "\"blocking\"" : "\"async\"");
            sb.Append(",\"completed\":").Append(result.Completed ? "true" : "false").Append(",\"error\":");
            if (result.Error == null) sb.Append("null"); else Json.AppendString(sb, result.Error);
            sb.Append(",\"width\":").Append(result.SourceWidth).Append(",\"height\":").Append(result.SourceHeight)
              .Append(",\"ratio\":");
            Json.AppendNumber(sb, (float)result.Ratio);
            sb.Append(",\"schedule_ms\":");
            Json.AppendNumber(sb, (float)result.ScheduleMs);
            sb.Append(",\"latency_ms\":");
            Json.AppendNumber(sb, (float)result.LatencyMs);
            sb.Append(",\"postprocess_ms\":");
            Json.AppendNumber(sb, (float)result.PostprocessMs);
            sb.Append(",\"frames_waited\":").Append(result.FramesWaited)
              .Append(",\"candidates\":").Append(result.Candidates)
              .Append(",\"detections_total\":").Append(result.Detections.Count).Append(",\"detections\":[");
            int n = Mathf.Min(result.Detections.Count, maxLoggedDetections);
            for (int k = 0; k < n; k++)
            {
                var d = result.Detections[k];
                if (k > 0) sb.Append(',');
                sb.Append('[').Append(d.ClassId).Append(',');
                Json.AppendNumber(sb, d.Score);
                sb.Append(',');
                Json.AppendNumber(sb, d.X1);
                sb.Append(',');
                Json.AppendNumber(sb, d.Y1);
                sb.Append(',');
                Json.AppendNumber(sb, d.X2);
                sb.Append(',');
                Json.AppendNumber(sb, d.Y2);
                sb.Append(']');
            }
            sb.Append("],\"snapshot\":");
            if (snapshot == null) sb.Append("null"); else Json.AppendString(sb, snapshot);
            EventLog.Write("detector.result", sb.Append('}').ToString());
        }
    }
}
