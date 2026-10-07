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
    /// Runs the detector in the app (A1.10c and A1.10d; D84 to D88). Turning it on loads the package, warms it up and
    /// runs the self-checks (the exact test canvases, any PNG canvas pushed to persistent data/parity/, and the
    /// full-size test images through the GPU letterbox), all through the same sliced, asynchronous path as camera
    /// frames. It then works in one of three modes: scan (stop-and-shoot keyframes: hovers x keyframes at a fixed
    /// rate, with repositioning pauses), burst (a short run at a higher rate, diagnostic) or continuous (back to back,
    /// stress). One inference is in flight at a time; a keyframe that comes due while one is running is rejected and
    /// logged, never queued. Turning it off stops new keyframes, lets the inference in flight finish and then releases
    /// the model. Every step is logged (docs/logging.md, detector.* and frame.second).
    /// </summary>
    public class DetectorRunner : MonoBehaviour
    {
        public enum Mode { Scan, Diagnostic, Profile, Burst, Continuous }

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
        [Tooltip("Also check every 416 x 416 PNG in <persistent data>/parity/, pushed with adb (a saved snapshot).")]
        [SerializeField] private bool deviceCanvases = true;

        [Header("Running (D88)")]
        [SerializeField] private bool useGpu = true;
        [SerializeField] private Mode mode = Mode.Scan;
        [Tooltip("Scheduling steps (layers) per rendered frame to choose from; 0 dispatches the whole network in one frame.")]
        [SerializeField] private int[] stepsChoices = { 4, 8, 16, 32, 64, 0 };
        [SerializeField] private int stepsIndex = 1;
        [SerializeField] private float keyframeRateHz = 1f;
        [SerializeField] private int hoversPerScan = 3;
        [SerializeField] private int keyframesPerHover = 12;
        [SerializeField] private float repositionS = 8f;
        [SerializeField] private float scanGapS = 15f;
        [Tooltip("Diagnostic mode: one scan at this rate, slow enough that inferences cannot overlap (D89).")]
        [SerializeField] private float diagnosticRateHz = 0.5f;
        [Tooltip("Profile mode: passes, and empty frames after the preprocessing and after each single step (D89).")]
        [SerializeField] private int profilePasses = 5;
        [SerializeField] private int profileSpacing = 4;
        [Tooltip("Before the mode starts: short deliberate main-thread stalls at logged times, so OVR Metrics' one-second buckets can be aligned with the log (D89).")]
        [SerializeField] private bool clockSync = true;
        [SerializeField] private int syncPulses = 8;
        [SerializeField] private float syncSpacingS = 1.37f;
        [SerializeField] private float syncStallMs = 60f;
        [SerializeField] private float burstRateHz = 2f;
        [SerializeField] private float burstSeconds = 30f;
        [Tooltip("Lowest score kept after suppression, for parity's threshold-crossing checks; detections are those at or above the package's threshold.")]
        [SerializeField] private float floor = 0.25f;
        [SerializeField] private int maxLoggedDetections = 20;
        [Tooltip("Read camera results back asynchronously (all three outputs requested together, D87).")]
        [SerializeField] private bool asyncCameraReadback = true;
        [Tooltip("Frames without progress before the watchdog reports the detector stuck and stops it.")]
        [SerializeField] private int watchdogFrames = 300;
        [Tooltip("When turned off, how long to wait for the inference in flight before releasing anyway.")]
        [SerializeField] private float releaseWaitS = 5f;

        [Serializable]
        private class ScheduleFile
        {
            public int format_version;
            public string record_type;
            public string schedule_id;
            public string model_sha256;
            public string backend;
            public int layer_count;
            public string layer_types_sha256;
            public int[] slices;
            public float budget_ms;
        }

        private readonly List<ScheduleFile> schedules = new List<ScheduleFile>();
        private ScheduleFile activeSchedule;
        private readonly ProfilePass profilePass = new ProfilePass();

        private struct KeyInfo
        {
            public int? Keyframe, Scan, Hover;
            public long? RequestedUtc;
            public long StartedUtc;
            public string Stamp;
            public bool HasPose;
            public Pose Pose;
        }

        private IObjectDetector detector;
        private readonly DetectionResult checkResult = new DetectionResult();
        private readonly DetectionResult keyResult = new DetectionResult();
        private readonly Stack<IEnumerator> mainStack = new Stack<IEnumerator>(), keyStack = new Stack<IEnumerator>();
        private Coroutine mainHandle, keyHandle;
        private bool wanted, loopRunning, inFlight, snapshotRequested, watchdogReported;
        private int snapshots, keyframes, lastProgressFrame, blockRequested, blockCompleted, blockRejected;
        private int scanRequested, scanCompleted, scanRejected;
        private string stage = "idle";

        /// <summary>One short line on what the detector is doing, for the debug overlay; null until the scene starts.</summary>
        public static string StatusLine { get; private set; }

        public bool IsOn => wanted;
        public bool IsBusy => inFlight;
        public string Backend => useGpu ? "gpu_compute" : "cpu";
        private int ChoiceCount => (stepsChoices?.Length ?? 0) + schedules.Count;
        private ScheduleFile ChosenSchedule => stepsIndex >= (stepsChoices?.Length ?? 0) && stepsIndex < ChoiceCount
            ? schedules[stepsIndex - stepsChoices.Length] : null;
        private int Steps => ChosenSchedule != null || stepsChoices == null || stepsChoices.Length == 0
            ? 0 : stepsChoices[Mathf.Clamp(stepsIndex, 0, stepsChoices.Length - 1)];
        private string ChoiceName => ChosenSchedule != null ? "schedule " + ChosenSchedule.schedule_id : StepsName(Steps);
        private static long UtcUs => (DateTime.UtcNow.Ticks - 621355968000000000L) / 10;

        private void Start()
        {
            LoadSchedules();
            StatusLine = OffLine();
        }

        /// <summary>Frozen schedules pushed to persistent data/schedules/ (D89); checked against the model at load.</summary>
        private void LoadSchedules()
        {
            schedules.Clear();
            string dir = Path.Combine(Application.persistentDataPath, "schedules");
            if (!Directory.Exists(dir)) return;
            var files = new List<string>(Directory.GetFiles(dir, "*.json"));
            files.Sort(StringComparer.Ordinal);
            foreach (var file in files)
            {
                try
                {
                    var s = JsonUtility.FromJson<ScheduleFile>(File.ReadAllText(file));
                    if (s != null && s.format_version == 1 && s.record_type == "detector_schedule" && s.slices != null && s.slices.Length > 0)
                        schedules.Add(s);
                    else
                        EventLog.Error("detector", Path.GetFileName(file) + " is not a version 1 detector_schedule");
                }
                catch (Exception e)
                {
                    EventLog.Error("detector", Path.GetFileName(file) + ": " + e.Message);
                }
            }
        }

        private string OffLine() => "off | " + (useGpu ? "GPU" : "CPU") + " | " + ModeName(mode) + " | " + ChoiceName;
        private static string ModeName(Mode m) => m == Mode.Scan ? "scan" : m == Mode.Diagnostic ? "diagnostic"
            : m == Mode.Profile ? "profile" : m == Mode.Burst ? "burst" : "continuous";
        private static string StepsName(int s) => s <= 0 ? "whole network per frame" : s + " steps/frame";

        public void Toggle(string source) => SetOn(!wanted, source);

        public void SetOn(bool on, string source)
        {
            if (on == wanted) return;
            if (on && loopRunning)
            {
                EventLog.Mark("detector: still stopping; turn it on again once the overlay says off");
                return;
            }
            wanted = on;
            var sb = new StringBuilder("{\"on\":").Append(on ? "true" : "false").Append(",\"source\":");
            Json.AppendString(sb, source);
            EventLog.Write("detector.toggle", sb.Append('}').ToString());
            if (on)
            {
                loopRunning = true;
                watchdogReported = false;
                lastProgressFrame = Time.frameCount;
                Stage("start", "starting");
                mainHandle = StartCoroutine(Guarded(Run(), mainStack));
            }
            else
            {
                StatusLine = "stopping: finishing the inference in flight";
            }
        }

        /// <summary>Switches between the GPU and CPU backends; only while the detector is off.</summary>
        public void SwitchBackend(string source)
        {
            if (!Idle()) return;
            useGpu = !useGpu;
            LogSetting(source);
        }

        /// <summary>Cycles scan, burst and continuous; only while the detector is off.</summary>
        public void CycleMode(int step, string source)
        {
            if (!Idle()) return;
            int n = Enum.GetValues(typeof(Mode)).Length;
            mode = (Mode)((((int)mode + step) % n + n) % n);
            LogSetting(source);
        }

        /// <summary>Cycles the scheduling steps per frame; only while the detector is off.</summary>
        public void CycleSteps(int step, string source)
        {
            if (!Idle() || ChoiceCount == 0) return;
            int n = ChoiceCount;
            stepsIndex = ((stepsIndex + step) % n + n) % n;
            LogSetting(source);
        }

        /// <summary>Saves the next camera keyframe's letterboxed input, as the model sees it, beside the logs.</summary>
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

        private bool Idle()
        {
            if (!wanted && !loopRunning) return true;
            EventLog.Error("detector", "turn the detector off before changing its settings");
            return false;
        }

        // ------------------------------------------------------------------------------------------------ guard
        /// <summary>
        /// Runs a coroutine and every coroutine it yields as one flat sequence, so an exception at any depth is caught:
        /// logged (an error event naming the stage, and Unity's console), shown on the overlay, and the detector is
        /// released. Every resume also counts as progress for the watchdog.
        /// </summary>
        private IEnumerator Guarded(IEnumerator root, Stack<IEnumerator> stack)
        {
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
                    DisposeStack(stack);
                    if (stack == keyStack)
                    {
                        inFlight = false;
                        keyHandle = null;
                    }
                    else
                    {
                        mainHandle = null;
                        ForceStop();
                    }
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

        private static void DisposeStack(Stack<IEnumerator> stack)
        {
            while (stack.Count > 0)
            {
                try { (stack.Pop() as IDisposable)?.Dispose(); }
                catch (Exception e) { Debug.LogException(e); }
            }
        }

        /// <summary>Stops everything at once and releases the detector: for failures and the watchdog only.</summary>
        private void ForceStop()
        {
            wanted = false;
            if (keyHandle != null) StopCoroutine(keyHandle);
            if (mainHandle != null) StopCoroutine(mainHandle);
            keyHandle = mainHandle = null;
            DisposeStack(keyStack);
            DisposeStack(mainStack);
            Release();
            loopRunning = false;
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
                             " (frame " + Time.frameCount + "): its coroutine is not being resumed; stopped";
            Debug.LogWarning("[SecondEyes] Detector: " + message);
            EventLog.Error("detector", message);
            ForceStop();
            StatusLine = "stuck at " + stage + "; stopped";
        }

        private void OnDestroy() => ForceStop();

        // ------------------------------------------------------------------------------------------------- run
        private IEnumerator Run()
        {
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
                detector.StepsPerFrame = Steps;
                detector.Floor = Mathf.Min(floor, package.postprocess.score_threshold);
                double loadMs = clock.Elapsed.TotalMilliseconds;
                string typesHash = LogLayers();
                activeSchedule = ChosenSchedule;
                if (activeSchedule != null)
                {
                    string why = ScheduleProblem(activeSchedule, package, typesHash);
                    LogSchedule(activeSchedule, why);
                    if (why != null)
                    {
                        Fail("schedule", why);
                        yield break;
                    }
                    detector.Slices = activeSchedule.slices;
                }
                var gray = new Color32[detector.InputSize * detector.InputSize];
                byte pad = (byte)package.input.pad_value;
                for (int k = 0; k < gray.Length; k++) gray[k] = new Color32(pad, pad, pad, 255);
                Stage("warm-up", "warming up (" + detector.Backend + ", " + ChoiceName + ")");
                var warm = Stopwatch.StartNew();
                yield return detector.DetectCanvas(gray, false, checkResult);
                LogLoad(package, loadMs, warm.Elapsed.TotalMilliseconds);
                if (!checkResult.Completed)
                {
                    Fail(stage, checkResult.Error ?? "the first inference did not complete");
                    yield break;
                }
                Stage("warm-up", "warmed up in " + Mathf.RoundToInt((float)warm.Elapsed.TotalMilliseconds) + " ms, " +
                                 detector.LayerCount + " layers, " + checkResult.ScheduleSteps + " steps");
                yield return null;
                if (runChecks && wanted) yield return Checks();
                if (clockSync && wanted) yield return SyncPulses();
                if (wanted) yield return Operate();
                var wait = Stopwatch.StartNew();
                while (inFlight && wait.Elapsed.TotalSeconds < releaseWaitS) yield return null;
            }
            finally
            {
                Release();
                loopRunning = false;
                mainHandle = null;
                if (StatusLine == null || !(StatusLine.StartsWith("failed") || StatusLine.StartsWith("stuck"))) StatusLine = OffLine();
            }
        }

        private IEnumerator Checks()
        {
            var device = new List<string>();
            string dir = Path.Combine(Application.persistentDataPath, "parity");
            if (deviceCanvases && Directory.Exists(dir)) device.AddRange(Directory.GetFiles(dir, "*.png"));
            device.Sort(StringComparer.Ordinal);
            int total = (canvasImages?.Length ?? 0) + device.Count + 2 * (fullImages?.Length ?? 0), done = 0;
            foreach (var asset in canvasImages ?? new TextAsset[0])
            {
                if (asset == null || !wanted) continue;
                Stage("self-check " + asset.name, "self-checks " + (++done) + "/" + total);
                yield return CheckCanvas(asset.bytes, ImageId(asset.name));
            }
            foreach (var path in device)
            {
                if (!wanted) break;
                Stage("self-check " + Path.GetFileName(path), "self-checks " + (++done) + "/" + total);
                yield return CheckCanvas(File.ReadAllBytes(path), Path.GetFileNameWithoutExtension(path));
            }
            foreach (var asset in fullImages ?? new TextAsset[0])
            {
                if (asset == null) continue;
                foreach (bool linear in new[] { false, true })
                {
                    if (!wanted) yield break;
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
                    var info = new KeyInfo { StartedUtc = UtcUs };
                    yield return detector.DetectTexture(texture, false, null, false, checkResult);
                    LogResult(checkResult, path, ImageId(asset.name), null, info);
                    Destroy(texture);
                    yield return null;
                }
            }
        }

        private IEnumerator CheckCanvas(byte[] bytes, string id)
        {
            var pixels = DecodeTopDown(bytes, out int w, out int h);
            if (pixels == null || w != detector.InputSize || h != detector.InputSize)
            {
                EventLog.Error("detector", id + " is not a " + detector.InputSize + "-pixel test canvas");
                yield break;
            }
            var info = new KeyInfo { StartedUtc = UtcUs };
            yield return detector.DetectCanvas(pixels, false, checkResult);
            LogResult(checkResult, "tensor", id, null, info);
            yield return null;
        }

        private Texture CameraTexture() => frameSource != null ? frameSource.CurrentTexture : null;

        private IEnumerator Operate()
        {
            while (wanted && CameraTexture() == null)
            {
                StatusLine = "on, waiting for camera frames (X)";
                yield return null;
            }
            if (!wanted) yield break;
            LogSource("camera", CameraTexture(), flipCamera);
            Stage("camera", "on: " + ModeName(mode) + ", " + ChoiceName);
            if (mode == Mode.Profile)
            {
                for (int pass = 1; pass <= profilePasses && wanted; pass++)
                {
                    var texture = CameraTexture();
                    if (texture == null)
                    {
                        yield return null;
                        pass--;
                        continue;
                    }
                    StatusLine = "profiling pass " + pass + "/" + profilePasses;
                    inFlight = true;
                    yield return detector.ProfileTexture(texture, Mathf.Max(1, profileSpacing), profilePass);
                    inFlight = false;
                    LogProfile(pass);
                    for (int k = 0; k < 2 * profileSpacing; k++) yield return null;
                }
                while (wanted)
                {
                    StatusLine = "profile done; Y to stop";
                    yield return null;
                }
            }
            else if (mode == Mode.Diagnostic)
            {
                scanRequested = scanCompleted = scanRejected = 0;
                LogScan(1, null, "start", 0, 0, 0);
                for (int hover = 1; hover <= hoversPerScan && wanted; hover++)
                {
                    yield return Ticks(diagnosticRateHz, keyframesPerHover, 1, hover);
                    if (hover < hoversPerScan) yield return Pause(repositionS, "diagnostic scan: repositioning");
                }
                LogScan(1, null, "end", scanRequested, scanCompleted, scanRejected);
                while (wanted)
                {
                    StatusLine = "diagnostic scan done; Y to stop";
                    yield return null;
                }
            }
            else if (mode == Mode.Continuous)
            {
                while (wanted)
                {
                    if (!inFlight) Begin(null, null, UtcUs);
                    yield return null;
                }
            }
            else if (mode == Mode.Burst)
            {
                yield return Ticks(burstRateHz, Mathf.Max(1, Mathf.RoundToInt(burstSeconds * burstRateHz)), null, null);
                while (wanted)
                {
                    StatusLine = "burst done; Y to stop";
                    yield return null;
                }
            }
            else
            {
                for (int scan = 1; wanted; scan++)
                {
                    scanRequested = scanCompleted = scanRejected = 0;
                    LogScan(scan, null, "start", 0, 0, 0);
                    for (int hover = 1; hover <= hoversPerScan && wanted; hover++)
                    {
                        yield return Ticks(keyframeRateHz, keyframesPerHover, scan, hover);
                        if (hover < hoversPerScan) yield return Pause(repositionS, "scan " + scan + ": repositioning");
                    }
                    LogScan(scan, null, "end", scanRequested, scanCompleted, scanRejected);
                    yield return Pause(scanGapS, "scan " + scan + " done; next scan soon");
                }
            }
        }

        /// <summary>
        /// Requests `count` keyframes on a fixed schedule at `rateHz`. A keyframe that comes due while an inference is
        /// in flight is rejected and logged; nothing is queued. After the last request, waits for the inference in
        /// flight, so the block's counts are final.
        /// </summary>
        private IEnumerator Ticks(float rateHz, int count, int? scan, int? hover)
        {
            blockRequested = blockCompleted = blockRejected = 0;
            if (hover != null) LogScan(scan ?? 0, hover, "start", 0, 0, 0);
            double start = Time.realtimeSinceStartupAsDouble, period = 1.0 / Mathf.Max(0.01f, rateHz);
            for (int k = 0; k < count && wanted; k++)
            {
                double due = start + k * period;
                while (wanted && Time.realtimeSinceStartupAsDouble < due) yield return null;
                if (!wanted) break;
                blockRequested++;
                long requested = UtcUs;
                if (inFlight || CameraTexture() == null)
                {
                    blockRejected++;
                    LogReject(scan, hover, requested, inFlight ? "busy" : "no_camera_frame");
                    continue;
                }
                Begin(scan, hover, requested);
            }
            var wait = Stopwatch.StartNew();
            while (inFlight && wait.Elapsed.TotalSeconds < 10) yield return null;
            scanRequested += blockRequested;
            scanCompleted += blockCompleted;
            scanRejected += blockRejected;
            if (hover != null) LogScan(scan ?? 0, hover, "end", blockRequested, blockCompleted, blockRejected);
        }

        /// <summary>
        /// Deliberate main-thread stalls at logged times (clock.sync), spaced so their sub-second phases differ: each
        /// shows up as stale frames in one OVR Metrics bucket, which fixes the offset between the two clocks far more
        /// tightly than incidental misses (D89). They run before the mode starts, outside any measured phase.
        /// </summary>
        private IEnumerator SyncPulses()
        {
            Stage("clock-sync", "clock sync pulses");
            double next = Time.realtimeSinceStartupAsDouble + 1.0;
            for (int k = 0; k < syncPulses && wanted; k++)
            {
                while (wanted && Time.realtimeSinceStartupAsDouble < next) yield return null;
                if (!wanted) yield break;
                long start = UtcUs;
                var spin = Stopwatch.StartNew();
                while (spin.Elapsed.TotalMilliseconds < syncStallMs) { }
                long end = UtcUs;
                var sb = new StringBuilder("{\"pulse\":").Append(k + 1).Append(",\"start_utc_us\":").Append(start)
                    .Append(",\"end_utc_us\":").Append(end).Append(",\"stall_ms\":");
                Json.AppendNumber(sb, syncStallMs);
                EventLog.Write("clock.sync", sb.Append('}').ToString());
                next += syncSpacingS;
            }
            double settle = Time.realtimeSinceStartupAsDouble + 2.0;
            while (wanted && Time.realtimeSinceStartupAsDouble < settle) yield return null;
        }

        private IEnumerator Pause(float seconds, string status)
        {
            double end = Time.realtimeSinceStartupAsDouble + seconds;
            while (wanted && Time.realtimeSinceStartupAsDouble < end)
            {
                StatusLine = status;
                yield return null;
            }
        }

        private void Begin(int? scan, int? hover, long requestedUtc)
        {
            var texture = CameraTexture();
            if (texture == null || detector == null) return;
            inFlight = true;
            keyframes++;
            keyHandle = StartCoroutine(Guarded(Keyframe(texture, keyframes, scan, hover, requestedUtc), keyStack));
        }

        private IEnumerator Keyframe(Texture texture, int index, int? scan, int? hover, long requestedUtc)
        {
            try
            {
                var info = new KeyInfo { Keyframe = index, Scan = scan, Hover = hover, RequestedUtc = requestedUtc };
                if (frameSource != null) info.HasPose = frameSource.TryGetCapture(out info.Stamp, out info.Pose);
                info.StartedUtc = UtcUs;
                string snapshot = null;
                if (snapshotRequested)
                {
                    snapshotRequested = false;
                    snapshots++;
                    snapshot = EventLog.SessionId + "_" + snapshots.ToString("D3");
                }
                yield return detector.DetectTexture(texture, flipCamera, snapshot == null ? null : SnapshotPath(snapshot),
                                                    !asyncCameraReadback, keyResult);
                LogResult(keyResult, "camera", null, snapshot, info);
                if (keyResult.Completed) blockCompleted++;
                StatusLine = CameraStatus(index, scan, hover, snapshot != null);
            }
            finally
            {
                inFlight = false;
                keyHandle = null;
            }
        }

        private void Release()
        {
            if (keyHandle != null)
            {
                StopCoroutine(keyHandle);
                keyHandle = null;
                DisposeStack(keyStack);
            }
            bool pending = inFlight;
            inFlight = false;
            if (detector == null) return;
            string backend = detector.Backend;
            var clock = Stopwatch.StartNew();
            detector.Dispose();
            detector = null;
            var sb = new StringBuilder("{\"backend\":");
            Json.AppendString(sb, backend);
            sb.Append(",\"release_ms\":");
            Json.AppendNumber(sb, (float)clock.Elapsed.TotalMilliseconds);
            sb.Append(",\"in_flight\":").Append(pending ? "true" : "false");
            EventLog.Write("detector.unload", sb.Append('}').ToString());
        }

        // ---------------------------------------------------------------------------------------------- helpers
        private string CameraStatus(int index, int? scan, int? hover, bool snapshot)
        {
            var sb = new StringBuilder(detector.Backend == "gpu_compute" ? "GPU " : "CPU ");
            sb.Append(ChoiceName).Append(" #").Append(index);
            if (scan != null) sb.Append(" scan ").Append(scan).Append('.').Append(hover);
            sb.Append(' ');
            if (!keyResult.Completed) return sb.Append("incomplete: ").Append(keyResult.Error ?? "?").ToString();
            sb.Append(Mathf.RoundToInt((float)keyResult.LatencyMs)).Append(" ms, ");
            float threshold = 0.3f;
            int found = 0;
            var names = detector.Classes;
            foreach (var d in keyResult.Detections)
            {
                if (d.Score < threshold) continue;
                if (found < 2)
                {
                    string name = d.ClassId >= 0 && d.ClassId < names.Count ? names[d.ClassId] : d.ClassId.ToString();
                    sb.Append(found == 0 ? "" : ", ").Append(name).Append(' ').Append(d.Score.ToString("0.00"));
                }
                found++;
            }
            if (found == 0) sb.Append("none");
            if (snapshot) sb.Append(" [snapshot]");
            return sb.ToString();
        }

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

        private static string ImageId(string name)
        {
            int cut = name.IndexOf("_416", StringComparison.Ordinal);
            if (cut < 0) cut = name.IndexOf("_full", StringComparison.Ordinal);
            return cut > 0 ? name.Substring(0, cut) : name;
        }

        private static string SnapshotPath(string snapshot) =>
            Path.Combine(Application.persistentDataPath, "snapshots", snapshot + ".png");

        private static void AppendNullable(StringBuilder sb, string name, int? value)
        {
            sb.Append(",\"").Append(name).Append("\":");
            if (value == null) sb.Append("null"); else sb.Append(value.Value);
        }

        private string LogLayers()
        {
            string joined = string.Join("\n", detector.LayerTypes);
            string hash;
            using (var sha = System.Security.Cryptography.SHA256.Create())
                hash = BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(joined))).Replace("-", "").ToLowerInvariant();
            var sb = new StringBuilder("{\"count\":").Append(detector.LayerCount).Append(",\"types_sha256\":");
            Json.AppendString(sb, hash);
            sb.Append(",\"types\":[");
            for (int k = 0; k < detector.LayerTypes.Count; k++)
            {
                if (k > 0) sb.Append(',');
                Json.AppendString(sb, detector.LayerTypes[k]);
            }
            EventLog.Write("detector.layers", sb.Append("]}").ToString());
            return hash;
        }

        private string ScheduleProblem(ScheduleFile s, DetectorPackage package, string typesHash)
        {
            if (s.model_sha256 != package.model_sha256) return "schedule " + s.schedule_id + " was made for another model";
            if (s.backend != detector.Backend) return "schedule " + s.schedule_id + " was made for " + s.backend;
            if (s.layer_count != detector.LayerCount || s.layer_types_sha256 != typesHash)
                return "schedule " + s.schedule_id + " does not match this runtime's layers";
            int sum = 0;
            foreach (int n in s.slices)
            {
                if (n < 0) return "schedule " + s.schedule_id + " has a negative slice";
                sum += n;
            }
            return sum == detector.LayerCount ? null : "schedule " + s.schedule_id + " covers " + sum + " of " + detector.LayerCount + " steps";
        }

        private void LogSchedule(ScheduleFile s, string problem)
        {
            var sb = new StringBuilder("{\"schedule_id\":");
            Json.AppendString(sb, s.schedule_id);
            sb.Append(",\"frames\":").Append(s.slices.Length).Append(",\"budget_ms\":");
            Json.AppendNumber(sb, s.budget_ms);
            sb.Append(",\"accepted\":").Append(problem == null ? "true" : "false").Append(",\"problem\":");
            if (problem == null) sb.Append("null"); else Json.AppendString(sb, problem);
            EventLog.Write("detector.schedule", sb.Append('}').ToString());
        }

        private void LogProfile(int pass)
        {
            var p = profilePass;
            var sb = new StringBuilder("{\"pass\":").Append(pass).Append(",\"spacing\":").Append(p.Spacing)
                .Append(",\"completed\":").Append(p.Completed ? "true" : "false").Append(",\"error\":");
            if (p.Error == null) sb.Append("null"); else Json.AppendString(sb, p.Error);
            sb.Append(",\"pre_frame\":").Append(p.PreFrame).Append(",\"readback_frame\":").Append(p.ReadbackFrame)
              .Append(",\"done_frame\":").Append(p.DoneFrame).Append(",\"step_frames\":[");
            for (int k = 0; k < p.StepFrames.Count; k++) sb.Append(k > 0 ? "," : "").Append(p.StepFrames[k]);
            EventLog.Write("detector.profile", sb.Append("]}").ToString());
        }

        private void LogSetting(string source)
        {
            StatusLine = OffLine();
            var sb = new StringBuilder("{\"backend\":");
            Json.AppendString(sb, Backend);
            sb.Append(",\"mode\":");
            Json.AppendString(sb, ModeName(mode));
            sb.Append(",\"schedule_id\":");
            if (ChosenSchedule == null) sb.Append("null"); else Json.AppendString(sb, ChosenSchedule.schedule_id);
            sb.Append(",\"steps_per_frame\":").Append(Steps).Append(",\"source\":");
            Json.AppendString(sb, source);
            EventLog.Write("detector.setting", sb.Append('}').ToString());
        }

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
            sb.Append(",\"warmup_completed\":").Append(checkResult.Completed ? "true" : "false");
            sb.Append(",\"color_space\":");
            Json.AppendString(sb, QualitySettings.activeColorSpace == ColorSpace.Linear ? "linear" : "gamma");
            sb.Append(",\"layer_count\":").Append(detector.LayerCount);
            sb.Append(",\"steps_per_frame\":").Append(detector.StepsPerFrame);
            sb.Append(",\"floor\":");
            Json.AppendNumber(sb, detector.Floor);
            sb.Append(",\"mode\":");
            Json.AppendString(sb, ModeName(mode));
            sb.Append(",\"keyframe_rate_hz\":");
            Json.AppendNumber(sb, mode == Mode.Burst ? burstRateHz : mode == Mode.Diagnostic ? diagnosticRateHz : keyframeRateHz);
            sb.Append(",\"schedule_id\":");
            if (activeSchedule == null) sb.Append("null"); else Json.AppendString(sb, activeSchedule.schedule_id);
            EventLog.Write("detector.load", sb.Append('}').ToString());
        }

        private void LogScan(int scan, int? hover, string state, int requested, int completed, int rejected)
        {
            var sb = new StringBuilder("{\"scan\":").Append(scan);
            AppendNullable(sb, "hover", hover);
            sb.Append(",\"state\":");
            Json.AppendString(sb, state);
            sb.Append(",\"requested\":").Append(requested).Append(",\"completed\":").Append(completed)
              .Append(",\"rejected\":").Append(rejected);
            EventLog.Write("detector.scan", sb.Append('}').ToString());
        }

        private void LogReject(int? scan, int? hover, long requestedUtc, string reason)
        {
            var sb = new StringBuilder("{\"reason\":");
            Json.AppendString(sb, reason);
            AppendNullable(sb, "scan", scan);
            AppendNullable(sb, "hover", hover);
            sb.Append(",\"requested_utc_us\":").Append(requestedUtc);
            EventLog.Write("detector.reject", sb.Append('}').ToString());
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

        private void LogResult(DetectionResult result, string path, string imageId, string snapshot, KeyInfo info)
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
              .Append(",\"steps_per_frame\":").Append(result.StepsPerFrame)
              .Append(",\"schedule_steps\":").Append(result.ScheduleSteps)
              .Append(",\"schedule_frames\":").Append(result.ScheduleFrames)
              .Append(",\"start_frame\":").Append(result.StartFrame)
              .Append(",\"schedule_end_frame\":").Append(result.ScheduleEndFrame)
              .Append(",\"end_frame\":").Append(result.EndFrame)
              .Append(",\"preprocess_ms\":");
            Json.AppendNumber(sb, (float)result.PreprocessMs);
            sb.Append(",\"schedule_id\":");
            if (activeSchedule == null || path != "camera") sb.Append("null"); else Json.AppendString(sb, activeSchedule.schedule_id);
            sb.Append(",\"floor\":");
            Json.AppendNumber(sb, result.Floor);
            AppendNullable(sb, "keyframe", info.Keyframe);
            AppendNullable(sb, "scan", info.Scan);
            AppendNullable(sb, "hover", info.Hover);
            sb.Append(",\"requested_utc_us\":");
            if (info.RequestedUtc == null) sb.Append("null"); else sb.Append(info.RequestedUtc.Value);
            sb.Append(",\"started_utc_us\":").Append(info.StartedUtc).Append(",\"capture_stamp\":");
            if (info.Stamp == null) sb.Append("null"); else Json.AppendString(sb, info.Stamp);
            sb.Append(",\"pose\":");
            if (!info.HasPose) sb.Append("null");
            else
            {
                var p = info.Pose.position;
                var q = info.Pose.rotation;
                float[] v = { p.x, p.y, p.z, q.x, q.y, q.z, q.w };
                sb.Append('[');
                for (int k = 0; k < v.Length; k++)
                {
                    if (k > 0) sb.Append(',');
                    Json.AppendNumber(sb, v[k]);
                }
                sb.Append(']');
            }
            sb.Append(",\"candidates\":").Append(result.Candidates)
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
