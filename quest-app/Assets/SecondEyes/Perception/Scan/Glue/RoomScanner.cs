using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Logging;
using SecondEyes.Perception.Capture;
using SecondEyes.Perception.Frames;
using UnityEngine;
using UnityEngine.XR;

namespace SecondEyes.Perception.Scan
{
    /// <summary>
    /// The automatic scan on the headset (A4 Delivery 1, stage 3; docs/scan.md): the Unity side of ScanSession. It installs
    /// itself when the app starts and registers with ScanRegistry. A scan switches the detector into its on-request mode,
    /// reads the camera's intrinsics (scaled from the sensor's pixels to the texture's) and the camera's API for the facts,
    /// then feeds the head's pose and tracking to the session each frame, starting a keyframe whenever the session says so.
    /// At Finish it waits for the last keyframe, writes facts.json, capture.json, scene.json and trace.json under
    /// files/scans/(capture ID)/, and puts the detector back as it found it. Keyframe images (the model's input) are saved
    /// only when files/scans/save-images exists.
    /// </summary>
    public sealed class RoomScanner : MonoBehaviour, IRoomScanner
    {
        private const double ReadyTimeoutS = 30, KeyframeTimeoutS = 20, FinishWaitS = 20;

        public string Phase { get; private set; } = "idle";
        public string StatusLine { get; private set; } = "";
        public string HudText => Phase == "scanning" && session != null ? ScanGuide.Text(session.Rule, session.Kept, session.Failed)
                               : Phase == "preparing" ? "SCAN: starting the camera and detector..." : Phase == "finishing" ? "SCAN: finishing..." : "";
        public string LastCaptureId { get; private set; }
        public static string ScansDir => Path.Combine(Application.persistentDataPath, "scans");

        private DetectorRunner runner;
        private PerceptionFolder folder;
        private ScanSession session;
        private ScanResult finished;
        private DetectorRunner.Mode previousMode;
        private bool runnerWasOn, changedRunner, pending, saveImages;
        private double pendingSince;
        private long epoch;
        private int scans;
        private readonly List<XRInputSubsystem> subsystems = new List<XRInputSubsystem>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Install()
        {
            var go = new GameObject("A4 room scanner");
            DontDestroyOnLoad(go);
            ScanRegistry.Scanner = go.AddComponent<RoomScanner>();
        }

        private void OnEnable()
        {
            SubsystemManager.GetSubsystems(subsystems);
            foreach (XRInputSubsystem s in subsystems) s.trackingOriginUpdated += OnTrackingOrigin;
        }

        private void OnDisable()
        {
            foreach (XRInputSubsystem s in subsystems) s.trackingOriginUpdated -= OnTrackingOrigin;
        }

        private void OnTrackingOrigin(XRInputSubsystem s)
        {
            epoch++;
            EventLog.Write("scan.tracking_origin", "{\"epoch\":" + epoch.ToString(CultureInfo.InvariantCulture) + "}");
        }

        public bool TryStart(PerceptionFolder f, out string why)
        {
            why = null;
            if (Phase != "idle") { why = "a scan is already " + Phase; return false; }
            if (f == null) { why = "no perception folder"; return false; }
            runner = FindFirstObjectByType<DetectorRunner>();
            if (runner == null) { why = "no detector in this scene"; return false; }
            if (runner.FrameSource == null) { why = "the detector has no camera"; return false; }
            folder = f;
            scans++;
            LastCaptureId = ("scan." + (EventLog.SessionId ?? "nosession") + "." + scans.ToString(CultureInfo.InvariantCulture)).ToLowerInvariant();
            saveImages = File.Exists(Path.Combine(ScansDir, "save-images"));
            finished = null;
            Phase = "preparing";
            StatusLine = "Scan: starting the camera and detector...";
            StartCoroutine(Prepare());
            return true;
        }

        public void Finish()
        {
            if (Phase != "scanning") return;
            Phase = "finishing";
            StartCoroutine(Complete());
        }

        public ScanResult TakeResult()
        {
            ScanResult r = finished;
            finished = null;
            return r;
        }

        private IEnumerator Prepare()
        {
            runnerWasOn = runner.IsOn;
            previousMode = runner.CurrentMode;
            changedRunner = true;
            if (runner.IsOn)
            {
                runner.SetOn(false, "a4_scan");
                double until0 = Time.realtimeSinceStartupAsDouble + ReadyTimeoutS;
                while ((runner.IsOn || runner.IsBusy) && Time.realtimeSinceStartupAsDouble < until0) yield return null;
            }
            if (!runner.SetMode(DetectorRunner.Mode.External, "a4_scan")) { yield return Stop("the detector couldn't switch to on-request mode"); yield break; }
            runner.SetOn(true, "a4_scan");
            double until = Time.realtimeSinceStartupAsDouble + ReadyTimeoutS;
            while (!runner.ReadyForScan)
            {
                if (Time.realtimeSinceStartupAsDouble > until) { yield return Stop("the camera or detector didn't start within 30 s"); yield break; }
                yield return null;
            }
            PassthroughFrameSource fs = runner.FrameSource;
            string api = fs.DescribeApi();
            double fx, fy, cx, cy;
            int sw, sh;
            string read;
            Texture texture = fs.CurrentTexture;
            if (!fs.TryGetIntrinsics(out fx, out fy, out cx, out cy, out sw, out sh, out read) || texture == null)
            {
                EventLog.Write("scan.camera_api", "{\"api\":" + Json(api) + ",\"intrinsics\":" + Json(read) + "}");
                yield return Stop("the camera's intrinsics couldn't be read (" + read + ")");
                yield break;
            }
            int tw = texture.width, th = texture.height;
            // The camera's model in the texture's pixels: fitted from the camera's own viewport rays when that fits a pinhole
            // to within 2 pixels, otherwise the reported intrinsics carried over by the centre-crop model (Meta's sample code).
            double kfx, kfy, kcx, kcy;
            CameraFit.Crop(fx, fy, cx, cy, sw, sh, tw, th, out kfx, out kfy, out kcx, out kcy);
            List<double[]> samples = fs.SampleViewportRays();
            CameraFit.Result fit = CameraFit.Fit(samples, tw, th);
            bool useFit = fit != null && fit.RmsPx < 2.0;
            var cam = new CaptureCamera
            {
                CameraId = ("quest_" + fs.CameraName).ToLowerInvariant(), WidthPx = tw, HeightPx = th, Distortion = "unknown",
                Fx = useFit ? fit.Fx : kfx, Fy = useFit ? fit.Fy : kfy, Cx = useFit ? fit.Cx : kcx, Cy = useFit ? fit.Cy : kcy,
                IntrinsicsSource = useFit
                    ? "fitted from the camera's ViewportPointToRay at " + fit.Samples + " viewport points (viewport origin " + fit.Origin + ", RMS "
                      + fit.RmsPx.ToString("0.00", CultureInfo.InvariantCulture) + " px)"
                    : "the camera component's " + read + ", carried from the sensor's " + sw + "x" + sh + " pixels to the texture's " + tw + "x" + th
                      + " by the centre-crop model" + (fit == null ? " (no viewport rays to fit)" : " (the fit's RMS was " + fit.RmsPx.ToString("0.00", CultureInfo.InvariantCulture) + " px)")
            };
            Transform rig = Camera.main != null ? Camera.main.transform.parent : null;
            JNode facts = JNode.Obj().Add("api", JNode.Str(api)).Add("intrinsics_read", JNode.Str(read))
                .Add("sensor_px", Pair(sw, sh)).Add("texture_px", Pair(tw, th))
                .Add("sensor_intrinsics", Four(fx, fy, cx, cy)).Add("crop_intrinsics", Four(kfx, kfy, kcx, kcy))
                .Add("fitted_intrinsics", fit == null ? JNode.Null : Four(fit.Fx, fit.Fy, fit.Cx, fit.Cy))
                .Add("fit", fit == null ? JNode.Null : JNode.Obj().Add("rms_px", Num(fit.RmsPx)).Add("origin", JNode.Str(fit.Origin))
                    .Add("samples", Localization.Localizer.Int(fit.Samples)))
                .Add("intrinsics_used", JNode.Str(useFit ? "fitted" : "crop")).Add("texture_intrinsics", Four(cam.Fx, cam.Fy, cam.Cx, cam.Cy))
                .Add("camera_name", JNode.Str(fs.CameraName)).Add("tracking_state_reported", Bool(TrackingReported()))
                .Add("rig_position", rig == null ? JNode.Null : Three(rig.position.x, rig.position.y, rig.position.z))
                .Add("rig_rotation_xyzw", rig == null ? JNode.Null : Four(rig.rotation.x, rig.rotation.y, rig.rotation.z, rig.rotation.w))
                .Add("save_images", Bool(saveImages));
            byte[] manifest = runner.ManifestBytes;
            session = new ScanSession(LastCaptureId, folder, JParse.Parse(runner.ManifestText), Sha256Hex(manifest), "second eyes " + Application.version);
            session.SetCamera(cam, facts);
            EventLog.Write("scan.start", "{\"capture_id\":" + Json(LastCaptureId) + ",\"perception_folder\":" + Json(folder.ManifestSha256)
                           + ",\"camera\":" + PyJson.Canonical(facts) + "}");
            Phase = "scanning";
        }

        private void Update()
        {
            if (Phase != "scanning" || session == null) return;
            if (pending && Time.realtimeSinceStartupAsDouble - pendingSince > KeyframeTimeoutS)
            {
                pending = false;
                EventLog.Write("scan.keyframe_timeout", "{\"capture_id\":" + Json(LastCaptureId) + "}");
            }
            Camera c = Camera.main;
            if (c == null) { StatusLine = "Scan: no head camera"; return; }
            Vector3 p = c.transform.position, fwd = c.transform.forward;
            double yaw = Math.Atan2(fwd.x, fwd.z) * 180.0 / Math.PI;
            bool tracked = HeadTracked();
            if (session.Update(Time.realtimeSinceStartupAsDouble, new Vec3(p.x, p.y, p.z), yaw, tracked, pending || runner.IsBusy))
            {
                int rulePosition = session.Rule.PendingPosition;   // known before the inference starts, so its result can't race it
                long epochAtStart = epoch;
                if (runner.RunScanKeyframe(saveImages, frame => OnFrame(frame, rulePosition, epochAtStart, tracked)))
                {
                    session.Started();
                    StartCoroutine(Buzz());   // a short pulse on the right controller: this view is taken
                    pending = true;
                    pendingSince = Time.realtimeSinceStartupAsDouble;
                }
            }
            if (session.Error != null)
            {
                Phase = "finishing";
                StartCoroutine(Complete());
                return;
            }
            StatusLine = "Scan: " + session.Rule.Hint + " | spot " + (Math.Max(session.Rule.Position, 0) + 1) + ", " + session.Kept + " keyframes"
                         + (session.Failed > 0 ? ", " + session.Failed + " left out" : "") + " | press Finish when done";
        }

        private void OnFrame(ScanFrame f, int rulePosition, long epochAtStart, bool trackedAtStart)
        {
            pending = false;
            if (session == null) return;
            f.TrackingEpoch = epochAtStart;
            f.Tracked = trackedAtStart;
            bool kept = session.Add(rulePosition, f);
            EventLog.Write("scan.keyframe", "{\"capture_id\":" + Json(LastCaptureId) + ",\"kept\":" + (kept ? "true" : "false")
                           + ",\"kept_total\":" + session.Kept.ToString(CultureInfo.InvariantCulture)
                           + ",\"detections\":" + f.Detections.Count.ToString(CultureInfo.InvariantCulture)
                           + ",\"snapshot\":" + (f.Snapshot == null ? "null" : Json(f.Snapshot)) + "}");
        }

        private static IEnumerator Buzz()
        {
            OVRInput.SetControllerVibration(1f, 0.5f, OVRInput.Controller.RTouch);
            yield return new WaitForSeconds(0.08f);
            OVRInput.SetControllerVibration(0f, 0f, OVRInput.Controller.RTouch);
        }

        private IEnumerator Complete()
        {
            StatusLine = "Scan: finishing...";
            double until = Time.realtimeSinceStartupAsDouble + FinishWaitS;
            while ((pending || runner.IsBusy) && Time.realtimeSinceStartupAsDouble < until) yield return null;
            ScanResult r = session.Finish();
            string dir = Path.Combine(ScansDir, LastCaptureId);
            try
            {
                Directory.CreateDirectory(dir);
                Write(dir, "facts.json", r.Facts);
                if (r.Capture != null) Write(dir, "capture.json", r.Capture);
                if (r.Scene != null) Write(dir, "scene.json", r.Scene);
                if (r.Trace != null) Write(dir, "trace.json", r.Trace);
            }
            catch (Exception e)
            {
                EventLog.Error("scan.files", e.GetType().Name + ": " + e.Message);
            }
            var issues = r.CaptureIssues.Concat(r.SceneIssues).Where(i => i.IsError).Select(i => Json(i.Code + " " + i.Path)).Take(10);
            EventLog.Write("scan.finish", "{\"capture_id\":" + Json(LastCaptureId) + ",\"ok\":" + (r.Ok ? "true" : "false")
                           + ",\"error\":" + (r.Error == null ? "null" : Json(r.Error))
                           + ",\"kept\":" + session.Kept.ToString(CultureInfo.InvariantCulture) + ",\"failed\":" + session.Failed.ToString(CultureInfo.InvariantCulture)
                           + ",\"objects\":" + (r.Scene == null ? "0" : r.Scene["objects"].Items.Count.ToString(CultureInfo.InvariantCulture))
                           + ",\"issues\":[" + string.Join(",", issues) + "],\"folder\":" + Json(dir) + "}");
            yield return Restore();
            session = null;
            finished = r;
            Phase = "idle";
            StatusLine = "";
        }

        private IEnumerator Stop(string why)
        {
            EventLog.Write("scan.stopped", "{\"capture_id\":" + Json(LastCaptureId) + ",\"why\":" + Json(why) + "}");
            yield return Restore();
            session = null;
            finished = new ScanResult { Error = why };
            Phase = "idle";
            StatusLine = "";
        }

        /// <summary>The detector back as the scan found it: its mode, and on or off.</summary>
        private IEnumerator Restore()
        {
            if (!changedRunner || runner == null) yield break;
            changedRunner = false;
            runner.SetOn(false, "a4_scan");
            double until = Time.realtimeSinceStartupAsDouble + ReadyTimeoutS;
            while ((runner.IsOn || runner.IsBusy) && Time.realtimeSinceStartupAsDouble < until) yield return null;
            runner.SetMode(previousMode, "a4_scan");
            if (runnerWasOn) runner.SetOn(true, "a4_scan");
        }

        private static bool TrackingReported()
        {
            InputDevice head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            InputTrackingState s;
            return head.isValid && head.TryGetFeatureValue(CommonUsages.trackingState, out s);
        }

        /// <summary>Whether the head's position and rotation are both tracked. A device that doesn't report its tracking
        /// state counts as tracked (the facts say so), and the capture's own check still refuses any untracked keyframe.</summary>
        private static bool HeadTracked()
        {
            InputDevice head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            InputTrackingState s;
            if (!head.isValid || !head.TryGetFeatureValue(CommonUsages.trackingState, out s)) return true;
            const InputTrackingState both = InputTrackingState.Position | InputTrackingState.Rotation;
            return (s & both) == both;
        }

        private static void Write(string dir, string name, JNode node)
        {
            File.WriteAllText(Path.Combine(dir, name), PyJson.Canonical(node) + "\n", new UTF8Encoding(false));
        }

        private static string Json(string s) { return PyJson.Canonical(JNode.Str(s ?? "")); }   // a JSON string, as records are written

        private static string Sha256Hex(byte[] b)
        {
            if (b == null) return new string('0', 64);
            using (var h = SHA256.Create()) return string.Concat(h.ComputeHash(b).Select(x => x.ToString("x2")));
        }

        private static JNode Bool(bool b) { return new JNode { Kind = JKind.Bool, Bool = b }; }
        private static JNode Num(double v) { return new JNode { Kind = JKind.Number, Num = v, Text = v.ToString("R", CultureInfo.InvariantCulture), IsInteger = false }; }
        private static JNode Pair(int a, int b) { return JNode.Arr(new List<JNode> { Localization.Localizer.Int(a), Localization.Localizer.Int(b) }); }
        private static JNode Three(double a, double b, double c) { return JNode.Arr(new List<JNode> { Num(a), Num(b), Num(c) }); }
        private static JNode Four(double a, double b, double c, double d) { return JNode.Arr(new List<JNode> { Num(a), Num(b), Num(c), Num(d) }); }
    }
}
