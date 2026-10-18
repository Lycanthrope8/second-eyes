using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Meta.XR;
using Meta.XR.MRUtilityKit;
using Meta.XR.EnvironmentDepth;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
using SecondEyes.Logging;
using SecondEyes.Perception.Frames;
using SecondEyes.Perception.Localization;
using SecondEyes.Perception.Scan;
using UnityEngine;
using UnityEngine.XR;

namespace SecondEyes.Perception.AutoScan
{
    /// <summary>
    /// The headset's automatic scan (A4; docs/auto-scan.md): the Unity glue around AutoScan. It installs itself when the app
    /// starts, registers with ScanRegistry, and scans with no button: it loads the perception folder, switches the detector
    /// to on-request mode, measures the camera's model, starts Meta's depth (asking for the spatial-data permission), and
    /// runs the detector every interval_s while the head is tracked and no command is being processed
    /// (ScanRegistry.HoldUntil). Each mapped detection at or above min_score is sampled with Meta's environment raycasts at
    /// five points of its box. The frame, every sample included, is one line of files/scans/(scan ID)/frames.jsonl, and
    /// AutoScan turns it into objects. A changed scene is published at most every publish_interval_s: scene.json, trace.json
    /// and facts.json are rewritten, the panel binds it to the session, and a label floats over each object. The panel's
    /// scan button pauses and resumes it.
    /// </summary>
    public sealed class AutoScanner : MonoBehaviour, IRoomScanner
    {
        private const double ReadyTimeoutS = 30, FrameTimeoutS = 10, PermissionTimeoutS = 60;
        private const int MaxLabels = 60;
        private const float LabelLiftM = 0.12f, LabelHeightM = 0.05f;

        public string Phase { get; private set; } = "idle";   // idle, preparing, scanning, paused or failed
        public string StatusLine { get; private set; } = "";
        public string HudText => Phase == "scanning" && scan != null
            ? "SCAN: " + scan.Confirmed + (scan.Confirmed == 1 ? " object" : " objects") + " so far"
              + (scan.Tentative > 0 ? ", " + scan.Tentative + " seen once" : "") + "\nLook around the room; Pause on the panel stops it"
            : Phase == "preparing" ? "SCAN: " + (StatusLine.StartsWith("Scan: ", StringComparison.Ordinal) ? StatusLine.Substring(6) : StatusLine) : "";
        public string LastCaptureId { get; private set; }
        public static string ScansDir => Path.Combine(Application.persistentDataPath, "scans");
        private static string PerceptionDir => Path.Combine(Application.persistentDataPath, "perception");

        private DetectorRunner runner;
        private PerceptionFolder folder;
        private AutoScanSettings settings;
        private AutoScan scan;
        private EnvironmentRaycastManager raycaster;
        private MRUK mruk;
        private static bool spaceSetupDone;   // Space Setup runs once per app launch: a restarted scan reloads the room
        private Shader seeThrough;
        private Transform boxRoot;
        private readonly List<Transform> boxViews = new List<Transform>();
        private readonly Dictionary<string, Material> boxMaterials = new Dictionary<string, Material>();
        private const float EdgeM = 0.008f, FillAlpha = 0.18f, EdgeAlpha = 0.9f;
        private RoomModel room = RoomModel.None("not loaded");
        private List<JNode> classes;
        private HashSet<string> mapped;
        private double fx, fy, cx, cy;
        private int texW, texH, frameCount, revision, savedImages, starts;
        private string scanId, dir;
        private StreamWriter frames;
        private ScanResult latest;
        private JNode camera;
        private bool pending, saveImages;
        private double pendingSince, nextRun, nextPublish;
        private long epoch;
        private float labelScale;
        private readonly List<XRInputSubsystem> subsystems = new List<XRInputSubsystem>();
        private readonly List<TextMesh> labels = new List<TextMesh>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Install()
        {
            var go = new GameObject("A4 auto scanner");
            DontDestroyOnLoad(go);
            ScanRegistry.Scanner = go.AddComponent<AutoScanner>();
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
            EventLog.Write("autoscan.tracking_origin", "{\"epoch\":" + epoch.ToString(CultureInfo.InvariantCulture) + "}");
        }

        private void Start() { StartCoroutine(Begin()); }

        private void OnApplicationPause(bool paused) { if (paused && scan != null && scan.Changed) Publish(); }

        private void OnDestroy()
        {
            if (frames != null) { frames.Dispose(); frames = null; }
        }

        /// <summary>Resumes a paused scan, or starts again after a failure.</summary>
        public bool TryStart(PerceptionFolder f, out string why)
        {
            why = null;
            if (Phase == "paused")
            {
                Phase = "scanning";
                StatusLine = "Scan resumed";
                EventLog.Write("autoscan.resume", "{\"capture_id\":" + Json(scanId) + "}");
                return true;
            }
            if (Phase == "idle" || Phase == "failed") { StartCoroutine(Begin()); return true; }
            why = "the scan is " + Phase;
            return false;
        }

        /// <summary>Pauses the scan; what it found stays in the session.</summary>
        public void Finish()
        {
            if (Phase != "scanning") return;
            Phase = "paused";
            StatusLine = "Scan paused: press Resume to go on";
            Publish();
            EventLog.Write("autoscan.pause", "{\"capture_id\":" + Json(scanId) + "}");
        }

        public ScanResult TakeResult()
        {
            ScanResult r = latest;
            latest = null;
            return r;
        }

        private IEnumerator Begin()
        {
            Phase = "preparing";
            StatusLine = "Scan: starting...";
            if (!PerceptionFolder.Present(PerceptionDir)) { Fail("no perception folder on the headset: push it from the PC"); yield break; }
            try { folder = PerceptionFolder.Load(PerceptionDir); }
            catch (Exception e) { Fail("the perception folder didn't load: " + e.Message); yield break; }
            if (folder.AutoScanConfig == null) { Fail("the perception folder has no automatic scan settings: push the new folder"); yield break; }
            settings = AutoScanSettings.From(folder.AutoScanConfig);
            mapped = new HashSet<string>(folder.CategoryMap["entries"].Items.Select(e => e["source_label"].Text), StringComparer.Ordinal);

            // Meta's depth and Meta's room both need the spatial data permission.
            double until;
            if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene))
            {
                StatusLine = "Scan: allow spatial data when the headset asks";
                OVRPermissionsRequester.Request(new[] { OVRPermissionsRequester.Permission.Scene });
                until = Time.realtimeSinceStartupAsDouble + PermissionTimeoutS;
                while (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene))
                {
                    if (Time.realtimeSinceStartupAsDouble > until) { Fail("the spatial data permission wasn't granted, so Meta's depth can't be used"); yield break; }
                    yield return null;
                }
            }
            yield return LoadRoom();   // Space Setup first, while nothing else runs: the app pauses during it

            StatusLine = "Scan: starting the camera, detector and depth...";
            until = Time.realtimeSinceStartupAsDouble + ReadyTimeoutS;
            while ((runner = FindFirstObjectByType<DetectorRunner>()) == null || runner.FrameSource == null)
            {
                if (Time.realtimeSinceStartupAsDouble > until) { Fail("no detector or camera in this scene"); yield break; }
                yield return null;
            }
            if (runner.IsOn)
            {
                runner.SetOn(false, "a4_autoscan");
                until = Time.realtimeSinceStartupAsDouble + ReadyTimeoutS;
                while ((runner.IsOn || runner.IsBusy) && Time.realtimeSinceStartupAsDouble < until) yield return null;
            }
            until = Time.realtimeSinceStartupAsDouble + ReadyTimeoutS;
            while (!runner.SetMode(DetectorRunner.Mode.External, "a4_autoscan"))   // refused while an inference runs: wait for it to end
            {
                if (Time.realtimeSinceStartupAsDouble > until) { Fail("the detector stayed busy for 30 s, so it couldn't switch to on-request mode"); yield break; }
                yield return null;
            }
            runner.SetOn(true, "a4_autoscan");
            until = Time.realtimeSinceStartupAsDouble + ReadyTimeoutS;
            bool retried = false;
            while (!runner.ReadyForScan)   // after Space Setup the camera can take longer to come back: one restart, then 60 s in all
            {
                if (Time.realtimeSinceStartupAsDouble > until)
                {
                    if (retried) { Fail("the camera or detector didn't start within 60 s, one restart included"); yield break; }
                    retried = true;
                    EventLog.Write("autoscan.camera_retry", "{\"waited_s\":30}");
                    runner.SetOn(false, "a4_autoscan");
                    yield return null;
                    runner.SetOn(true, "a4_autoscan");
                    until = Time.realtimeSinceStartupAsDouble + ReadyTimeoutS;
                }
                yield return null;
            }
            JNode manifest = JParse.Parse(runner.ManifestText);
            classes = manifest["classes"].Items;
            string why;
            if (!MeasureCamera(out why)) { Fail(why); yield break; }
            if (!EnvironmentRaycastManager.IsSupported) { Fail("this headset doesn't support Meta's depth"); yield break; }
            if (raycaster == null)
            {
                var depthObject = new GameObject("A4 depth");
                DontDestroyOnLoad(depthObject);
                EnvironmentDepthManager depth = depthObject.AddComponent<EnvironmentDepthManager>();
                depth.OcclusionShadersMode = OcclusionShadersMode.None;   // depth for raycasts only, not for rendering
                depth.RemoveHands = true;   // hands in view aren't surfaces
                raycaster = depthObject.AddComponent<EnvironmentRaycastManager>();
            }

            starts++;
            scanId = ("scan." + (EventLog.SessionId ?? "nosession") + "." + starts.ToString(CultureInfo.InvariantCulture)).ToLowerInvariant();
            dir = Path.Combine(ScansDir, scanId);
            Directory.CreateDirectory(dir);
            saveImages = settings.SaveImagesMax > 0;
            int deleted = DeleteOldImages();   // each session keeps only its own images
            scan = new AutoScan(scanId, settings, folder.CategoryMap, folder.SizePriors, manifest["package_id"].Text, Sha256Hex(runner.ManifestBytes),
                                ("quest_" + runner.FrameSource.CameraName).ToLowerInvariant(), fx);
            scan.SetRoom(room);
            if (frames != null) frames.Dispose();
            frames = new StreamWriter(Path.Combine(dir, "frames.jsonl"), false, new UTF8Encoding(false)) { AutoFlush = true };
            frames.WriteLine(PyJson.Canonical(scan.Header(folder.ManifestSha256, texW, texH, fy, cx, cy, (int)manifest["input"]["size"].Num)));
            frames.WriteLine(PyJson.Canonical(scan.Room.ToJson()));   // the room exactly as the scan holds it, for the replay
            frameCount = 0; revision = 0; savedImages = 0; pending = false;
            nextRun = nextPublish = Time.realtimeSinceStartupAsDouble;
            Write("facts.json", Facts());
            EventLog.Write("autoscan.start", "{\"capture_id\":" + Json(scanId) + ",\"perception_folder\":" + Json(folder.ManifestSha256)
                           + ",\"config\":" + Json(settings.ConfigId) + ",\"camera\":" + PyJson.Canonical(camera)
                           + ",\"old_images_deleted\":" + deleted.ToString(CultureInfo.InvariantCulture) + "}");
            EventLog.Write("autoscan.room", "{\"capture_id\":" + Json(scanId) + ",\"status\":" + Json(room.Status) + ",\"why\":" + Json(room.Why)
                           + ",\"boxes\":" + room.Boxes.Count.ToString(CultureInfo.InvariantCulture) + "}");
            Phase = "scanning";
            StatusLine = room.Loaded ? "Scanning: look around the room; Meta's room has " + room.Boxes.Count + " boxes"
                                     : "Scanning by depth only (" + room.Why + "): look around the room";
        }

        /// <summary>Meta's room (docs/auto-scan.md): Space Setup when the settings ask for it, then the room MRUK loads.
        /// MRUK is made inactive, given its settings (it reads them when it wakes) and its world lock turned off, so every
        /// position stays in the tracking space the camera's poses and the depth's points are in. No room: the scan goes
        /// on by depth alone, and says why.</summary>
        private IEnumerator LoadRoom()
        {
            room = RoomModel.None("space_setup is never");
            if (settings.SpaceSetup == "never") yield break;
            if (settings.SpaceSetup == "every_start" && !spaceSetupDone)
            {
                spaceSetupDone = true;
                StatusLine = "Scan: Meta's Space Setup is open: look around until it has the walls and furniture, then finish";
                OVRTask<bool> setup = OVRScene.RequestSpaceSetup();
                while (!setup.IsCompleted) yield return null;
                EventLog.Write("autoscan.space_setup", "{\"ok\":" + (setup.GetResult() ? "true" : "false") + "}");
            }
            if (mruk == null)
            {
                var go = new GameObject("A4 Meta room");
                go.SetActive(false);
                DontDestroyOnLoad(go);
                mruk = go.AddComponent<MRUK>();
                mruk.SceneSettings = new MRUK.MRUKSettings { DataSource = MRUK.SceneDataSource.Device, LoadSceneOnStartup = false };
                mruk.EnableWorldLock = false;
                go.SetActive(true);
            }
            StatusLine = "Scan: loading Meta's room...";
            Task<MRUK.LoadDeviceResult> load = mruk.LoadSceneFromDevice(settings.SpaceSetup == "if_missing");
            while (!load.IsCompleted) yield return null;
            if (load.IsFaulted || load.IsCanceled)
            {
                room = RoomModel.None("loading Meta's room failed: " + (load.Exception != null ? load.Exception.GetBaseException().Message : "cancelled"));
                yield break;
            }
            if (load.Result != MRUK.LoadDeviceResult.Success) { room = RoomModel.None("no room from Meta: " + load.Result); yield break; }
            MRUKRoom r = mruk.GetCurrentRoom() ?? mruk.Rooms.FirstOrDefault();
            room = r != null ? FromMeta(r) : RoomModel.None("Meta loaded no room");
        }

        /// <summary>A Meta room as the scan's room model, in quest_tracking: the floor's outline and heights, and every
        /// anchor with a volume as a box, its centre, rotation and size as MRUK gives them.</summary>
        private static RoomModel FromMeta(MRUKRoom r)
        {
            var m = new RoomModel { Status = "loaded" };
            MRUKAnchor floor = r.FloorAnchor, ceiling = r.CeilingAnchor;
            if (floor != null)
            {
                m.FloorY = floor.transform.position.y;
                foreach (Vector2 p in floor.PlaneBoundary2D)
                {
                    Vector3 w = floor.transform.TransformPoint(new Vector3(p.x, p.y, 0));
                    m.FloorXz.Add(new double[] { w.x, w.z });
                }
            }
            m.CeilingY = ceiling != null ? ceiling.transform.position.y : m.FloorY + 3;
            foreach (MRUKAnchor a in r.Anchors)
            {
                if (!a.VolumeBounds.HasValue) continue;
                Bounds b = a.VolumeBounds.Value;
                Vector3 c = a.GetAnchorCenter();
                Quaternion q = a.transform.rotation;
                string label = a.Label.ToString();
                int comma = label.IndexOf(',');
                m.Boxes.Add(new RoomBox { Id = a.Anchor.Uuid.ToString("N"), MetaLabel = (comma < 0 ? label : label.Substring(0, comma)).Trim(),
                                          Centre = new Vec3(c.x, c.y, c.z), Size = new Vec3(b.size.x, b.size.y, b.size.z),
                                          Rotation = new double[] { q.x, q.y, q.z, q.w } });
            }
            return m;
        }

        /// <summary>Deletes the frame images of earlier sessions (files/snapshots/*.png); returns how many.</summary>
        private static int DeleteOldImages()
        {
            string images = Path.Combine(Application.persistentDataPath, "snapshots");
            if (!Directory.Exists(images)) return 0;
            int n = 0;
            foreach (string f in Directory.GetFiles(images, "*.png"))
            {
                try { File.Delete(f); n++; }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
            return n;
        }

        private bool MeasureCamera(out string why)
        {
            why = null;
            PassthroughFrameSource fs = runner.FrameSource;
            double rfx, rfy, rcx, rcy;
            int sw, sh;
            string read;
            Texture texture = fs.CurrentTexture;
            if (!fs.TryGetIntrinsics(out rfx, out rfy, out rcx, out rcy, out sw, out sh, out read) || texture == null)
            {
                why = "the camera's intrinsics couldn't be read (" + read + ")";
                return false;
            }
            texW = texture.width; texH = texture.height;
            double kfx, kfy, kcx, kcy;
            CameraFit.Crop(rfx, rfy, rcx, rcy, sw, sh, texW, texH, out kfx, out kfy, out kcx, out kcy);
            CameraFit.Result fit = CameraFit.Fit(fs.SampleViewportRays(), texW, texH);
            bool useFit = fit != null && fit.RmsPx < 2.0;
            fx = useFit ? fit.Fx : kfx; fy = useFit ? fit.Fy : kfy; cx = useFit ? fit.Cx : kcx; cy = useFit ? fit.Cy : kcy;
            camera = JNode.Obj().Add("camera_name", JNode.Str(fs.CameraName)).Add("intrinsics_read", JNode.Str(read))
                .Add("sensor_px", JNode.Arr(new List<JNode> { Localizer.Int(sw), Localizer.Int(sh) })).Add("texture_px", JNode.Arr(new List<JNode> { Localizer.Int(texW), Localizer.Int(texH) }))
                .Add("crop_intrinsics", Four(kfx, kfy, kcx, kcy)).Add("fitted_intrinsics", fit == null ? JNode.Null : Four(fit.Fx, fit.Fy, fit.Cx, fit.Cy))
                .Add("fit_rms_px", fit == null ? JNode.Null : Localizer.Num(fit.RmsPx)).Add("viewport_rays", JNode.Str(fs.ViewportRayNote ?? ""))
                .Add("intrinsics_used", JNode.Str(useFit ? "fitted" : "crop")).Add("texture_intrinsics", Four(fx, fy, cx, cy))
                .Add("tracking_state_reported", new JNode { Kind = JKind.Bool, Bool = TrackingReported() });
            return true;
        }

        private void Update()
        {
            FaceLabels();
            if (Phase != "scanning" || scan == null) return;
            double now = Time.realtimeSinceStartupAsDouble;
            if (pending && now - pendingSince > FrameTimeoutS)
            {
                pending = false;
                EventLog.Write("autoscan.frame_timeout", "{\"capture_id\":" + Json(scanId) + "}");
            }
            if (scan.Changed && now >= nextPublish) Publish();
            StatusLine = "Scanning: " + scan.Confirmed + " objects, " + scan.Tentative + " seen once, " + scan.Frames + " frames"
                         + (now < ScanRegistry.HoldUntil ? " (waiting for the command)" : "");
            if (pending || now < nextRun || now < ScanRegistry.HoldUntil || runner.IsBusy) return;
            bool tracked = HeadTracked();
            long epochAtStart = epoch;
            bool snapshot = saveImages && savedImages < settings.SaveImagesMax;
            if (runner.RunScanKeyframe(snapshot, f => OnFrame(f, epochAtStart, tracked)))
            {
                pending = true;
                pendingSince = now;
                nextRun = now + settings.IntervalS;
                if (snapshot) savedImages++;
            }
        }

        private void OnFrame(ScanFrame f, long epochAtStart, bool trackedAtStart)
        {
            pending = false;
            if (scan == null || f == null) return;
            if (!f.Completed || f.Error != null || f.TranslationM == null || f.RotationXyzw == null)
            {
                EventLog.Write("autoscan.frame_error", "{\"capture_id\":" + Json(scanId) + ",\"error\":" + Json(f.Error ?? "the frame wasn't completed") + "}");
                return;
            }
            var rec = new FrameRecord
            {
                Frame = ++frameCount, TimeUs = f.CopyMonoUs, Epoch = epochAtStart, Tracked = trackedAtStart, Snapshot = f.Snapshot,
                Position = new Vec3(f.TranslationM[0], f.TranslationM[1], f.TranslationM[2]),
                Rotation = new Quat(f.RotationXyzw[0], f.RotationXyzw[1], f.RotationXyzw[2], f.RotationXyzw[3])
            };
            var origin = new Vector3((float)f.TranslationM[0], (float)f.TranslationM[1], (float)f.TranslationM[2]);
            var rotation = new Quaternion((float)f.RotationXyzw[0], (float)f.RotationXyzw[1], (float)f.RotationXyzw[2], (float)f.RotationXyzw[3]);
            foreach (RawDetection d in f.Detections)
            {
                string label = d.ClassId >= 0 && d.ClassId < classes.Count ? classes[d.ClassId].Text : "class_" + d.ClassId.ToString(CultureInfo.InvariantCulture);
                var fd = new FrameDetection { ClassId = d.ClassId, Label = label, Score = d.Score, X1 = d.X1, Y1 = d.Y1, X2 = d.X2, Y2 = d.Y2 };
                if (trackedAtStart && mapped.Contains(label) && d.Score >= settings.MinScore)   // only detections the scan can keep are raycast
                    foreach (double[] px in AutoScan.SamplePixels(d.X1, d.Y1, d.X2, d.Y2, settings))
                    {
                        Vec3 local = AutoScan.CameraDirection(px[0], px[1], fx, fy, cx, cy);
                        Vector3 world = rotation * new Vector3((float)local.X, (float)local.Y, (float)local.Z);
                        EnvironmentRaycastHit hit;
                        bool ok = raycaster.Raycast(new Ray(origin, world), out hit, (float)(settings.MaxRangeM + 1.0));
                        fd.Samples.Add(new DepthSample { U = px[0], V = px[1], Status = Status(hit.status),
                                                         Point = ok ? new Vec3(hit.point.x, hit.point.y, hit.point.z) : (Vec3?)null });
                    }
                rec.Detections.Add(fd);
            }
            frames.WriteLine(PyJson.Canonical(rec.ToJson()));
            FrameOutcome o = scan.Add(rec);
            if (rec.Detections.Count > 0)
                EventLog.Write("autoscan.frame", "{\"capture_id\":" + Json(scanId) + ",\"frame\":" + rec.Frame.ToString(CultureInfo.InvariantCulture)
                               + ",\"detections\":" + rec.Detections.Count.ToString(CultureInfo.InvariantCulture)
                               + ",\"kept\":" + o.Fates.Count(x => x == "kept").ToString(CultureInfo.InvariantCulture) + "}");
        }

        private void Publish()
        {
            if (scan == null) return;
            nextPublish = Time.realtimeSinceStartupAsDouble + settings.PublishIntervalS;
            revision++;
            LocalizationOutput o = scan.Snapshot(revision);
            var r = new ScanResult { Scene = o.Scene, Trace = o.Trace, Facts = Facts() };
            r.SceneIssues = new Contract(folder.Schema).Validate(new List<KeyValuePair<string, JNode>> {
                new KeyValuePair<string, JNode>("scene", o.Scene), new KeyValuePair<string, JNode>("map", folder.CategoryMap) });
            Write("scene.json", o.Scene);
            Write("trace.json", o.Trace);
            Write("facts.json", r.Facts);
            LastCaptureId = scanId + ".r" + revision.ToString(CultureInfo.InvariantCulture);
            latest = r;
            ShowLabels();
            if (settings.Boxes) ShowBoxes();
            EventLog.Write("autoscan.publish", "{\"capture_id\":" + Json(scanId) + ",\"snapshot\":" + Json(LastCaptureId)
                           + ",\"objects\":" + o.Scene["objects"].Items.Count.ToString(CultureInfo.InvariantCulture)
                           + ",\"frames\":" + scan.Frames.ToString(CultureInfo.InvariantCulture) + ",\"ok\":" + (r.Ok ? "true" : "false") + "}");
        }

        private JNode Facts()
        {
            return JNode.Obj().Add("record_type", JNode.Str("auto_scan_facts")).Add("capture_id", JNode.Str(scanId)).Add("config_id", JNode.Str(settings.ConfigId))
                .Add("perception_folder_sha256", JNode.Str(folder.ManifestSha256)).Add("camera", camera)
                .Add("depth_supported", new JNode { Kind = JKind.Bool, Bool = EnvironmentRaycastManager.IsSupported })
                .Add("save_images", new JNode { Kind = JKind.Bool, Bool = saveImages }).Add("images_saved", Localizer.Int(savedImages))
                .Add("frames", Localizer.Int(scan == null ? 0 : scan.Frames)).Add("sightings", Localizer.Int(scan == null ? 0 : scan.Sightings))
                .Add("objects", Localizer.Int(scan == null ? 0 : scan.Confirmed)).Add("tentative", Localizer.Int(scan == null ? 0 : scan.Tentative))
                .Add("scene_revision", Localizer.Int(revision))
                .Add("room", JNode.Obj().Add("status", JNode.Str(room.Status)).Add("why", JNode.Str(room.Why)).Add("boxes", Localizer.Int(room.Boxes.Count)))
                .Add("boxes_drawn", new JNode { Kind = JKind.Bool, Bool = settings != null && settings.Boxes })
                .Add("see_through_shader", new JNode { Kind = JKind.Bool, Bool = Resources.Load<Shader>("SecondEyesSeeThrough") != null });
        }

        private void ShowLabels()
        {
            List<KeyValuePair<string, Vec3>> objs = scan.ConfirmedInTracking();
            while (labels.Count < Math.Min(objs.Count, MaxLabels)) labels.Add(NewLabel());
            for (int i = 0; i < labels.Count; i++)
            {
                bool on = i < objs.Count;
                labels[i].gameObject.SetActive(on);
                if (!on) continue;
                labels[i].text = objs[i].Key;
                Vec3 p = objs[i].Value;
                labels[i].transform.position = new Vector3((float)p.X, (float)p.Y + LabelLiftM, (float)p.Z);
            }
        }

        /// <summary>v3: every object of the scene so far as a see-through block with bright edges (docs/auto-scan.md),
        /// coloured by who named it: Meta blue, the scan green, unnamed grey.</summary>
        private void ShowBoxes()
        {
            if (seeThrough == null) seeThrough = Resources.Load<Shader>("SecondEyesSeeThrough");
            if (seeThrough == null) return;   // facts.json says so (see_through_shader)
            if (boxRoot == null)
            {
                var root = new GameObject("A4 boxes");
                DontDestroyOnLoad(root);
                boxRoot = root.transform;
            }
            List<AutoScan.ShownBox> boxes = scan.ConfirmedBoxes();
            while (boxViews.Count < Math.Min(boxes.Count, MaxLabels)) boxViews.Add(NewBoxView());
            for (int i = 0; i < boxViews.Count; i++)
            {
                bool on = i < boxes.Count;
                boxViews[i].gameObject.SetActive(on);
                if (on) PlaceBox(boxViews[i], boxes[i]);
            }
        }

        private Transform NewBoxView()
        {
            var view = new GameObject("A4 box").transform;
            view.SetParent(boxRoot, false);
            for (int k = 0; k < 13; k++)   // the fill, then the 12 edges
            {
                GameObject part = GameObject.CreatePrimitive(PrimitiveType.Cube);
                Destroy(part.GetComponent<Collider>());   // nothing to hit with the controller's ray
                part.transform.SetParent(view, false);
            }
            return view;
        }

        private void PlaceBox(Transform view, AutoScan.ShownBox b)
        {
            view.position = new Vector3((float)b.Box.Centre.X, (float)b.Box.Centre.Y, (float)b.Box.Centre.Z);
            view.rotation = Quaternion.Euler(0f, (float)b.Box.YawDeg, 0f);
            float sx = (float)b.Box.Size.X, sy = (float)b.Box.Size.Y, sz = (float)b.Box.Size.Z;
            Color c = b.Kind == "meta" ? new Color(0.25f, 0.7f, 1f, 1f) : b.Kind == "unnamed" ? new Color(0.75f, 0.75f, 0.75f, 1f) : new Color(0.3f, 0.9f, 0.5f, 1f);
            Transform fill = view.GetChild(0);
            fill.localPosition = Vector3.zero;
            fill.localScale = new Vector3(sx, sy, sz);
            fill.GetComponent<MeshRenderer>().sharedMaterial = BoxMaterial(b.Kind + "/fill", new Color(c.r, c.g, c.b, FillAlpha));
            Material edge = BoxMaterial(b.Kind + "/edge", new Color(c.r, c.g, c.b, EdgeAlpha));
            int n = 1;
            foreach (int ax in new[] { 0, 1, 2 })   // 4 edges along each axis, at the corners of the other two
                foreach (int s1 in new[] { -1, 1 })
                    foreach (int s2 in new[] { -1, 1 })
                    {
                        Transform e = view.GetChild(n++);
                        e.localPosition = ax == 0 ? new Vector3(0, s1 * sy / 2, s2 * sz / 2) : ax == 1 ? new Vector3(s1 * sx / 2, 0, s2 * sz / 2) : new Vector3(s1 * sx / 2, s2 * sy / 2, 0);
                        e.localScale = ax == 0 ? new Vector3(sx, EdgeM, EdgeM) : ax == 1 ? new Vector3(EdgeM, sy, EdgeM) : new Vector3(EdgeM, EdgeM, sz);
                        e.GetComponent<MeshRenderer>().sharedMaterial = edge;
                    }
        }

        private Material BoxMaterial(string key, Color colour)
        {
            Material m;
            if (boxMaterials.TryGetValue(key, out m)) return m;
            m = new Material(seeThrough);
            m.SetColor("_Color", colour);
            boxMaterials[key] = m;
            return m;
        }

        private TextMesh NewLabel()
        {
            var go = new GameObject("A4 object label");
            DontDestroyOnLoad(go);
            TextMesh t = go.AddComponent<TextMesh>();
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            t.fontSize = 64;
            t.characterSize = 1f;
            t.anchor = TextAnchor.LowerCenter;
            t.color = Color.white;
            go.GetComponent<MeshRenderer>().sharedMaterial = t.font.material;
            if (labelScale > 0) go.transform.localScale = Vector3.one * labelScale;
            return t;
        }

        /// <summary>Turns each label to face the head; TextMesh sizes aren't in meters, so the first label sets the scale.</summary>
        private void FaceLabels()
        {
            Camera c = Camera.main;
            if (c == null) return;
            foreach (TextMesh t in labels)
            {
                if (!t.gameObject.activeSelf) continue;
                if (labelScale <= 0)
                {
                    float h = t.GetComponent<MeshRenderer>().bounds.size.y;
                    if (h > 0)
                    {
                        labelScale = LabelHeightM / h;
                        foreach (TextMesh u in labels) u.transform.localScale = Vector3.one * labelScale;
                    }
                }
                t.transform.rotation = Quaternion.LookRotation(t.transform.position - c.transform.position);
            }
        }

        private void Fail(string why)
        {
            Phase = "failed";
            StatusLine = "The scan stopped: " + why;
            EventLog.Write("autoscan.failed", "{\"capture_id\":" + Json(scanId) + ",\"why\":" + Json(why) + "}");
        }

        private void Write(string name, JNode node)
        {
            File.WriteAllText(Path.Combine(dir, name), PyJson.Canonical(node) + "\n", new UTF8Encoding(false));
        }

        private static string Status(EnvironmentRaycastHitStatus s)
        {
            switch (s)
            {
                case EnvironmentRaycastHitStatus.Hit: return "hit";
                case EnvironmentRaycastHitStatus.HitPointOccluded: return "hit_point_occluded";
                case EnvironmentRaycastHitStatus.NotReady: return "not_ready";
                case EnvironmentRaycastHitStatus.HitPointOutsideOfCameraFrustum: return "outside_depth_view";
                case EnvironmentRaycastHitStatus.RayOccluded: return "ray_occluded";
                case EnvironmentRaycastHitStatus.NoHit: return "no_hit";
                default: return "not_supported";
            }
        }

        private static bool TrackingReported()
        {
            InputDevice head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            InputTrackingState s;
            return head.isValid && head.TryGetFeatureValue(CommonUsages.trackingState, out s);
        }

        private static bool HeadTracked()
        {
            InputDevice head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            InputTrackingState s;
            if (!head.isValid || !head.TryGetFeatureValue(CommonUsages.trackingState, out s)) return true;
            const InputTrackingState both = InputTrackingState.Position | InputTrackingState.Rotation;
            return (s & both) == both;
        }

        private static string Json(string s) { return PyJson.Canonical(JNode.Str(s ?? "")); }

        private static JNode Four(double a, double b, double c, double d)
        {
            return JNode.Arr(new List<JNode> { Localizer.Num(a), Localizer.Num(b), Localizer.Num(c), Localizer.Num(d) });
        }

        private static string Sha256Hex(byte[] b)
        {
            if (b == null) return new string('0', 64);
            using (var h = SHA256.Create()) return string.Concat(h.ComputeHash(b).Select(x => x.ToString("x2")));
        }
    }
}
