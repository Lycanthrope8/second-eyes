using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Meta.XR;
using SecondEyes.Logging;
using UnityEngine;
using UnityEngine.UI;

namespace SecondEyes.Perception
{
    /// <summary>
    /// The headset's passthrough camera as a frame source (A1.10a), through MRUK's PassthroughCameraAccess (D80,
    /// proposed). The camera starts only when asked (CameraButton: X on the left controller) unless Start On Launch
    /// is set, so one build can measure the app with the camera off and then on. It requests only the
    /// headset-camera permission and logs the camera's state about once per second (docs/logging.md), so the A1
    /// profiling procedure can measure what the camera costs (docs/profiling.md).
    /// It calls only members that Meta's samples use. A per-frame signal (an "updated this frame" flag or a
    /// timestamp) is read through reflection when the component has one; when it has neither, frame counts are
    /// logged as null, never guessed.
    /// </summary>
    public class PassthroughFrameSource : MonoBehaviour, IFrameSource
    {
        [Tooltip("MRUK's Passthrough Camera Access component. Leave it disabled in the scene: this component enables it.")]
        [SerializeField] private PassthroughCameraAccess cameraAccess;

        [Tooltip("Start the camera as soon as the permission is granted. Leave off for measurements (camera off first).")]
        [SerializeField] private bool startOnLaunch;

        [Tooltip("Optional preview of the camera. Showing it costs rendering, so measured runs leave this empty.")]
        [SerializeField] private RawImage preview;

        private PropertyInfo updatedFlag, timestamp;
        private MethodInfo cameraPose;
        private string frameSignal = "none";
        private object lastStamp;
        private bool wanted, permissionKnown, permissionGranted, playingLogged;
        private float windowStart;
        private int windowFrames;
        private long framesSeen;

        public string SourceId => "passthrough";
        public bool IsOn => wanted;
        public bool IsDelivering => cameraAccess != null && cameraAccess.enabled && cameraAccess.IsPlaying;
        public Texture CurrentTexture => IsDelivering ? cameraAccess.GetTexture() : null;
        public long FramesSeen => frameSignal == "none" ? -1 : framesSeen;

        public Vector2Int Resolution
        {
            get
            {
                if (!IsDelivering) return Vector2Int.zero;
                var r = cameraAccess.CurrentResolution;
                return new Vector2Int((int)r.x, (int)r.y);
            }
        }

        private void Awake()
        {
            const BindingFlags flags = BindingFlags.Public | BindingFlags.Instance;
            var type = typeof(PassthroughCameraAccess);
            updatedFlag = type.GetProperty("IsUpdatedThisFrame", flags);
            if (updatedFlag != null && updatedFlag.PropertyType != typeof(bool)) updatedFlag = null;
            timestamp = type.GetProperty("Timestamp", flags);
            frameSignal = updatedFlag != null ? "updated_flag" : timestamp != null ? "timestamp" : "none";
            cameraPose = type.GetMethod("GetCameraPose", flags, null, System.Type.EmptyTypes, null);
            if (cameraPose != null && cameraPose.ReturnType != typeof(Pose)) cameraPose = null;
        }

        private void Start()
        {
            if (cameraAccess == null)
            {
                EventLog.Error("camera", "no PassthroughCameraAccess component is assigned");
                enabled = false;
                return;
            }
            if (cameraAccess.enabled)
            {
                EventLog.Error("camera", "PassthroughCameraAccess was enabled in the scene; it stays off until the camera is turned on");
                cameraAccess.enabled = false;
            }
            LogSupportedResolutions();
            if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.PassthroughCameraAccess))
            {
                OVRPermissionsRequester.Request(new[] { OVRPermissionsRequester.Permission.PassthroughCameraAccess });
            }
            windowStart = Time.realtimeSinceStartup;
            if (startOnLaunch) SetOn(true, "launch");
        }

        /// <summary>Ask for the camera on or off; it starts once the permission is granted.</summary>
        public void SetOn(bool on, string source)
        {
            wanted = on;
            var sb = new StringBuilder("{\"on\":").Append(on ? "true" : "false").Append(",\"source\":");
            Json.AppendString(sb, source);
            EventLog.Write("camera.toggle", sb.Append('}').ToString());
        }

        public void Toggle(string source) => SetOn(!wanted, source);

        /// <summary>
        /// The current frame's capture timestamp, as the camera component reports it (its text form, or null), and the
        /// camera's pose (A1.10, D88). Read in the frame an inference copies its image, so the three stay together.
        /// </summary>
        public bool TryGetCapture(out string stamp, out Pose pose)
        {
            stamp = null;
            pose = default;
            if (!IsDelivering) return false;
            try
            {
                if (timestamp != null)
                {
                    object value = timestamp.GetValue(cameraAccess);
                    if (value is System.DateTime when)
                        stamp = when.ToUniversalTime().ToString("o", System.Globalization.CultureInfo.InvariantCulture);
                    else if (value != null)
                        stamp = System.Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture);
                }
                if (cameraPose == null) return false;
                pose = (Pose)cameraPose.Invoke(cameraAccess, null);
                return true;
            }
            catch (System.Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// A4 (stage 3): the capture timestamp as the camera component holds it (its raw value, or null when it has none),
        /// with the camera's pose, read as TryGetCapture reads them, so a scan's capture record can state the number and
        /// its unit.
        /// </summary>
        public bool TryGetCaptureRaw(out object stamp, out Pose pose)
        {
            stamp = null;
            pose = default;
            if (!IsDelivering) return false;
            try
            {
                if (timestamp != null) stamp = timestamp.GetValue(cameraAccess);
                if (cameraPose == null) return false;
                pose = (Pose)cameraPose.Invoke(cameraAccess, null);
                return true;
            }
            catch (System.Exception)
            {
                return false;
            }
        }

        /// <summary>A4: which camera this is ("left" or "right"), as the component names it, or "unknown".</summary>
        public string CameraName
        {
            get
            {
                object v = Member(cameraAccess, "CameraPosition") ?? Member(cameraAccess, "Eye") ?? Member(cameraAccess, "CameraEye");
                return v == null ? "unknown" : v.ToString().ToLowerInvariant();
            }
        }

        /// <summary>
        /// A4: the camera's intrinsics as the component reports them, read by reflection since MRUK's names differ between
        /// versions: the focal length and principal point in the sensor's pixels, the sensor's resolution, and what was read
        /// from where. False when no intrinsics are found; the scan then stops, rather than guess them.
        /// </summary>
        public bool TryGetIntrinsics(out double fx, out double fy, out double cx, out double cy, out int sensorW, out int sensorH, out string read)
        {
            fx = fy = cx = cy = 0; sensorW = sensorH = 0; read = "none";
            if (cameraAccess == null) return false;
            try
            {
                object intr = Member(cameraAccess, "Intrinsics") ?? Member(cameraAccess, "CameraIntrinsics");
                string from = "Intrinsics";
                if (intr == null)
                {
                    foreach (string name in new[] { "GetCameraIntrinsics", "GetIntrinsics" })
                    {
                        MethodInfo m = cameraAccess.GetType().GetMethod(name, BindingFlags.Public | BindingFlags.Instance, null, System.Type.EmptyTypes, null);
                        if (m != null) { intr = m.Invoke(cameraAccess, null); from = name + "()"; break; }
                    }
                }
                if (intr == null) return false;
                object focal = Member(intr, "FocalLength"), principal = Member(intr, "PrincipalPoint");
                object res = Member(intr, "SensorResolution") ?? Member(intr, "Resolution");
                if (!(focal is Vector2) || !(principal is Vector2)) return false;
                Vector2 f = (Vector2)focal, p = (Vector2)principal;
                fx = f.x; fy = f.y; cx = p.x; cy = p.y;
                if (res is Vector2Int) { var r = (Vector2Int)res; sensorW = r.x; sensorH = r.y; }
                else if (res is Vector2) { var r = (Vector2)res; sensorW = Mathf.RoundToInt(r.x); sensorH = Mathf.RoundToInt(r.y); }
                read = from + " (" + intr.GetType().Name + "): FocalLength, PrincipalPoint" + (res != null ? ", " + (Member(intr, "SensorResolution") != null ? "SensorResolution" : "Resolution") : ", no resolution");
                return sensorW > 0 && sensorH > 0;
            }
            catch (System.Exception e)
            {
                read = "failed: " + e.GetType().Name;
                return false;
            }
        }

        /// <summary>
        /// A4: the camera's own rays for a 3x3 grid of viewport points (0.1, 0.5 and 0.9 on each axis), each direction in
        /// the camera's axes through its pose at the same moment: rows of u, v, x, y, z. They let the scan fit the camera's
        /// model in the texture's pixels instead of assuming one (CameraFit). Null when the component has no
        /// ViewportPointToRay taking a Vector2 or Vector3, or no pose.
        /// </summary>
        public List<double[]> SampleViewportRays()
        {
            if (cameraAccess == null || cameraPose == null) return null;
            MethodInfo m = null;
            foreach (MethodInfo c in cameraAccess.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance))
            {
                if (c.Name != "ViewportPointToRay" || c.ReturnType != typeof(Ray)) continue;
                ParameterInfo[] ps = c.GetParameters();
                if (ps.Length == 1 && (ps[0].ParameterType == typeof(Vector2) || ps[0].ParameterType == typeof(Vector3))) { m = c; break; }
            }
            if (m == null) return null;
            try
            {
                Pose pose = (Pose)cameraPose.Invoke(cameraAccess, null);
                Quaternion toCamera = Quaternion.Inverse(pose.rotation);
                bool two = m.GetParameters()[0].ParameterType == typeof(Vector2);
                var rows = new List<double[]>();
                foreach (float u in new[] { 0.1f, 0.5f, 0.9f })
                    foreach (float v in new[] { 0.1f, 0.5f, 0.9f })
                    {
                        object arg = two ? (object)new Vector2(u, v) : new Vector3(u, v, 0f);
                        Ray r = (Ray)m.Invoke(cameraAccess, new[] { arg });
                        Vector3 d = toCamera * r.direction;
                        rows.Add(new double[] { u, v, d.x, d.y, d.z });
                    }
                return rows;
            }
            catch (System.Exception)
            {
                return null;
            }
        }

        /// <summary>A4: the camera component's public properties and methods, names and types, for the scan's facts.</summary>
        public string DescribeApi()
        {
            if (cameraAccess == null) return "no camera component";
            var sb = new System.Text.StringBuilder(cameraAccess.GetType().FullName).Append(":");
            foreach (PropertyInfo p in cameraAccess.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
                sb.Append(' ').Append(p.Name).Append(':').Append(p.PropertyType.Name);
            foreach (MethodInfo m in cameraAccess.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
                if (!m.IsSpecialName) sb.Append(' ').Append(m.Name).Append("()");
            return sb.ToString();
        }

        private static object Member(object o, string name)
        {
            if (o == null) return null;
            System.Type t = o.GetType();
            PropertyInfo p = t.GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
            if (p != null && p.GetIndexParameters().Length == 0) return p.GetValue(o);
            FieldInfo f = t.GetField(name, BindingFlags.Public | BindingFlags.Instance);
            return f != null ? f.GetValue(o) : null;
        }

        private void Update()
        {
            bool granted = OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.PassthroughCameraAccess);
            if (!permissionKnown || granted != permissionGranted)
            {
                permissionKnown = true;
                permissionGranted = granted;
                EventLog.Write("camera.permission", granted ? "{\"granted\":true}" : "{\"granted\":false}");
            }

            bool run = wanted && granted;
            if (cameraAccess.enabled != run)
            {
                cameraAccess.enabled = run;
                playingLogged = false;
                lastStamp = null;
                if (preview != null) preview.texture = null;
                EventLog.Write("camera.state", run ? "{\"enabled\":true}" : "{\"enabled\":false}");
            }

            bool delivering = IsDelivering;
            if (delivering && !playingLogged)
            {
                playingLogged = true;
                var r = Resolution;
                var sb = new StringBuilder("{\"width\":").Append(r.x).Append(",\"height\":").Append(r.y).Append(",\"frame_signal\":");
                Json.AppendString(sb, frameSignal);
                EventLog.Write("camera.playing", sb.Append('}').ToString());
                if (preview != null) preview.texture = cameraAccess.GetTexture();
            }
            if (delivering && NewFrame())
            {
                framesSeen++;
                windowFrames++;
            }

            float now = Time.realtimeSinceStartup;
            if (now - windowStart >= 1f)
            {
                var r = Resolution;
                var sb = new StringBuilder("{\"on\":").Append(wanted ? "true" : "false")
                    .Append(",\"playing\":").Append(delivering ? "true" : "false")
                    .Append(",\"frames\":").Append(frameSignal == "none" ? "null" : windowFrames.ToString())
                    .Append(",\"window_ms\":").Append(Mathf.RoundToInt((now - windowStart) * 1000f))
                    .Append(",\"width\":").Append(r.x).Append(",\"height\":").Append(r.y).Append('}');
                EventLog.Write("camera.second", sb.ToString());
                windowStart = now;
                windowFrames = 0;
            }
        }

        private bool NewFrame()
        {
            if (updatedFlag != null) return (bool)updatedFlag.GetValue(cameraAccess);
            if (timestamp == null) return false;
            object stamp = timestamp.GetValue(cameraAccess);
            if (Equals(stamp, lastStamp)) return false;
            lastStamp = stamp;
            return true;
        }

        private static void LogSupportedResolutions()
        {
            var sb = new StringBuilder("{\"camera\":\"left\",\"supported\":");
            var list = PassthroughCameraAccess.GetSupportedResolutions(PassthroughCameraAccess.CameraPositionType.Left);
            if (list == null)
            {
                sb.Append("null");
            }
            else
            {
                sb.Append('[');
                bool first = true;
                foreach (var r in list)
                {
                    if (!first) sb.Append(',');
                    first = false;
                    sb.Append('[').Append((int)r.x).Append(',').Append((int)r.y).Append(']');
                }
                sb.Append(']');
            }
            EventLog.Write("camera.resolutions", sb.Append('}').ToString());
        }
    }
}
