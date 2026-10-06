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
