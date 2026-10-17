// Stubs of the members PassthroughFrameSource uses, with the real APIs' signatures; nothing here runs.
namespace UnityEngine
{
    public class Object { }
    public class Component : Object { }
    public class Behaviour : Component { public bool enabled { get; set; } }
    public class MonoBehaviour : Behaviour { }
    public class Texture : Object { public int width { get { return 0; } } public int height { get { return 0; } } }
    public struct Vector2 { public float x, y; public Vector2(float x, float y) { this.x = x; this.y = y; } }
    public struct Vector2Int { public int x { get; set; } public int y { get; set; } public Vector2Int(int x, int y) { this.x = x; this.y = y; } public static Vector2Int zero { get { return default(Vector2Int); } } }
    public struct Vector3 { public float x, y, z; public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; } }
    public struct Quaternion
    {
        public float x, y, z, w;
        public static Quaternion Inverse(Quaternion rotation) { return rotation; }
        public static Vector3 operator *(Quaternion rotation, Vector3 point) { return point; }
    }
    public struct Pose { public Vector3 position; public Quaternion rotation; }
    public struct Ray { public Vector3 origin { get; set; } public Vector3 direction { get; set; } }
    public sealed class SerializeField : System.Attribute { }
    public static class Mathf { public static int RoundToInt(float f) { return 0; } }
    public static class Time { public static float realtimeSinceStartup { get { return 0; } } }
    public sealed class TooltipAttribute : System.Attribute { public TooltipAttribute(string tooltip) { } }
}
namespace UnityEngine.UI
{
    public class RawImage : UnityEngine.MonoBehaviour { public UnityEngine.Texture texture { get; set; } }
}
namespace Meta.XR
{
    public class PassthroughCameraAccess : UnityEngine.MonoBehaviour
    {
        public enum CameraPositionType { Left, Right }
        public static UnityEngine.Vector2Int[] GetSupportedResolutions(CameraPositionType cameraPosition) { return null; }
        public bool IsPlaying { get { return false; } }
        public UnityEngine.Texture GetTexture() { return null; }
        public UnityEngine.Vector2Int CurrentResolution { get { return default(UnityEngine.Vector2Int); } }
    }
}
public static class OVRPermissionsRequester
{
    public enum Permission { FaceTracking, BodyTracking, EyeTracking, Scene, RecordAudio, PassthroughCameraAccess }
    public static bool IsPermissionGranted(Permission permission) { return false; }
    public static void Request(System.Collections.Generic.IEnumerable<Permission> permissions) { }
}
namespace SecondEyes.Logging
{
    public static class Json { public static void AppendString(System.Text.StringBuilder sb, string value) { } }
    public static class EventLog
    {
        public static void Write(string ev, string dataJson) { }
        public static void Error(string where, string message) { }
    }
}
