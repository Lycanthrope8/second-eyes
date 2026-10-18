// Type-check stubs for Perception/Scan/Glue/RoomScanner.cs and Perception/AutoScan/Glue/AutoScanner.cs (A4): the Unity, XR,
// Meta XR SDK 207.0.0 (signatures copied from its source), logging, detector and camera members they call, with the
// shapes they have in Unity 6 and in this app. They only make the scanner compile here;
// Unity compiles it against the real ones.
#pragma warning disable 0067, 0649
namespace UnityEngine
{
    using System.Collections;
    using System.Collections.Generic;
    public class Object
    {
        public static T FindFirstObjectByType<T>() where T : Object { return null; }
        public static void DontDestroyOnLoad(Object target) { }
        public static void Destroy(Object obj) { }
    }
    public class Component : Object
    {
        public Transform transform { get { return null; } }
        public GameObject gameObject { get { return null; } }
        public T GetComponent<T>() { return default(T); }
    }
    public class Behaviour : Component { }
    public class Coroutine { }
    public class MonoBehaviour : Behaviour { public Coroutine StartCoroutine(IEnumerator routine) { return null; } }
    public class Transform : Component
    {
        public Vector3 position { get; set; }
        public Quaternion rotation { get; set; }
        public Vector3 forward { get { return default(Vector3); } }
        public Transform parent { get { return null; } }
        public Vector3 localScale { get; set; }
        public Vector3 TransformPoint(Vector3 position) { return position; }
        public Vector3 localPosition { get; set; }
        public void SetParent(Transform parent, bool worldPositionStays) { }
        public Transform GetChild(int index) { return null; }
    }
    public sealed class GameObject : Object
    {
        public GameObject(string name) { }
        public T AddComponent<T>() where T : Component { return null; }
        public T GetComponent<T>() { return default(T); }
        public Transform transform { get { return null; } }
        public bool activeSelf { get { return false; } }
        public void SetActive(bool value) { }
        public static GameObject CreatePrimitive(PrimitiveType type) { return null; }
    }
    public sealed class Camera : Behaviour { public static Camera main { get { return null; } } }
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 one { get { return new Vector3(1, 1, 1); } }
        public static Vector3 zero { get { return default(Vector3); } }
        public static Vector3 operator -(Vector3 a, Vector3 b) { return new Vector3(a.x - b.x, a.y - b.y, a.z - b.z); }
        public static Vector3 operator *(Vector3 a, float d) { return new Vector3(a.x * d, a.y * d, a.z * d); }
    }
    public struct Quaternion
    {
        public float x, y, z, w;
        public Quaternion(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }
        public static Vector3 operator *(Quaternion rotation, Vector3 point) { return point; }
        public static Quaternion LookRotation(Vector3 forward) { return default(Quaternion); }
        public static Quaternion Euler(float x, float y, float z) { return default(Quaternion); }
    }
    public struct Ray { public Ray(Vector3 origin, Vector3 direction) { } }
    public struct Bounds { public Vector3 size { get { return default(Vector3); } } }
    public struct Vector2 { public float x, y; }
    public struct Color
    {
        public float r, g, b, a;
        public Color(float r, float g, float b, float a) { this.r = r; this.g = g; this.b = b; this.a = a; }
        public static Color white { get { return default(Color); } }
    }
    public enum PrimitiveType { Sphere = 0, Capsule = 1, Cylinder = 2, Cube = 3, Plane = 4, Quad = 5 }
    public class Collider : Component { }
    public sealed class Shader : Object { }
    public sealed class Mesh : Object
    {
        public Vector3[] vertices { get { return null; } }
        public int vertexCount { get { return 0; } }
    }
    public enum TextAnchor { UpperLeft, UpperCenter, UpperRight, MiddleLeft, MiddleCenter, MiddleRight, LowerLeft, LowerCenter, LowerRight }
    public class Material : Object
    {
        public Material(Shader shader) { }
        public void SetColor(string name, Color value) { }
    }
    public class Font : Object { public Material material { get { return null; } } }
    public class Renderer : Component { public Bounds bounds { get { return default(Bounds); } } public Material sharedMaterial { get; set; } }
    public class MeshRenderer : Renderer { }
    public sealed class TextMesh : Component
    {
        public string text { get; set; }
        public Font font { get; set; }
        public int fontSize { get; set; }
        public float characterSize { get; set; }
        public TextAnchor anchor { get; set; }
        public Color color { get; set; }
    }
    public static class Resources
    {
        public static T GetBuiltinResource<T>(string path) where T : Object { return null; }
        public static T Load<T>(string path) where T : Object { return null; }
    }
    public class Texture : Object { public int width { get { return 0; } } public int height { get { return 0; } } }
    public static class Time { public static double realtimeSinceStartupAsDouble { get { return 0; } } }
    public static class Application { public static string persistentDataPath { get { return ""; } } public static string version { get { return ""; } } }
    public enum RuntimeInitializeLoadType { AfterSceneLoad, BeforeSceneLoad }
    [System.AttributeUsage(System.AttributeTargets.Method)]
    public sealed class RuntimeInitializeOnLoadMethodAttribute : System.Attribute { public RuntimeInitializeOnLoadMethodAttribute(RuntimeInitializeLoadType t) { } }
    public static class SubsystemManager { public static void GetSubsystems<T>(List<T> subsystems) { } }
    public sealed class WaitForSeconds { public WaitForSeconds(float seconds) { } }
}
namespace Meta.XR
{
    public class EnvironmentRaycastManager : UnityEngine.MonoBehaviour
    {
        public static bool IsSupported { get { return false; } }
        public bool Raycast(UnityEngine.Ray ray, out EnvironmentRaycastHit hit, float maxDistance = 100f) { hit = default(EnvironmentRaycastHit); return false; }
    }
    public struct EnvironmentRaycastHit
    {
        public EnvironmentRaycastHitStatus status;
        public UnityEngine.Vector3 point;
        public UnityEngine.Vector3 normal;
        public float normalConfidence;
    }
    public enum EnvironmentRaycastHitStatus { Hit, HitPointOccluded, NotReady, HitPointOutsideOfCameraFrustum, RayOccluded, NoHit, NotSupported }
}
namespace Meta.XR.MRUtilityKit
{
    using System.Collections.Generic;
    using System.Threading.Tasks;
    public class MRUK : UnityEngine.MonoBehaviour
    {
        public enum SceneDataSource { Device, Prefab, DeviceWithPrefabFallback }
        public enum SceneModel { V1 }
        public enum LoadDeviceResult { Success = 0, NoScenePermission = 1 }
        public partial class MRUKSettings
        {
            public SceneDataSource DataSource = SceneDataSource.Device;
            public bool LoadSceneOnStartup = true;
        }
        public MRUKSettings SceneSettings;
        public bool EnableWorldLock = true;
        public List<MRUKRoom> Rooms { get { return null; } }
        public MRUKRoom GetCurrentRoom() { return null; }
        public Task<LoadDeviceResult> LoadSceneFromDevice(bool requestSceneCaptureIfNoDataFound = true, bool removeMissingRooms = true, SceneModel sceneModel = SceneModel.V1) { return null; }
    }
    public class MRUKRoom : UnityEngine.MonoBehaviour
    {
        public List<MRUKAnchor> Anchors { get { return null; } }
        public MRUKAnchor FloorAnchor { get { return null; } }
        public MRUKAnchor CeilingAnchor { get { return null; } }
        public MRUKAnchor GlobalMeshAnchor { get { return null; } }
    }
    public class MRUKAnchor : UnityEngine.MonoBehaviour
    {
        [System.Flags] public enum SceneLabels { FLOOR = 1, CEILING = 2, WALL_FACE = 4, TABLE = 8, COUCH = 16, OTHER = 32, STORAGE = 64, BED = 128, SCREEN = 256 }
        public SceneLabels Label { get; internal set; }
        public UnityEngine.Bounds? VolumeBounds { get; internal set; }
        public List<UnityEngine.Vector2> PlaneBoundary2D { get; internal set; }
        public OVRAnchor Anchor { get; internal set; }
        public UnityEngine.Vector3 GetAnchorCenter() { return default(UnityEngine.Vector3); }
        public UnityEngine.Mesh GlobalMesh { get { return null; } }
    }
}
namespace Meta.XR.EnvironmentDepth
{
    public class EnvironmentDepthManager : UnityEngine.MonoBehaviour
    {
        public OcclusionShadersMode OcclusionShadersMode { get; set; }
        public bool RemoveHands { get; set; }
    }
    public enum OcclusionShadersMode { None, HardOcclusion, SoftOcclusion }
}
public readonly struct OVRAnchor { public System.Guid Uuid { get { return default(System.Guid); } } }
public readonly struct OVRTask<TResult>
{
    public bool IsCompleted { get { return true; } }
    public TResult GetResult() { return default(TResult); }
}
public static class OVRScene { public static OVRTask<bool> RequestSpaceSetup() { return default(OVRTask<bool>); } }
public static class OVRPermissionsRequester
{
    public enum Permission { FaceTracking, BodyTracking, EyeTracking, Scene, RecordAudio, PassthroughCameraAccess }
    public static bool IsPermissionGranted(Permission permission) { return false; }
    public static void Request(System.Collections.Generic.IEnumerable<Permission> permissions) { }
}
public static class OVRInput
{
    public enum Controller { LTouch, RTouch }
    public static void SetControllerVibration(float frequency, float amplitude, Controller controllerMask) { }
}
namespace UnityEngine.XR
{
    public enum XRNode { Head }
    [System.Flags] public enum InputTrackingState { None = 0, Position = 1, Rotation = 2 }
    public struct InputFeatureUsage<T> { }
    public static class CommonUsages { public static InputFeatureUsage<InputTrackingState> trackingState; }
    public struct InputDevice
    {
        public bool isValid { get { return false; } }
        public bool TryGetFeatureValue(InputFeatureUsage<InputTrackingState> usage, out InputTrackingState value) { value = 0; return false; }
    }
    public static class InputDevices { public static InputDevice GetDeviceAtXRNode(XRNode node) { return default(InputDevice); } }
    public class XRInputSubsystem { public event System.Action<XRInputSubsystem> trackingOriginUpdated; }
}
namespace SecondEyes.Logging
{
    public static class EventLog
    {
        public static string SessionId { get { return ""; } }
        public static void Write(string ev, string dataJson) { }
        public static void Error(string where, string message) { }
        public static long NowMonoUs() { return 0; }
    }
    public static class Json { public static void AppendString(System.Text.StringBuilder sb, string value) { } }
}
namespace SecondEyes.Perception
{
    using System;
    using SecondEyes.Perception.Scan;
    using UnityEngine;
    public class PassthroughFrameSource : MonoBehaviour
    {
        public Texture CurrentTexture { get { return null; } }
        public string CameraName { get { return ""; } }
        public bool TryGetIntrinsics(out double fx, out double fy, out double cx, out double cy, out int sensorW, out int sensorH, out string read)
        { fx = fy = cx = cy = 0; sensorW = sensorH = 0; read = ""; return false; }
        public string DescribeApi() { return ""; }
        public System.Collections.Generic.List<double[]> SampleViewportRays() { return null; }
        public string ViewportRayNote { get; private set; } = "";
    }
    public class DetectorRunner : MonoBehaviour
    {
        public enum Mode { Scan, Sequential, Diagnostic, Profile, Burst, Continuous, External }
        public bool IsOn { get { return false; } }
        public bool IsBusy { get { return false; } }
        public void SetOn(bool on, string source) { }
        public bool SetMode(Mode m, string source) { return true; }
        public Mode CurrentMode { get { return Mode.Scan; } }
        public bool ReadyForScan { get { return false; } }
        public PassthroughFrameSource FrameSource { get { return null; } }
        public string ManifestText { get { return null; } }
        public byte[] ManifestBytes { get { return null; } }
        public bool RunScanKeyframe(bool snapshot, Action<ScanFrame> done) { return false; }
    }
}
