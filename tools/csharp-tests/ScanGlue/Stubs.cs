// Type-check stubs for Perception/Scan/Glue/RoomScanner.cs (A4 Delivery 1, stage 3): the Unity, XR, logging, detector and
// camera members it calls, with the shapes they have in Unity 6 and in this app. They only make the scanner compile here;
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
    }
    public class Component : Object { public Transform transform { get { return null; } } }
    public class Behaviour : Component { }
    public class Coroutine { }
    public class MonoBehaviour : Behaviour { public Coroutine StartCoroutine(IEnumerator routine) { return null; } }
    public class Transform : Component
    {
        public Vector3 position { get; set; }
        public Quaternion rotation { get; set; }
        public Vector3 forward { get { return default(Vector3); } }
        public Transform parent { get { return null; } }
    }
    public sealed class GameObject : Object
    {
        public GameObject(string name) { }
        public T AddComponent<T>() where T : Component { return null; }
    }
    public sealed class Camera : Behaviour { public static Camera main { get { return null; } } }
    public struct Vector3 { public float x, y, z; }
    public struct Quaternion { public float x, y, z, w; }
    public class Texture : Object { public int width { get { return 0; } } public int height { get { return 0; } } }
    public static class Time { public static double realtimeSinceStartupAsDouble { get { return 0; } } }
    public static class Application { public static string persistentDataPath { get { return ""; } } public static string version { get { return ""; } } }
    public enum RuntimeInitializeLoadType { AfterSceneLoad, BeforeSceneLoad }
    [System.AttributeUsage(System.AttributeTargets.Method)]
    public sealed class RuntimeInitializeOnLoadMethodAttribute : System.Attribute { public RuntimeInitializeOnLoadMethodAttribute(RuntimeInitializeLoadType t) { } }
    public static class SubsystemManager { public static void GetSubsystems<T>(List<T> subsystems) { } }
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
