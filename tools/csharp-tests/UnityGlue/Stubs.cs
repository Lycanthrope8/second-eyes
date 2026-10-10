// Stand-ins for the Unity and event-log calls the replay glue uses: a type check only, never run.
namespace UnityEngine
{
    public static class Application { public static string persistentDataPath = "/tmp", version = "0", unityVersion = "0"; }
    public static class SystemInfo { public static string deviceModel = "stub"; }
    public static class JsonUtility { public static T FromJson<T>(string json) { throw new System.NotImplementedException(); } }
}
namespace SecondEyes.Logging
{
    public static class EventLog
    {
        public static string SessionId { get; private set; }
        public static string FilePath { get; private set; }
        public static void Write(string ev, string dataJson) { }
    }
}
namespace SecondEyes.Logging
{
    public static class Json { public static void AppendString(System.Text.StringBuilder sb, string value) { } }   // the real one exists
}
