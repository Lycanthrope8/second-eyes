// Stand-ins for the Unity calls ChatPanel.Layout.cs, ChatPanel.Prompt.cs and ChatPanel.Session.cs make (A4 Delivery 1): a type check only,
// never run. Each keeps Unity's own name, namespace and member signature, and properties stay properties, so code that
// compiles here uses only what Unity has. Only the members the two files use are here; add one only when a panel file
// starts using it.
namespace UnityEngine
{
    public static class Time { public static double realtimeSinceStartupAsDouble { get { return 0; } } }
    public class Object { }
    public class Component : Object
    {
        public Transform transform { get { return null; } }
        public GameObject gameObject { get { return null; } }
        public T GetComponent<T>() { return default(T); }
    }
    public class Behaviour : Component { }
    public class MonoBehaviour : Behaviour { }
    public class Transform : Component
    {
        public Vector3 position { get; set; }
        public Vector3 localScale { get; set; }
        public void SetParent(Transform parent, bool worldPositionStays) { }
    }
    public sealed class RectTransform : Transform
    {
        public Vector2 anchorMin { get; set; }
        public Vector2 anchorMax { get; set; }
        public Vector2 pivot { get; set; }
        public Vector2 anchoredPosition { get; set; }
        public Vector2 sizeDelta { get; set; }
    }
    public sealed class GameObject : Object
    {
        public GameObject(string name, params System.Type[] components) { }
        public Transform transform { get { return null; } }
        public T GetComponent<T>() { return default(T); }
        public T AddComponent<T>() where T : Component { return null; }
        public void SetActive(bool value) { }
    }
    public struct Vector2 { public Vector2(float x, float y) { } }
    public struct Vector3
    {
        public Vector3(float x, float y, float z) { }
        public static Vector3 one { get { return default(Vector3); } }
        public static Vector3 operator *(Vector3 a, float d) { return a; }
    }
    public struct Color
    {
        public Color(float r, float g, float b) { }
        public Color(float r, float g, float b, float a) { }
        public static Color white { get { return default(Color); } }
    }
    public sealed class Font : Object { }
    public static class Resources { public static T GetBuiltinResource<T>(string path) where T : Object { return null; } }
    public enum FontStyle { Normal, Bold, Italic, BoldAndItalic }
    public enum TextAnchor { UpperLeft, UpperCenter, UpperRight, MiddleLeft, MiddleCenter, MiddleRight, LowerLeft, LowerCenter, LowerRight }
    public enum HorizontalWrapMode { Wrap, Overflow }
    public enum VerticalWrapMode { Truncate, Overflow }
    public enum RenderMode { ScreenSpaceOverlay, ScreenSpaceCamera, WorldSpace }
    public sealed class Canvas : Behaviour { public RenderMode renderMode { get; set; } }
}
namespace UnityEngine.Events
{
    public delegate void UnityAction();
    public class UnityEvent { public void AddListener(UnityAction call) { } }
}
namespace UnityEngine.UI
{
    public class CanvasScaler : MonoBehaviour { public float dynamicPixelsPerUnit { get; set; } }
    public abstract class Graphic : MonoBehaviour { public virtual Color color { get; set; } }
    public abstract class MaskableGraphic : Graphic { }
    public class Image : MaskableGraphic { }
    public class Text : MaskableGraphic
    {
        public virtual string text { get; set; }
        public Font font { get; set; }
        public int fontSize { get; set; }
        public FontStyle fontStyle { get; set; }
        public TextAnchor alignment { get; set; }
        public HorizontalWrapMode horizontalOverflow { get; set; }
        public VerticalWrapMode verticalOverflow { get; set; }
        public bool supportRichText { get; set; }
    }
    public class Selectable : MonoBehaviour
    {
        public bool interactable { get; set; }
        public Graphic targetGraphic { get; set; }
    }
    public class Button : Selectable
    {
        public class ButtonClickedEvent : UnityEngine.Events.UnityEvent { }
        public ButtonClickedEvent onClick { get; set; }
    }
}
