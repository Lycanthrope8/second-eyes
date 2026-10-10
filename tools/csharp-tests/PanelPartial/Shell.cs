// Type-check shell for ChatPanel.Session.cs: the panel members it uses, with their types in ChatPanel.cs.
namespace UnityEngine.Events { public delegate void UnityAction(); }
namespace UnityEngine.UI
{
    public class Button { public bool interactable; }
    public class Text { public string text; public int fontSize; }
}
namespace SecondEyes.Grounding
{
    using UnityEngine.UI;
    public partial class ChatPanel
    {
        private Text boxText, status;
        private Button loadButton;
        private bool replaying, panelModelUsed;
        private string text = "";
        private string ggufFile = "qwen2.5-0.5b-instruct-q8_0.gguf";
        private Button RowButton(string name, float x, float width, out Text label, UnityEngine.Events.UnityAction onClick, float y = 362f)
        { label = new Text(); return new Button(); }
        private void SetStatus(string value) { }
        private void Fail(string where, string message) { }
        private void UpdateReplayButton() { }
    }
}
