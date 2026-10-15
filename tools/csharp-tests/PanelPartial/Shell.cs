// Type-check shell for ChatPanel.Session.cs, ChatPanel.Layout.cs and ChatPanel.Prompt.cs: the members of ChatPanel.cs they use, with
// their types there (A2.5 delivery 4; A4 Delivery 1 added the layout and the prompt). The Unity calls they make are in UiStubs.cs.
#pragma warning disable 0169, 0414, 0649   // the stand-in fields exist for the type check only
namespace SecondEyes.Grounding
{
    using System.Collections.Generic;
    using UnityEngine;
    using UnityEngine.UI;

    public partial class ChatPanel : MonoBehaviour
    {
        private class PromptFile { public string id, user; }
        private static readonly Color PresetColor = new Color(0.22f, 0.25f, 0.3f);
        private readonly List<PromptFile> choices = new List<PromptFile>();
        private readonly List<Button> presetButtons = new List<Button>();
        private RectTransform panel;
        private Button sendButton, loadButton, repeatButton, stepsButton, backendButton, weightsButton, runtimeButton;
        private Button replayButton, goldenButton, interactiveButton, parserButton;
        private Text boxText, buttonLabel, loadLabel, repeatLabel, stepsLabel, backendLabel, weightsLabel, status, answer;
        private Text runtimeLabel, replayLabel, goldenLabel, interactiveLabel, parserLabel;
        private bool replaying, panelModelUsed;
        private string text = "";
        private string ggufFile = "qwen2.5-0.5b-instruct-q8_0.gguf";
        private void SetStatus(string value) { }
        private void Fail(string where, string message) { }
        private void UpdateReplayButton() { }
        private void UpdateLabels() { }
        private void Choose(int k) { }
        private static string Command(PromptFile file) { return ""; }
        private void LoadModel() { }
        private void ToggleRepeat() { }
        private void NextSteps() { }
        private void OnButton() { }
        private void ToggleRuntime() { }
        private void ToggleBackend() { }
        private void ToggleWeights() { }
        private void StartReplay() { }
        private void StartGolden() { }
        private void StartInteractive() { }
        private void StartParserCheck() { }
    }
}
