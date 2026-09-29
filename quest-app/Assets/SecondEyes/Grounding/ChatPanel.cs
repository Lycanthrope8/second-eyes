using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using Meta.XR.BuildingBlocks.AIBlocks;
using SecondEyes.Logging;
using UnityEngine;
using UnityEngine.UI;
using LogJson = SecondEyes.Logging.Json;   // alias: Meta's namespace could have its own Json
using Stopwatch = System.Diagnostics.Stopwatch;

namespace SecondEyes.Grounding
{
    /// <summary>
    /// A1.7c-2's test panel (D40, D43, D49). At startup it loads the on-device model and shows, about a meter ahead, the
    /// prompt's user text, a row of preset buttons (the fixed prompt and the presets, each a prompt file like
    /// grounding/prompts/a17-fixed.json), a Send button and the answer. There is no typing: the Quest's system keyboard
    /// doesn't appear in this setup (r015), and presets cost the least memory (D49). Send runs the shown text through
    /// Meta's provider, and while it runs the button is Stop. Each step is logged as it happens, so an unfinished answer
    /// still leaves data: model.request (the prompt's ID and token IDs, from the provider's own template and tokenizer),
    /// model.token (each piece of the answer and its time) and model.generate (the end). grounding/check_headset.py
    /// compares a run with the PC references. In the Unity editor's Play mode, right-click the component and choose
    /// "Next preset" or "Send or Stop" to run it without a headset; the answer then also goes to the Console. The right
    /// thumbstick's click puts the panel in front again.
    /// </summary>
    public class ChatPanel : MonoBehaviour
    {
        [Tooltip("Meta's provider asset, filled by Second Eyes > Fill chat provider.")]
        [SerializeField] private UnityInferenceEngineProvider provider;
        [Tooltip("The fixed prompt, a copy of grounding/prompts/a17-fixed.json; its user text fills the text box.")]
        [SerializeField] private TextAsset prompt;
        [Tooltip("More prompts, each a copy of a file in grounding/prompts/; each gets a button next to the fixed prompt's.")]
        [SerializeField] private TextAsset[] presets = new TextAsset[0];
        [Tooltip("Where the panel appears relative to the head, in meters: right, up, forward.")]
        [SerializeField] private Vector3 offsetM = new Vector3(0f, -0.1f, 1.0f);

        private const string TypedId = "typed";
        private static readonly string[] MetaPrefixes =
            { "[TextOnlyLLMRunner]", "[UnityInferenceEngineProvider]", "[GPT2Tokenizer]", "[OnDeviceLLMConfig]" };
        private static readonly Color PresetColor = new Color(0.22f, 0.25f, 0.3f), ChosenColor = new Color(0.3f, 0.45f, 0.6f);

#pragma warning disable 0649   // filled by JsonUtility
        [Serializable] private class PromptFile { public string id, user; }
#pragma warning restore 0649

        private OnDeviceLlmConfig config;
        private readonly List<PromptFile> choices = new List<PromptFile>();   // the fixed prompt first, then the presets
        private readonly List<Button> presetButtons = new List<Button>();
        private int chosen;
        private RectTransform panel;
        private Button sendButton;
        private Text boxText, buttonLabel, status, answer;
        private string text = "";
        private CancellationTokenSource running;
        private int requests;
        private bool loaded, placed;

        private void OnEnable() { Application.logMessageReceived += OnLog; }

        private void OnDisable() { Application.logMessageReceived -= OnLog; }

        private async void Start()
        {
            foreach (TextAsset asset in new[] { prompt }.Concat(presets ?? new TextAsset[0]))
            {
                PromptFile parsed = asset != null ? JsonUtility.FromJson<PromptFile>(asset.text) : null;
                if (parsed != null && !string.IsNullOrEmpty(parsed.id) && parsed.user != null) choices.Add(parsed);
            }
            BuildPanel();
            Choose(0);
            if (Application.isEditor)
            {
                Debug.Log("[ChatPanel] This session's event log: " + EventLog.FilePath);
            }
            if (provider == null)
            {
                Fail("setup", "No provider asset is assigned to ChatPanel.");
                return;
            }
            config = Field<OnDeviceLlmConfig>(provider, "llmConfig");   // Meta keeps this field internal
            if (config == null)
            {
                Fail("setup", "Can't read the provider's chat settings (llmConfig). Meta's SDK may have changed.");
                return;
            }

            string file = Field<string>(provider, "streamingAssetFileName");
            string source = Application.streamingAssetsPath;
            bool copies = !string.IsNullOrEmpty(file) && (source.Contains("://") || source.Contains("jar:"))
                          && !File.Exists(Path.Combine(Application.persistentDataPath, file));
            SetStatus(copies ? "Loading model (first start: copying it out of the app first)..." : "Loading model...");
            var watch = Stopwatch.StartNew();
            try
            {
                await provider.WarmUp();
            }
            catch (Exception e)
            {
                Fail("model.load", e.Message);
                return;
            }
            double ms = watch.Elapsed.TotalMilliseconds;

            var data = new StringBuilder("{\"file\":");
            LogJson.AppendString(data, file);
            data.Append(",\"copied\":").Append(copies ? "true" : "false");
            data.Append(",\"ms\":").Append(Number(ms));
            data.Append(",\"backend\":");
            LogJson.AppendString(data, config.backendType.ToString());
            data.Append(",\"execution_mode\":");
            LogJson.AppendString(data, config.inferenceExecutionMode.ToString());
            data.Append(",\"steps_per_frame\":").Append(config.stepsPerFrame.ToString(CultureInfo.InvariantCulture)).Append('}');
            EventLog.Write("model.load", data.ToString());

            loaded = true;
            sendButton.interactable = true;
            SetStatus($"Ready. The model loaded in {ms / 1000:F1} s.");
            if (Application.isEditor)
            {
                Debug.Log($"[ChatPanel] Ready. The model loaded in {ms / 1000:F1} s.");
            }
        }

        private void LateUpdate()
        {
            if (!placed && Time.timeSinceLevelLoad > 0.5f)   // wait until the head pose is tracked
            {
                placed = Place();
            }
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick, OVRInput.Controller.RTouch))
            {
                Place();
            }
        }

        [ContextMenu("Next preset")]
        private void NextPresetFromInspector()
        {
            if (!Application.isPlaying || choices.Count == 0) return;
            Choose((chosen + 1) % choices.Count);
            Debug.Log("[ChatPanel] Preset " + choices[chosen].id + ": " + Command(choices[chosen]));
        }

        /// <summary>Shows choice k's user text; Send then sends it under that prompt's ID.</summary>
        private void Choose(int k)
        {
            if (running != null || k < 0 || k >= choices.Count)
            {
                if (choices.Count == 0) SetText("");
                return;
            }
            chosen = k;
            SetText(choices[k].user);
            for (int i = 0; i < presetButtons.Count; i++)
                presetButtons[i].targetGraphic.color = i == k ? ChosenColor : PresetColor;
        }

        /// <summary>The prompt's command line, without "Command: ", for its button.</summary>
        private static string Command(PromptFile file)
        {
            string last = file.user.Split('\n').Last();
            return last.StartsWith("Command: ", StringComparison.Ordinal) ? last.Substring("Command: ".Length) : file.id;
        }

        [ContextMenu("Send or Stop")]
        private void SendOrStopFromInspector()
        {
            if (!Application.isPlaying)
            {
                Debug.LogWarning("[ChatPanel] Enter Play mode first.");
                return;
            }
            OnButton();
        }

        private void OnButton()
        {
            if (running != null)
            {
                running.Cancel();
                SetStatus("Stopping...");
                return;
            }
            Send();
        }

        private async void Send()
        {
            if (!loaded || string.IsNullOrWhiteSpace(text))
            {
                if (Application.isEditor)
                {
                    Debug.Log(loaded ? "[ChatPanel] The text box is empty." : "[ChatPanel] The model hasn't loaded yet.");
                }
                return;
            }
            int request = ++requests;
            string sent = text;
            PromptFile match = choices.FirstOrDefault(c => c.user == sent);
            string promptId = match != null ? match.id : TypedId;
            running = new CancellationTokenSource();
            buttonLabel.text = "Stop";
            foreach (Button b in presetButtons) b.interactable = false;
            answer.text = "";
            SetStatus("Generating...");
            var stream = new AnswerStream(this, request);
            ChatResponse response = null;
            try
            {
                // The same two steps Meta's runner takes, so these are exactly the IDs the model sees.
                string formatted = config.ApplyChatTemplate(sent, config.defaultSystemMessage);
                List<int> ids = await config.Tokenizer.EncodeAsync(formatted);
                LogRequest(request, promptId, ids);
                stream.Watch.Restart();
                response = await provider.ChatAsync(new ChatRequest(sent), stream, running.Token);
            }
            catch (OperationCanceledException)
            {
                // stopped before the model started; the end is logged below
            }
            catch (Exception e) when (running.IsCancellationRequested)
            {
                // Stopping mid-pass makes Meta's runner throw (a NullReferenceException in r015): expected, so a warning.
                LogMessage("warning", "[ChatPanel] Stop interrupted Meta's runner: " + e.GetType().Name + ": " + e.Message +
                                      FirstFrame(e.StackTrace));
            }
            catch (Exception e)
            {
                Fail("model.generate", e.Message + FirstFrame(e.StackTrace));
            }

            bool stopped = running.IsCancellationRequested;
            double total = stream.Watch.Elapsed.TotalMilliseconds;
            string reply = response != null && !string.IsNullOrEmpty(response.text) ? response.text : stream.Text;
            answer.text = reply;
            LogGenerate(request, promptId, reply, stream, total, stopped);
            if (Application.isEditor)
            {
                Debug.Log($"[ChatPanel] {(stopped ? "Stopped" : "Done")} after {Tokens(stream.Pieces)}: {reply}");
            }
            SetStatus(stream.Pieces == 0
                ? (stopped ? "Stopped before the first token." : "No answer came back. The event log has Meta's messages.")
                : $"{(stopped ? "Stopped" : "Done")}: {Tokens(stream.Pieces)}, the first after {stream.FirstMs / 1000:F1} s, " +
                  $"all after {total / 1000:F1} s.");

            running.Dispose();
            running = null;
            buttonLabel.text = "Send";
            foreach (Button b in presetButtons) b.interactable = true;
        }

        /// <summary>Receives the answer one token at a time, straight from the provider's loop, and logs each piece.</summary>
        private sealed class AnswerStream : IProgress<ChatDelta>
        {
            private readonly ChatPanel owner;
            private readonly int request;
            private readonly StringBuilder text = new StringBuilder();
            public readonly Stopwatch Watch = new Stopwatch();
            public int Pieces;
            public double FirstMs;

            public AnswerStream(ChatPanel owner, int request)
            {
                this.owner = owner;
                this.request = request;
            }

            public string Text { get { return text.ToString(); } }

            public void Report(ChatDelta delta)
            {
                double ms = Watch.Elapsed.TotalMilliseconds;
                if (Pieces == 0)
                {
                    FirstMs = ms;
                }
                string piece = delta != null && delta.textFragment != null ? delta.textFragment : "";
                text.Append(piece);
                owner.answer.text = text.ToString();

                var data = new StringBuilder("{\"request\":").Append(request.ToString(CultureInfo.InvariantCulture));
                data.Append(",\"index\":").Append(Pieces.ToString(CultureInfo.InvariantCulture));
                data.Append(",\"text\":");
                LogJson.AppendString(data, piece);
                data.Append(",\"ms\":").Append(Number(ms)).Append('}');
                EventLog.Write("model.token", data.ToString());
                if (Application.isEditor)
                {
                    Debug.Log($"[ChatPanel] token {Pieces} after {ms / 1000:F2} s: \"{piece}\"");
                }
                Pieces++;
            }
        }

        private void LogRequest(int request, string promptId, List<int> ids)
        {
            var data = new StringBuilder("{\"request\":").Append(request.ToString(CultureInfo.InvariantCulture));
            data.Append(",\"prompt_id\":");
            LogJson.AppendString(data, promptId);
            data.Append(",\"prompt_tokens\":").Append(ids.Count.ToString(CultureInfo.InvariantCulture));
            data.Append(",\"prompt_token_ids\":[");
            for (int i = 0; i < ids.Count; i++)
            {
                data.Append(i == 0 ? "" : ",").Append(ids[i].ToString(CultureInfo.InvariantCulture));
            }
            data.Append("]}");
            EventLog.Write("model.request", data.ToString());
        }

        private static void LogGenerate(int request, string promptId, string reply, AnswerStream stream, double total, bool stopped)
        {
            var data = new StringBuilder("{\"request\":").Append(request.ToString(CultureInfo.InvariantCulture));
            data.Append(",\"prompt_id\":");
            LogJson.AppendString(data, promptId);
            data.Append(",\"answer\":");
            LogJson.AppendString(data, reply);
            data.Append(",\"answer_tokens\":").Append(stream.Pieces.ToString(CultureInfo.InvariantCulture));
            data.Append(",\"first_token_ms\":").Append(stream.Pieces > 0 ? Number(stream.FirstMs) : "null");
            data.Append(",\"total_ms\":").Append(Number(total));
            data.Append(",\"stopped\":").Append(stopped ? "true" : "false").Append('}');
            EventLog.Write("model.generate", data.ToString());
        }

        private void OnLog(string message, string stackTrace, LogType type)
        {
            if (type == LogType.Log || message == null)
            {
                return;
            }
            foreach (string prefix in MetaPrefixes)
            {
                if (message.StartsWith(prefix, StringComparison.Ordinal))
                {
                    LogMessage(type == LogType.Warning ? "warning" : "error", message);
                    return;
                }
            }
        }

        private static void LogMessage(string level, string text)
        {
            var data = new StringBuilder("{\"level\":");
            LogJson.AppendString(data, level);
            data.Append(",\"text\":");
            LogJson.AppendString(data, text);
            data.Append('}');
            EventLog.Write("model.message", data.ToString());
        }

        /// <summary>" (at <first line of the stack trace>)", so a logged error says where it came from; empty if unknown.</summary>
        private static string FirstFrame(string stackTrace)
        {
            if (string.IsNullOrEmpty(stackTrace)) return "";
            string first = stackTrace.Split('\n')[0].Trim();
            return first.Length > 0 ? " (" + first + ")" : "";
        }

        private void Fail(string where, string message)
        {
            Debug.LogError($"[ChatPanel] {where}: {message}");
            EventLog.Error(where, message);
            SetStatus("Error: " + message);
        }

        private void SetStatus(string value)
        {
            if (status != null)
            {
                status.text = value;
            }
        }

        private void SetText(string value)
        {
            text = value ?? "";
            boxText.text = text;
        }

        private bool Place()
        {
            Camera head = Camera.main != null ? Camera.main : FindAnyObjectByType<Camera>();
            if (head == null || panel == null)
            {
                return false;
            }
            Quaternion yaw = Quaternion.Euler(0f, head.transform.eulerAngles.y, 0f);
            panel.position = head.transform.position + yaw * offsetM;
            panel.rotation = yaw;
            return true;
        }

        private static T Field<T>(object target, string name) where T : class
        {
            FieldInfo field = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            return field != null ? field.GetValue(target) as T : null;
        }

        private static string Tokens(int count)
        {
            return count == 1 ? "1 token" : count.ToString(CultureInfo.InvariantCulture) + " tokens";
        }

        private static string Number(double value)
        {
            return value.ToString("F1", CultureInfo.InvariantCulture);
        }

        // ---- The panel, built in code like DebugOverlay: 640 x 600 units at 1 mm each. ----

        private void BuildPanel()
        {
            var canvasObject = new GameObject("ChatPanelCanvas", typeof(Canvas), typeof(CanvasScaler));
            canvasObject.transform.SetParent(transform, false);
            canvasObject.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            canvasObject.GetComponent<CanvasScaler>().dynamicPixelsPerUnit = 4f;   // sharper text in world space
            panel = (RectTransform)canvasObject.transform;
            panel.sizeDelta = new Vector2(640f, 600f);
            panel.localScale = Vector3.one * 0.001f;
            panel.position = new Vector3(0f, -100f, 0f);   // out of sight until Place() runs

            Image background = Box(panel, "Background", 0f, 0f, 640f, 600f).gameObject.AddComponent<Image>();
            background.color = new Color(0.08f, 0.09f, 0.11f, 0.92f);
            Label(panel, "Title", 20f, 16f, 600f, 36f, 26, FontStyle.Bold).text = "Second Eyes · model test (A1.7c)";
            status = Label(panel, "Status", 20f, 56f, 600f, 30f, 18, FontStyle.Normal);
            status.color = new Color(0.75f, 0.8f, 0.85f);

            // The text box only shows the prompt; the preset buttons below choose it (D49).
            RectTransform box = Box(panel, "TextBox", 20f, 96f, 600f, 196f);
            Image boxImage = box.gameObject.AddComponent<Image>();
            boxImage.color = new Color(0.16f, 0.18f, 0.21f);
            boxText = Label(box, "Text", 12f, 8f, 576f, 180f, 16, FontStyle.Normal);
            boxText.supportRichText = false;
            for (int k = 0; k < choices.Count && k < 4; k++)   // up to four prompts fit in a row
            {
                int which = k;
                RectTransform presetBox = Box(panel, "Preset " + choices[k].id, 20f + k * 153f, 300f, 140f, 50f);
                Image presetImage = presetBox.gameObject.AddComponent<Image>();
                presetImage.color = PresetColor;
                Text presetText = Label(presetBox, "Label", 6f, 3f, 128f, 44f, 14, FontStyle.Normal);
                presetText.text = Command(choices[k]);
                presetText.alignment = TextAnchor.MiddleCenter;
                Button preset = presetBox.gameObject.AddComponent<Button>();
                preset.targetGraphic = presetImage;
                preset.onClick.AddListener(() => Choose(which));
                presetButtons.Add(preset);
            }

            RectTransform buttonBox = Box(panel, "Send", 460f, 362f, 160f, 52f);
            Image buttonImage = buttonBox.gameObject.AddComponent<Image>();
            buttonImage.color = new Color(0.2f, 0.5f, 0.95f);
            buttonLabel = Label(buttonBox, "Label", 0f, 0f, 160f, 56f, 22, FontStyle.Bold);
            buttonLabel.text = "Send";
            buttonLabel.alignment = TextAnchor.MiddleCenter;
            sendButton = buttonBox.gameObject.AddComponent<Button>();
            sendButton.targetGraphic = buttonImage;
            sendButton.interactable = false;
            sendButton.onClick.AddListener(OnButton);

            Label(panel, "AnswerTitle", 20f, 426f, 600f, 28f, 18, FontStyle.Bold).text = "Answer";
            answer = Label(panel, "Answer", 20f, 456f, 600f, 130f, 18, FontStyle.Normal);
        }

        /// <summary>A child rectangle placed from the parent's top-left corner, in panel units.</summary>
        private static RectTransform Box(Transform parent, string name, float x, float y, float width, float height)
        {
            var box = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
            box.SetParent(parent, false);
            box.anchorMin = box.anchorMax = box.pivot = new Vector2(0f, 1f);
            box.anchoredPosition = new Vector2(x, -y);
            box.sizeDelta = new Vector2(width, height);
            return box;
        }

        private static Text Label(Transform parent, string name, float x, float y, float width, float height, int size, FontStyle style)
        {
            Text label = Box(parent, name, x, y, width, height).gameObject.AddComponent<Text>();
            label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            label.fontSize = size;
            label.fontStyle = style;
            label.color = Color.white;
            label.horizontalOverflow = HorizontalWrapMode.Wrap;
            label.verticalOverflow = VerticalWrapMode.Truncate;
            return label;
        }
    }
}
