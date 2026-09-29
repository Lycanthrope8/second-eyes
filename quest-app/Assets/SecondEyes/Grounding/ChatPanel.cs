using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using Meta.XR.BuildingBlocks.AIBlocks;
using SecondEyes.Logging;
using UnityEngine;
using UnityEngine.UI;
using Stopwatch = System.Diagnostics.Stopwatch;
using LogJson = SecondEyes.Logging.Json;   // alias: Meta's namespace could have its own Json

namespace SecondEyes.Grounding
{
    /// <summary>
    /// A1.7c-2's test panel (D40). At startup it loads the on-device model and shows, about a meter ahead, a text box
    /// filled with the fixed prompt's user text, a Send button and the answer. Send runs the text through Meta's provider,
    /// shows the answer as it streams in, and logs model.generate with the prompt's token IDs, taken from the provider's
    /// own template and tokenizer, so grounding/check_headset.py can compare them and the answer with the PC reference.
    /// Sends of the unedited fixed prompt carry its ID; edited ones are logged as "typed". Point with UiPointer; selecting
    /// the text box opens the Quest system keyboard (D42). The right thumbstick's click puts the panel in front again.
    /// </summary>
    public class ChatPanel : MonoBehaviour
    {
        [Tooltip("Meta's provider asset, filled by Second Eyes > Fill chat provider.")]
        [SerializeField] private UnityInferenceEngineProvider provider;
        [Tooltip("The fixed prompt, a copy of grounding/prompts/a17-fixed.json; its user text fills the text box.")]
        [SerializeField] private TextAsset prompt;
        [Tooltip("Where the panel appears relative to the head, in meters: right, up, forward.")]
        [SerializeField] private Vector3 offsetM = new Vector3(0f, -0.1f, 1.0f);

        private const string TypedId = "typed";
        private static readonly string[] MetaPrefixes =
            { "[TextOnlyLLMRunner]", "[UnityInferenceEngineProvider]", "[GPT2Tokenizer]", "[OnDeviceLLMConfig]" };

#pragma warning disable 0649   // filled by JsonUtility
        [Serializable] private class PromptFile { public string id, user; }
#pragma warning restore 0649

        private OnDeviceLlmConfig config;
        private PromptFile fixedPrompt;
        private RectTransform panel;
        private InputField input;
        private Button send;
        private Text status, answer;
        private bool placed, busy;

        private void OnEnable() { Application.logMessageReceived += OnLog; }

        private void OnDisable() { Application.logMessageReceived -= OnLog; }

        private async void Start()
        {
            BuildPanel();
            fixedPrompt = prompt != null ? JsonUtility.FromJson<PromptFile>(prompt.text) : null;
            input.text = fixedPrompt != null ? fixedPrompt.user : "";
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

            SetStatus($"Ready. The model loaded in {ms / 1000:F1} s.");
            send.interactable = true;
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

        private async void OnSend()
        {
            string text = input.text;
            if (busy || config == null || string.IsNullOrWhiteSpace(text))
            {
                return;
            }
            busy = true;
            send.interactable = false;
            answer.text = "";
            SetStatus("Generating...");
            string promptId = fixedPrompt != null && text == fixedPrompt.user ? fixedPrompt.id : TypedId;
            try
            {
                // The same two steps Meta's runner takes, so these are exactly the IDs the model sees.
                string formatted = config.ApplyChatTemplate(text, config.defaultSystemMessage);
                List<int> ids = await config.Tokenizer.EncodeAsync(formatted);
                var stream = new AnswerStream(answer);
                ChatResponse response = await provider.ChatAsync(new ChatRequest(text), stream);
                double total = stream.Watch.Elapsed.TotalMilliseconds;
                string reply = response != null ? response.text : "";
                answer.text = reply;

                var data = new StringBuilder("{\"prompt_id\":");
                LogJson.AppendString(data, promptId);
                data.Append(",\"prompt_tokens\":").Append(ids.Count.ToString(CultureInfo.InvariantCulture));
                data.Append(",\"prompt_token_ids\":[");
                for (int i = 0; i < ids.Count; i++)
                {
                    data.Append(i == 0 ? "" : ",").Append(ids[i].ToString(CultureInfo.InvariantCulture));
                }
                data.Append("],\"answer\":");
                LogJson.AppendString(data, reply);
                data.Append(",\"answer_tokens\":").Append(stream.Pieces.ToString(CultureInfo.InvariantCulture));
                data.Append(",\"first_token_ms\":").Append(stream.Pieces > 0 ? Number(stream.FirstMs) : "null");
                data.Append(",\"total_ms\":").Append(Number(total)).Append('}');
                EventLog.Write("model.generate", data.ToString());

                SetStatus(stream.Pieces > 0
                    ? $"Done: {stream.Pieces} tokens, the first after {stream.FirstMs / 1000:F1} s, all after {total / 1000:F1} s."
                    : "No answer came back. The event log has Meta's messages.");
            }
            catch (Exception e)
            {
                Fail("model.generate", e.Message);
            }
            finally
            {
                busy = false;
                send.interactable = true;
            }
        }

        /// <summary>Receives the answer one token at a time, straight from the provider's loop, and times the first.</summary>
        private sealed class AnswerStream : IProgress<ChatDelta>
        {
            private readonly Text target;
            private readonly StringBuilder text = new StringBuilder();
            public readonly Stopwatch Watch = Stopwatch.StartNew();
            public int Pieces;
            public double FirstMs;

            public AnswerStream(Text target) { this.target = target; }

            public void Report(ChatDelta delta)
            {
                if (Pieces == 0)
                {
                    FirstMs = Watch.Elapsed.TotalMilliseconds;
                }
                Pieces++;
                text.Append(delta.textFragment);
                target.text = text.ToString();
            }
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
                    var data = new StringBuilder("{\"level\":");
                    LogJson.AppendString(data, type == LogType.Warning ? "warning" : "error");
                    data.Append(",\"text\":");
                    LogJson.AppendString(data, message);
                    data.Append('}');
                    EventLog.Write("model.message", data.ToString());
                    return;
                }
            }
        }

        private void Fail(string where, string message)
        {
            EventLog.Error(where, message);
            SetStatus("Error: " + message);
        }

        private void SetStatus(string text)
        {
            if (status != null)
            {
                status.text = text;
            }
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

            RectTransform fieldBox = Box(panel, "PromptField", 20f, 96f, 600f, 250f);
            Image fieldImage = fieldBox.gameObject.AddComponent<Image>();
            fieldImage.color = new Color(0.16f, 0.18f, 0.21f);
            Text fieldText = Label(fieldBox, "Text", 12f, 10f, 576f, 230f, 18, FontStyle.Normal);
            fieldText.supportRichText = false;
            Text hint = Label(fieldBox, "Placeholder", 12f, 10f, 576f, 230f, 18, FontStyle.Italic);
            hint.text = "Type a command";
            hint.color = new Color(1f, 1f, 1f, 0.35f);
            input = fieldBox.gameObject.AddComponent<InputField>();
            input.textComponent = fieldText;
            input.placeholder = hint;
            input.targetGraphic = fieldImage;
            input.lineType = InputField.LineType.MultiLineNewline;

            RectTransform buttonBox = Box(panel, "Send", 460f, 358f, 160f, 56f);
            Image buttonImage = buttonBox.gameObject.AddComponent<Image>();
            buttonImage.color = new Color(0.2f, 0.5f, 0.95f);
            Text buttonText = Label(buttonBox, "Label", 0f, 0f, 160f, 56f, 22, FontStyle.Bold);
            buttonText.text = "Send";
            buttonText.alignment = TextAnchor.MiddleCenter;
            send = buttonBox.gameObject.AddComponent<Button>();
            send.targetGraphic = buttonImage;
            send.interactable = false;
            send.onClick.AddListener(OnSend);

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
            Text text = Box(parent, name, x, y, width, height).gameObject.AddComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = size;
            text.fontStyle = style;
            text.color = Color.white;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Truncate;
            return text;
        }
    }
}
