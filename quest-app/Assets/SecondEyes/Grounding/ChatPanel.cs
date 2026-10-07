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
    /// The model test panel (A1.7c-2: D40, D43, D49; A1.7d: D50). It shows, about a meter ahead, the prompt's user text,
    /// a row of preset buttons (the fixed prompt and the presets, each a prompt file like grounding/prompts/a17-fixed.json)
    /// and a row with Load model, Repeat, Steps and Send, then the answer. There is no typing (D49).
    /// Load model loads the on-device model; unless Load at Start is ticked, the app starts without it, so its cost can be
    /// measured with the model off (A1.7d). Backend (CPU or GPU) and Weights (16-bit, or 32-bit from a file pushed to the
    /// headset with adb) apply at Load and are fixed after it (A1.8a, D53); model.load logs the file and backend used.
    /// Runtime (A1.8c, D58) chooses Meta's runner or llama.cpp before Load. llama.cpp runs on a worker thread
    /// (LlamaRuntime), keeps the scene in its cache, answers greedily as the PC references do, then scores the objects
    /// the prompt lists (model.scores); its Steps button sets threads instead. Its model is a GGUF file pushed to the app's
    /// data folder (python grounding/llama_headset.py push-model), and its library exists on the headset only. Send runs the shown text through Meta's provider, and while it runs the button
    /// is Stop. Repeat sends the chosen prompt again a few seconds after each answer, until Repeat or Stop is pressed.
    /// Steps cycles Meta's steps per frame (its prompt pass uses half), between answers only. Each step is logged as it
    /// happens: model.load, model.setting (Repeat and Steps), model.request (the prompt's ID and token IDs, from the
    /// provider's own template and tokenizer), model.token (each piece of the answer and its time) and model.generate
    /// (the end). grounding/check_headset.py compares a run with the PC references. In the Unity editor's Play mode,
    /// right-click the component for the same actions without a headset; the answer then also goes to the Console.
    /// The right thumbstick's click puts the panel in front again.
    /// </summary>
    public class ChatPanel : MonoBehaviour
    {
        [Tooltip("Meta's provider asset, filled by Second Eyes > Fill chat provider.")]
        [SerializeField] private UnityInferenceEngineProvider provider;
        [Tooltip("The fixed prompt, a copy of grounding/prompts/a17-fixed.json; its user text fills the text box.")]
        [SerializeField] private TextAsset prompt;
        [Tooltip("More prompts, each a copy of a file in grounding/prompts/; each gets a button next to the fixed prompt's.")]
        [SerializeField] private TextAsset[] presets = new TextAsset[0];
        [Tooltip("Load the model as the app starts. Off: the Load model button loads it, so the app can run without it (D50).")]
        [SerializeField] private bool loadAtStart;
        [Tooltip("With Repeat on, the pause after each answer before the next send, in seconds.")]
        [SerializeField] private float repeatPauseS = 5f;
        [Tooltip("The steps per frame the Steps button cycles through (Meta's setting; its prompt pass uses half).")]
        [SerializeField] private int[] stepsChoices = { 150, 50, 15 };
        [Tooltip("llama.cpp's model, in the app's data folder (python grounding/llama_headset.py push-model).")]
        [SerializeField] private string ggufFile = "qwen2.5-0.5b-instruct-q8_0.gguf";
        [Tooltip("The threads llama.cpp's Threads button cycles through.")]
        [SerializeField] private int[] threadChoices = { 2, 4 };
        [Tooltip("Where the panel appears relative to the head, in meters: right, up, forward.")]
        [SerializeField] private Vector3 offsetM = new Vector3(0f, -0.1f, 1.0f);

        private const string TypedId = "typed";
        private static readonly string[] MetaPrefixes =
            { "[TextOnlyLLMRunner]", "[UnityInferenceEngineProvider]", "[GPT2Tokenizer]", "[OnDeviceLLMConfig]" };
        private static readonly Color PresetColor = new Color(0.22f, 0.25f, 0.3f), ChosenColor = new Color(0.3f, 0.45f, 0.6f);
        private static readonly Color RepeatOnColor = new Color(0.75f, 0.45f, 0.15f);

#pragma warning disable 0649   // filled by JsonUtility
        [Serializable] private class PromptFile { public string id, user; }
#pragma warning restore 0649

        private OnDeviceLlmConfig config;
        private readonly List<PromptFile> choices = new List<PromptFile>();   // the fixed prompt first, then the presets
        private readonly List<Button> presetButtons = new List<Button>();
        private int chosen;
        private RectTransform panel;
        private Button sendButton, loadButton, repeatButton, stepsButton, backendButton, weightsButton;
        private Text boxText, buttonLabel, loadLabel, repeatLabel, stepsLabel, backendLabel, weightsLabel, status, answer;
        [SerializeField, Tooltip("Start on llama.cpp, the headset's runtime (D60); Meta's runner stays one press of Runtime away, for A1.7's comparison.")]
        private bool startWithLlama = true;

        private bool useGpu, use32Bit, useLlama;
        private LlamaRuntime llama;
        private int llamaThreads = 2;
        private Button runtimeButton;
        private Text runtimeLabel;
        private const string AnswerPrefix = "{\"action\": \"INSPECT\", \"target\": \"";   // as grounding/scene.py
        private const string CandidateSuffix = "\"}";
        private static readonly System.Text.RegularExpressions.Regex ObjectLine =
            new System.Text.RegularExpressions.Regex(@"^([A-Za-z][A-Za-z0-9]*_[0-9]+) ", System.Text.RegularExpressions.RegexOptions.Multiline);
        private UnityInferenceEngineProvider working;   // the copy the panel uses; the asset stays as it is (D53)
        private string file16;   // the provider's own model file: the 16-bit one
        private string text = "";
        private CancellationTokenSource running;
        private int requests;
        private bool loaded, loading, placed, repeat;
        private float nextRepeatAt = -1f;   // Time.unscaledTime of Repeat's next send; negative: none planned

        private void OnEnable() { Application.logMessageReceived += OnLog; }

        private void OnDisable() { Application.logMessageReceived -= OnLog; }

        private void OnDestroy()
        {
            if (working != null)
            {
                Destroy(working);   // Meta's OnDisable on the copy then releases the model
            }
            if (llama != null)
            {
                llama.Dispose();    // frees llama.cpp's model on its worker thread
                llama = null;
            }
        }

        private void Start()
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
            // Work on a copy: Steps, Backend and Weights change the provider's settings, and in the editor a
            // ScriptableObject changed in Play mode stays changed, so a test would silently change the next build (D53).
            working = Instantiate(provider);
            config = Field<OnDeviceLlmConfig>(working, "llmConfig");   // Meta keeps this field internal
            if (config == null)
            {
                Fail("setup", "Can't read the provider's chat settings (llmConfig). Meta's SDK may have changed.");
                return;
            }
            file16 = Field<string>(working, "streamingAssetFileName");
            if (threadChoices != null && threadChoices.Length > 0) llamaThreads = Math.Max(1, threadChoices[0]);
            useLlama = startWithLlama;
            runtimeButton.interactable = true;
            backendButton.interactable = !useLlama;   // Backend and Weights are Meta's runner's settings
            weightsButton.interactable = !useLlama && File32() != null;
            UpdateLabels();
            if (loadAtStart)
            {
                LoadModel();
            }
            else
            {
                loadButton.interactable = true;
                SetStatus("The model is off. Press Load model to load it.");
            }
        }

        [ContextMenu("Load model")]
        private void LoadModelFromInspector()
        {
            if (Application.isPlaying) LoadModel();
        }

        private async void LoadModel()
        {
            if (loaded || loading || config == null)
            {
                return;
            }
            if (useLlama)
            {
                LoadLlama();
                return;
            }
            string chosen = use32Bit ? File32() : file16;
            if (use32Bit && !OnHeadset(chosen))
            {
                SetStatus("The 32-bit model isn't on the headset. Push it with adb first (docs/setup/quest-model.md).");
                return;
            }
            config.backendType = useGpu ? Unity.InferenceEngine.BackendType.GPUCompute : Unity.InferenceEngine.BackendType.CPU;
            SetField(working, "streamingAssetFileName", chosen);
            loading = true;
            loadButton.interactable = false;
            backendButton.interactable = weightsButton.interactable = false;   // fixed from here on
            loadLabel.text = "Loading...";
            string file = Field<string>(working, "streamingAssetFileName");
            string source = Application.streamingAssetsPath;
            bool copies = !string.IsNullOrEmpty(file) && (source.Contains("://") || source.Contains("jar:"))
                          && !File.Exists(Path.Combine(Application.persistentDataPath, file));
            SetStatus(copies ? "Loading model (first start: copying it out of the app first)..." : "Loading model...");
            // Written before the load, so a run whose app is killed while loading still shows what it tried (A1.8a).
            var mark = new StringBuilder("{\"text\":");
            LogJson.AppendString(mark, "loading " + file + " on the " + config.backendType + " backend");
            EventLog.Write("mark", mark.Append('}').ToString());
            var watch = Stopwatch.StartNew();
            try
            {
                await working.WarmUp();
            }
            catch (Exception e)
            {
                loading = false;
                // Meta's provider keeps a runner whose load failed, and a second WarmUp then does nothing (r024), so a
                // new attempt needs a fresh copy of the asset.
                Destroy(working);
                working = Instantiate(provider);
                config = Field<OnDeviceLlmConfig>(working, "llmConfig");
                UpdateLabels();
                loadButton.interactable = true;
                backendButton.interactable = true;
                weightsButton.interactable = File32() != null;
                loadLabel.text = "Load model";
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
            loading = false;
            loadLabel.text = "Model loaded";
            sendButton.interactable = true;
            repeatButton.interactable = true;
            SetStatus($"Ready. The model loaded in {ms / 1000:F1} s.");
            if (Application.isEditor)
            {
                Debug.Log($"[ChatPanel] Ready. The model loaded in {ms / 1000:F1} s.");
            }
        }

        private void LateUpdate()
        {
            if (repeat && running == null && nextRepeatAt >= 0f && Time.unscaledTime >= nextRepeatAt)
            {
                nextRepeatAt = -1f;
                Send();
            }
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

        /// <summary>Whether the model is loaded, and whether a send is running (A1.10d, D88).</summary>
        public bool IsLoaded => loaded;
        public bool IsBusy => running != null;

        /// <summary>The number of prompt choices (the fixed prompt, then the presets) and choice k's prompt ID.</summary>
        public int ChoiceCount => choices.Count;
        public string ChoiceId(int k) => k >= 0 && k < choices.Count ? choices[k].id : null;

        /// <summary>
        /// Sends choice k exactly as pressing its button and Send would (A1.10d's fixed command schedule, D88, O21).
        /// Returns null when sent, otherwise why not: "not_loaded", "busy" or "no_such_choice". Never queues.
        /// </summary>
        public string TryDispatch(int k)
        {
            if (!loaded) return "not_loaded";
            if (running != null) return "busy";
            if (k < 0 || k >= choices.Count) return "no_such_choice";
            Choose(k);
            Send();
            return null;
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

        [ContextMenu("Repeat on or off")]
        private void RepeatFromInspector()
        {
            if (Application.isPlaying) ToggleRepeat();
        }

        [ContextMenu("Next steps per frame")]
        private void StepsFromInspector()
        {
            if (Application.isPlaying) NextSteps();
        }

        /// <summary>Repeat on: sends now if idle, then again a pause after each answer. Off: no more sends are planned.</summary>
        private void ToggleRepeat()
        {
            if (!loaded)
            {
                return;
            }
            SetRepeat(!repeat);
            if (repeat && running == null)
            {
                Send();
            }
        }

        private void SetRepeat(bool on)
        {
            if (repeat == on) return;
            repeat = on;
            nextRepeatAt = -1f;
            if (!on && status.text.EndsWith(RepeatNote(), StringComparison.Ordinal))
            {
                SetStatus(status.text.Substring(0, status.text.Length - RepeatNote().Length));
            }
            UpdateLabels();
            LogSetting();
        }

        /// <summary>The next value of stepsChoices, between answers only, so one answer never mixes two settings.</summary>
        private void NextSteps()
        {
            if (useLlama)
            {
                NextThreads();
                return;
            }
            if (config == null || running != null || stepsChoices == null || stepsChoices.Length == 0)
            {
                return;
            }
            int k = Array.IndexOf(stepsChoices, config.stepsPerFrame);
            config.stepsPerFrame = Math.Max(1, stepsChoices[(k + 1) % stepsChoices.Length]);   // not in the list: its first
            UpdateLabels();
            LogSetting();
        }

        private string RepeatNote() { return $" Again in {repeatPauseS:F0} s."; }

        private async void NextThreads()
        {
            if (running != null || loading || threadChoices == null || threadChoices.Length == 0) return;
            int k = Array.IndexOf(threadChoices, llamaThreads);
            llamaThreads = Math.Max(1, threadChoices[(k + 1) % threadChoices.Length]);
            UpdateLabels();
            LogSetting();
            if (llama != null && loaded)
            {
                stepsButton.interactable = false;
                try { await llama.SetThreadsAsync(llamaThreads); }
                finally { stepsButton.interactable = running == null; }
            }
        }

        private async void LoadLlama()
        {
            string path = Path.Combine(Application.persistentDataPath, ggufFile);
            if (!File.Exists(path))
            {
                SetStatus("llama.cpp's model isn't on the headset. Push it: python grounding/llama_headset.py push-model");
                return;
            }
            loading = true;
            loadButton.interactable = runtimeButton.interactable = false;
            loadLabel.text = "Loading...";
            SetStatus("Loading model with llama.cpp...");
            var mark = new StringBuilder("{\"text\":");
            LogJson.AppendString(mark, "loading " + ggufFile + " with llama.cpp, " + llamaThreads + " threads");
            EventLog.Write("mark", mark.Append('}').ToString());
            llama = new LlamaRuntime(System.Threading.SynchronizationContext.Current);
            double ms;
            try
            {
                ms = await llama.LoadAsync(path, 1024, llamaThreads, 0);
            }
            catch (Exception e)
            {
                llama.Dispose();
                llama = null;
                loading = false;
                loadButton.interactable = runtimeButton.interactable = true;
                loadLabel.text = "Load model";
                string why = e is DllNotFoundException || e.InnerException is DllNotFoundException
                    ? "llama.cpp runs on the headset only: libse_llama isn't built for this platform." : e.Message;
                Fail("model.load", why);
                return;
            }
            var data = new StringBuilder("{\"file\":");
            LogJson.AppendString(data, ggufFile);
            data.Append(",\"copied\":false,\"ms\":").Append(Number(ms));
            data.Append(",\"backend\":\"llama.cpp\",\"execution_mode\":\"worker thread\",\"runtime\":\"llama.cpp\"");
            data.Append(",\"threads\":").Append(llamaThreads.ToString(CultureInfo.InvariantCulture));
            data.Append(",\"llama_cpp\":");
            LogJson.AppendString(data, llama.Version);
            data.Append(",\"memory_kb\":").Append(LlamaRuntime.MemoryKb(false).ToString(CultureInfo.InvariantCulture)).Append('}');
            EventLog.Write("model.load", data.ToString());
            loaded = true;
            loading = false;
            loadLabel.text = "Model loaded";
            sendButton.interactable = true;
            repeatButton.interactable = true;
            SetStatus($"Ready. llama.cpp loaded the model in {ms / 1000:F1} s ({llamaThreads} threads).");
        }

        private void ToggleRuntime()
        {
            if (loaded || loading) return;
            useLlama = !useLlama;
            backendButton.interactable = !useLlama;
            weightsButton.interactable = !useLlama && File32() != null;
            UpdateLabels();
        }

        private void ToggleBackend()
        {
            if (loaded || loading) return;
            useGpu = !useGpu;
            UpdateLabels();
        }

        private void ToggleWeights()
        {
            if (loaded || loading || File32() == null) return;
            use32Bit = !use32Bit;
            UpdateLabels();
        }

        /// <summary>The 32-bit file next to the provider's 16-bit one (the same export, D48 naming), or null.</summary>
        private string File32()
        {
            const string tag = "-f16.sentis";
            return file16 != null && file16.EndsWith(tag, StringComparison.Ordinal)
                ? file16.Substring(0, file16.Length - tag.Length) + "-f32.sentis" : null;
        }

        /// <summary>Whether Meta's runner can load this file without copying it out of the app: in the app's data folder,
        /// or, in the editor, in StreamingAssets itself.</summary>
        private static bool OnHeadset(string file)
        {
            if (string.IsNullOrEmpty(file)) return false;
            if (File.Exists(Path.Combine(Application.persistentDataPath, file))) return true;
            string source = Application.streamingAssetsPath;
            return !source.Contains("://") && !source.Contains("jar:") && File.Exists(Path.Combine(source, file));
        }

        private void UpdateLabels()
        {
            repeatLabel.text = repeat ? "Repeat: on" : "Repeat: off";
            repeatButton.targetGraphic.color = repeat ? RepeatOnColor : PresetColor;
            stepsLabel.text = useLlama ? "Threads: " + llamaThreads.ToString(CultureInfo.InvariantCulture)
                            : config != null ? "Steps: " + config.stepsPerFrame.ToString(CultureInfo.InvariantCulture) : "Steps: ?";
            runtimeLabel.text = useLlama ? "Runtime: llama.cpp" : "Runtime: Unity";
            backendLabel.text = useLlama ? "Backend: CPU" : useGpu ? "Backend: GPU" : "Backend: CPU";
            weightsLabel.text = useLlama ? "Weights: 8-bit" : use32Bit ? "Weights: 32-bit" : "Weights: 16-bit";
        }

        private void LogSetting()
        {
            var data = new StringBuilder("{\"repeat\":").Append(repeat ? "true" : "false");
            if (useLlama) data.Append(",\"threads\":").Append(llamaThreads.ToString(CultureInfo.InvariantCulture));
            else data.Append(",\"steps_per_frame\":").Append(config != null ? config.stepsPerFrame.ToString(CultureInfo.InvariantCulture) : "1");
            data.Append(",\"pause_s\":").Append(Number(repeatPauseS)).Append('}');
            EventLog.Write("model.setting", data.ToString());
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
                SetRepeat(false);   // Stop always means stop
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
            stepsButton.interactable = false;
            answer.text = "";
            SetStatus("Generating...");
            var stream = new AnswerStream(this, request);
            ChatResponse response = null;
            LlamaRuntime.Prepared prepared = null;   // llama.cpp: scored after the answer is logged
            try
            {
                // The same two steps Meta's runner takes, so these are exactly the IDs the model sees.
                string formatted = config.ApplyChatTemplate(sent, config.defaultSystemMessage);
                if (useLlama)
                {
                    prepared = await SendLlama(request, promptId, formatted, stream);
                }
                else
                {
                    List<int> ids = await config.Tokenizer.EncodeAsync(formatted);
                    LogRequest(request, promptId, ids);
                    stream.Watch.Restart();
                    response = await working.ChatAsync(new ChatRequest(sent), stream, running.Token);
                }
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
                SetRepeat(false);   // don't repeat a failure
                Fail("model.generate", e.Message + FirstFrame(e.StackTrace));
            }

            bool stopped = running.IsCancellationRequested;
            double total = stream.Watch.Elapsed.TotalMilliseconds;
            string reply = response != null && !string.IsNullOrEmpty(response.text) ? response.text : stream.Text;
            answer.text = reply;
            LogGenerate(request, promptId, reply, stream, total, stopped);
            string scored = "";
            if (prepared != null && !stopped)
            {
                try
                {
                    scored = await ScoreLlama(request, promptId, sent, prepared);
                }
                catch (Exception e)
                {
                    SetRepeat(false);
                    Fail("model.scores", e.Message + FirstFrame(e.StackTrace));
                }
            }
            if (Application.isEditor)
            {
                Debug.Log($"[ChatPanel] {(stopped ? "Stopped" : "Done")} after {Tokens(stream.Pieces)}: {reply}");
            }
            SetStatus(stream.Pieces == 0
                ? (stopped ? "Stopped before the first token." : "No answer came back. The event log has Meta's messages.")
                : $"{(stopped ? "Stopped" : "Done")}: {Tokens(stream.Pieces)}, the first after {stream.FirstMs / 1000:F1} s, " +
                  $"all after {total / 1000:F1} s." + scored);

            running.Dispose();
            running = null;
            buttonLabel.text = "Send";
            foreach (Button b in presetButtons) b.interactable = true;
            stepsButton.interactable = true;
            if (repeat && !stopped)
            {
                nextRepeatAt = Time.unscaledTime + Mathf.Max(0f, repeatPauseS);
                SetStatus(status.text + RepeatNote());
            }
        }

        /// <summary>A command through llama.cpp: the scene from the cache (evaluated when it changes), then the rest and
        /// the greedy answer token by token. Returns what ScoreLlama needs.</summary>
        private async System.Threading.Tasks.Task<LlamaRuntime.Prepared> SendLlama(int request, string promptId,
                                                                                   string formatted, AnswerStream stream)
        {
            int sceneChars = formatted.IndexOf("\nUser at ", StringComparison.Ordinal) + 1;
            stream.Watch.Restart();
            LlamaRuntime.Prepared p = await llama.PrepareAsync(formatted, sceneChars);
            LogRequest(request, promptId, new List<int>(p.Ids), p.SceneReused ? p.SceneTokens : 0, p.SceneMs);
            await llama.AnswerAsync(p, Math.Max(1, config.maxNewTokens), stream.Watch, stream.Add, running.Token);
            stream.Watch.Stop();   // the answer's time ends here; scoring is logged on its own
            return p;
        }

        /// <summary>Scores the objects the prompt lists (model.scores); returns a note for the status line.</summary>
        private async System.Threading.Tasks.Task<string> ScoreLlama(int request, string promptId, string sent,
                                                                     LlamaRuntime.Prepared p)
        {
            var names = new List<string>();
            string head = sent.Split(new[] { "Objects:" }, 2, StringSplitOptions.None).Last();
            int user = head.IndexOf("User at", StringComparison.Ordinal);
            foreach (System.Text.RegularExpressions.Match m in ObjectLine.Matches(user >= 0 ? head.Substring(0, user) : head))
            {
                names.Add(m.Groups[1].Value);
            }
            if (names.Count == 0) return "";
            LlamaRuntime.Scores sc = await llama.ScoreAsync(p, AnswerPrefix, names, CandidateSuffix);
            var data = new StringBuilder("{\"request\":").Append(request.ToString(CultureInfo.InvariantCulture));
            data.Append(",\"prompt_id\":");
            LogJson.AppendString(data, promptId);
            data.Append(",\"candidates\":{");
            for (int i = 0; i < names.Count; i++)
            {
                if (i > 0) data.Append(',');
                LogJson.AppendString(data, names[i]);
                data.Append(':').Append(sc.LogProbs[names[i]].ToString("R", CultureInfo.InvariantCulture));
            }
            data.Append("},\"best\":");
            LogJson.AppendString(data, sc.Best);
            data.Append(",\"ms\":").Append(Number(sc.Ms)).Append('}');
            EventLog.Write("model.scores", data.ToString());
            return $" Scored {names.Count} objects in {sc.Ms / 1000:F2} s: best {sc.Best}.";
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
                Add(delta != null && delta.textFragment != null ? delta.textFragment : "", Watch.Elapsed.TotalMilliseconds);
            }

            /// <summary>One piece of the answer, at `ms` on this stream's clock (llama.cpp's worker measures it).</summary>
            public void Add(string piece, double ms)
            {
                if (Pieces == 0)
                {
                    FirstMs = ms;
                }
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

        private void LogRequest(int request, string promptId, List<int> ids, int cachedTokens = -1, double sceneMs = 0)
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
            data.Append(']');
            if (cachedTokens >= 0)   // llama.cpp: how many prompt tokens came from the cache, and the scene's time if not
            {
                data.Append(",\"runtime\":\"llama.cpp\",\"cached_tokens\":").Append(cachedTokens.ToString(CultureInfo.InvariantCulture));
                data.Append(",\"scene_ms\":").Append(Number(sceneMs));
            }
            data.Append('}');
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

        private static void SetField(object target, string name, object value)
        {
            FieldInfo f = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (f == null) throw new InvalidOperationException("Meta's provider has no field " + name + " (SDK changed?)");
            f.SetValue(target, value);
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
            panel.sizeDelta = new Vector2(640f, 666f);
            panel.localScale = Vector3.one * 0.001f;
            panel.position = new Vector3(0f, -100f, 0f);   // out of sight until Place() runs

            Image background = Box(panel, "Background", 0f, 0f, 640f, 666f).gameObject.AddComponent<Image>();
            background.color = new Color(0.08f, 0.09f, 0.11f, 0.92f);
            Label(panel, "Title", 20f, 16f, 600f, 36f, 26, FontStyle.Bold).text = "Second Eyes · model test (A1.7)";
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

            loadButton = RowButton("Load", 20f, 140f, out loadLabel, LoadModel);
            loadLabel.text = "Load model";
            loadButton.interactable = false;   // until the settings are read
            repeatButton = RowButton("Repeat", 173f, 140f, out repeatLabel, ToggleRepeat);
            repeatButton.interactable = false;   // until the model is loaded
            stepsButton = RowButton("Steps", 326f, 120f, out stepsLabel, NextSteps);

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

            // Backend and Weights apply at Load (A1.8a, D53)
            runtimeButton = RowButton("Runtime", 20f, 190f, out runtimeLabel, ToggleRuntime, 424f);
            backendButton = RowButton("Backend", 223f, 190f, out backendLabel, ToggleBackend, 424f);
            weightsButton = RowButton("Weights", 426f, 194f, out weightsLabel, ToggleWeights, 424f);
            runtimeButton.interactable = backendButton.interactable = weightsButton.interactable = false;   // until the settings are read

            Label(panel, "AnswerTitle", 20f, 488f, 600f, 28f, 18, FontStyle.Bold).text = "Answer";
            answer = Label(panel, "Answer", 20f, 518f, 600f, 130f, 18, FontStyle.Normal);
            UpdateLabels();
        }

        private Button RowButton(string name, float x, float width, out Text label, UnityEngine.Events.UnityAction onClick,
                                 float y = 362f)
        {
            RectTransform rect = Box(panel, name, x, y, width, 52f);
            Image image = rect.gameObject.AddComponent<Image>();
            image.color = PresetColor;
            label = Label(rect, "Label", 4f, 0f, width - 8f, 52f, 17, FontStyle.Normal);
            label.alignment = TextAnchor.MiddleCenter;
            Button button = rect.gameObject.AddComponent<Button>();
            button.targetGraphic = image;
            button.onClick.AddListener(onClick);
            return button;
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
