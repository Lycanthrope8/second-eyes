// Second Eyes > Test model on the PC (A1.7c-2, D46): runs the model in the editor without Meta's runner, token by token,
// and compares every step with the PC reference.
#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Meta.XR.BuildingBlocks.AIBlocks;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace SecondEyes.EditorTools
{
    /// <summary>
    /// A window that checks, on the PC, each part the headset's answer depends on, against the PC reference
    /// (runs/REFERENCE/raw/reference_PROMPT_VARIANT.json from grounding/reference.py):
    ///   1. Tokenizer: do Meta's template and tokenizer, with the provider's settings, give the reference's prompt IDs?
    ///   2-4. The model, converted from ONNX in memory with 32-bit weights, then with Unity's 16-bit rounding, and the
    ///      .sentis file the app carries: fed the reference's own prompt IDs, does it give the reference's answer?
    ///   5. The 32-bit model with the prompt fed one token at a time, so no pass ever handles several tokens at once.
    ///   6. The 32-bit model on the GPU backend instead of the CPU backend.
    /// The model runs in a plain greedy loop of ours, the same steps as grounding/meta_runner.py, not Meta's runner, on the
    /// CPU backend. If everything here matches but the app still answers wrongly, the difference is in Meta's runner.
    /// Unity doesn't respond while a check runs. The report goes to this window, the Console and, optionally, a run.
    /// </summary>
    public class ModelTest : EditorWindow
    {
        const string PrefReference = "SecondEyes.ModelTest.Reference";
        const string PrefTokens = "SecondEyes.ModelTest.Tokens";
        const string PrefSaveRun = "SecondEyes.ModelTest.SaveRun";
        const string PromptFile = "grounding/prompts/a17-fixed.json";   // A1.7c's fixed prompt, as in Fill chat provider

        [Flags] enum Checks { Tokenizer = 1, Onnx32 = 2, Onnx16 = 4, Sentis = 8, All = 15, OneByOne = 16, Gpu = 32 }

        [Serializable] public class RefPrompt { public int[] token_ids; }
        [Serializable] public class RefAnswer { public int[] token_ids; public string text_meta_style; public int tokens; }
        [Serializable] public class RefFile { public RefPrompt prompt; public RefAnswer answer; }
        [Serializable] public class Prompt { public string id, system, user; }

        string referenceRun = "", saveRun = "", report = "";
        int tokens = 16, folderIndex;
        Vector2 scroll;

        [MenuItem("Second Eyes/Test model on the PC", false, 3)]
        static void Open()
        {
            GetWindow<ModelTest>("Model test");
        }

        void OnEnable()
        {
            referenceRun = EditorPrefs.GetString(PrefReference, "20260928_A1_r008");
            saveRun = EditorPrefs.GetString(PrefSaveRun, "");
            tokens = EditorPrefs.GetInt(PrefTokens, 16);
        }

        void OnGUI()
        {
            EditorGUILayout.LabelField("Runs the model on the PC without Meta's runner and compares every step with the PC " +
                                       "reference. Unity doesn't respond while a check runs.", EditorStyles.wordWrappedLabel);
            string[] folders = ModelFolders();
            if (folders.Length == 0)
            {
                EditorGUILayout.HelpBox("No model folder in " + ModelFiles.ModelsFolder + "/.", MessageType.Warning);
                return;
            }
            folderIndex = Mathf.Clamp(folderIndex, 0, folders.Length - 1);
            folderIndex = EditorGUILayout.Popup("Model", folderIndex, folders.Select(f => Path.GetFileName(f)).ToArray());
            referenceRun = EditorGUILayout.TextField("Reference run", referenceRun);
            tokens = EditorGUILayout.IntSlider("Answer tokens", tokens, 1, 100);
            saveRun = EditorGUILayout.TextField(new GUIContent("Save report into run",
                "Optional: a run ID. The report is then also saved as runs/<run>/raw/model-test_<UTC time>.txt."), saveRun);

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("1. Tokenizer")) Run(folders[folderIndex], Checks.Tokenizer);
            if (GUILayout.Button("2. 32-bit, in memory")) Run(folders[folderIndex], Checks.Onnx32);
            if (GUILayout.Button("3. 16-bit, in memory")) Run(folders[folderIndex], Checks.Onnx16);
            if (GUILayout.Button("4. The .sentis file")) Run(folders[folderIndex], Checks.Sentis);
            EditorGUILayout.EndHorizontal();
            if (GUILayout.Button("Run 1 to 4")) Run(folders[folderIndex], Checks.All);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("5. 32-bit, prompt one token at a time")) Run(folders[folderIndex], Checks.OneByOne);
            if (GUILayout.Button("6. 32-bit, on the GPU")) Run(folders[folderIndex], Checks.Gpu);
            EditorGUILayout.EndHorizontal();

            scroll = EditorGUILayout.BeginScrollView(scroll);
            EditorGUILayout.TextArea(report, GUILayout.ExpandHeight(true));
            EditorGUILayout.EndScrollView();
        }

        static string[] ModelFolders()
        {
            string root = Path.Combine(ModelFiles.ProjectRoot, ModelFiles.ModelsFolder);
            if (!Directory.Exists(root)) return new string[0];
            return Directory.GetDirectories(root).Select(d => ModelFiles.ModelsFolder + "/" + Path.GetFileName(d))
                            .Where(f => File.Exists(Path.Combine(ModelFiles.ProjectRoot, f, "model.json"))).OrderBy(f => f).ToArray();
        }

        void Run(string folder, Checks checks)
        {
            EditorPrefs.SetString(PrefReference, referenceRun);
            EditorPrefs.SetString(PrefSaveRun, saveRun);
            EditorPrefs.SetInt(PrefTokens, tokens);
            if ((checks & ~Checks.Tokenizer) != 0 && !EditorUtility.DisplayDialog("Model test",
                    "Each model check takes a few minutes and several GB of memory, and Unity doesn't respond meanwhile. " +
                    "Close big programs such as a web browser first.\n\nRun now?", "Run", "Cancel"))
            {
                return;
            }
            var output = new StringBuilder();
            try
            {
                var setup = Setup.Load(folder, referenceRun.Trim());
                output.AppendLine("Model test, " + DateTime.Now.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture) +
                                  ": " + Path.GetFileName(folder) + ", prompt " + setup.Prompt.id + ", reference run " +
                                  referenceRun.Trim() + ", Unity Inference Engine " + setup.Engine.Version + ".");
                if ((checks & Checks.Tokenizer) != 0) output.AppendLine(Guarded("1. Tokenizer", () => TokenizerCheck(setup)));
                if ((checks & Checks.Onnx32) != 0)
                    output.AppendLine(ModelCheck(setup, "2. 32-bit weights, converted from ONNX in memory, CPU backend", setup.Ref32, () =>
                        (Model)setup.Engine.Convert(setup.OnnxPath)));
                if ((checks & Checks.Onnx16) != 0)
                    output.AppendLine(ModelCheck(setup, "3. 16-bit weights (Unity's rounding), converted from ONNX in memory, CPU backend",
                        setup.Ref16, () =>
                        {
                            var model = (Model)setup.Engine.Convert(setup.OnnxPath);
                            ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
                            return model;
                        }));
                if ((checks & Checks.Sentis) != 0)
                    output.AppendLine(ModelCheck(setup, "4. The .sentis file " + setup.Model.sentis_file + ", CPU backend", setup.Ref16, () =>
                    {
                        if (!File.Exists(setup.SentisPath))
                            throw new ModelFiles.ToolError(ModelFiles.StreamingAssets + "/" + setup.Model.sentis_file +
                                                           " isn't there. Convert it first (Second Eyes > Convert model).");
                        return ModelLoader.Load(setup.SentisPath);
                    }));
                if ((checks & Checks.OneByOne) != 0)
                    output.AppendLine(ModelCheck(setup, "5. 32-bit weights, converted from ONNX in memory, CPU backend, the prompt fed one token at a time",
                        setup.Ref32, () => (Model)setup.Engine.Convert(setup.OnnxPath), BackendType.CPU, true));
                if ((checks & Checks.Gpu) != 0)
                    output.AppendLine(ModelCheck(setup, "6. 32-bit weights, converted from ONNX in memory, GPU backend (GPUCompute)",
                        setup.Ref32, () => (Model)setup.Engine.Convert(setup.OnnxPath), BackendType.GPUCompute, false));
            }
            catch (ModelFiles.ToolError e)
            {
                output.AppendLine("Stopped: " + e.Message);
            }
            catch (Exception e)
            {
                output.AppendLine("Stopped by " + e.GetType().Name + ": " + e.Message);
                Debug.LogException(e);
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
            report = output.ToString();
            Debug.Log("[Model test]\n" + report);
            if (!string.IsNullOrWhiteSpace(saveRun)) report += SaveReport(saveRun.Trim(), report);
            Repaint();
        }

        /// <summary>Everything the checks read, loaded and checked once.</summary>
        internal sealed class Setup
        {
            public FillChatProvider.Architecture Arch;
            public Prompt Prompt;
            public ModelFiles.UnityModel Model;
            public RefFile Ref32, Ref16;
            public OnDeviceLlmConfig Config;   // the provider's chat settings, for the tokenizer; null if no provider asset
            public ConvertModel.InferenceEngine Engine;
            public string OnnxPath, SentisPath;

            public static Setup Load(string folder, string referenceRun)
            {
                string name = Path.GetFileName(folder), repo = ModelFiles.RepoRoot;
                string descFile = "grounding/models/" + name + ".json";
                var s = new Setup();
                var desc = ModelFiles.ReadJson<FillChatProvider.Description>(Path.Combine(repo, descFile), descFile, "");
                s.Arch = desc.architecture;
                if (s.Arch == null || s.Arch.max_layers <= 0)
                    throw new ModelFiles.ToolError(descFile + " has no complete architecture section.");
                s.Prompt = ModelFiles.ReadJson<Prompt>(Path.Combine(repo, PromptFile), PromptFile, "");
                s.Model = ModelFiles.ReadModelJson(folder);
                string refDir = Path.Combine(repo, "runs", referenceRun, "raw");
                string refName = "runs/" + referenceRun + "/raw/reference_" + s.Prompt.id + "_";
                s.Ref32 = ModelFiles.ReadJson<RefFile>(Path.Combine(refDir, "reference_" + s.Prompt.id + "_fp32.json"),
                                                       refName + "fp32.json", "Make it with grounding/reference.py.");
                string ref16 = Path.Combine(refDir, "reference_" + s.Prompt.id + "_fp16w.json");
                s.Ref16 = File.Exists(ref16) ? ModelFiles.ReadJson<RefFile>(ref16, refName + "fp16w.json", "") : s.Ref32;
                if (s.Ref32.prompt == null || s.Ref32.prompt.token_ids == null || s.Ref32.answer == null || s.Ref32.answer.token_ids == null)
                    throw new ModelFiles.ToolError(refName + "fp32.json has no prompt or answer token IDs.");

                s.OnnxPath = Path.Combine(repo, s.Model.onnx_folder ?? "", "model.onnx");
                foreach (var f in s.Model.onnx_files ?? new ModelFiles.FileRecord[0])
                {
                    string path = Path.Combine(repo, s.Model.onnx_folder ?? "", f.file);
                    if (!File.Exists(path) || new FileInfo(path).Length != f.bytes)
                        throw new ModelFiles.ToolError(s.Model.onnx_folder + "/" + f.file + " is missing or has changed since model.json was written.");
                }
                s.SentisPath = Path.Combine(ModelFiles.ProjectRoot, ModelFiles.StreamingAssets, s.Model.sentis_file ?? "");
                s.Engine = ConvertModel.InferenceEngine.Find();

                foreach (string guid in AssetDatabase.FindAssets("t:UnityInferenceEngineProvider", new[] { folder }))
                {
                    var provider = AssetDatabase.LoadAssetAtPath<ScriptableObject>(AssetDatabase.GUIDToAssetPath(guid));
                    FieldInfo field = provider != null ? provider.GetType().GetField("llmConfig",
                        BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public) : null;
                    s.Config = field != null ? field.GetValue(provider) as OnDeviceLlmConfig : null;
                    if (s.Config != null) break;
                }
                return s;
            }

            /// <summary>A token as ID and text, like 90 '{"'.</summary>
            public string Show(int id)
            {
                string text = null;
                try { if (Config != null && Config.Tokenizer != null) text = Config.Tokenizer.Decode(new List<int> { id }); }
                catch (Exception) { text = null; }
                return text == null ? id.ToString(CultureInfo.InvariantCulture)
                                    : id.ToString(CultureInfo.InvariantCulture) + " '" + text.Replace("\n", "\\n") + "'";
            }
        }

        /// <summary>Runs one check so that an error in it is reported and the other checks still run.</summary>
        static string Guarded(string title, Func<string> check)
        {
            try { return check(); }
            catch (Exception e)
            {
                Debug.LogException(e);
                return title + ": stopped by " + e.GetType().Name + ": " + e.Message;
            }
        }

        static string TokenizerCheck(Setup s)
        {
            if (s.Config == null)
                return "1. Tokenizer: no provider asset in the model folder, so Meta's tokenizer can't be checked here.";
            Gpt2Tokenizer tokenizer = s.Config.Tokenizer;   // Meta starts it on first use; null if that failed
            if (tokenizer == null)
                return "1. Tokenizer: Meta's tokenizer didn't start. The provider has vocab: " + Name(s.Config.vocabFile) +
                       ", merges: " + Name(s.Config.mergesFile) + ", config: " + Name(s.Config.tokenizerConfigFile) +
                       ". The Console has Meta's error just above this report. The model checks don't need the tokenizer; " +
                       "they show token IDs without their text.";
            string formatted = s.Config.ApplyChatTemplate(s.Prompt.user, s.Config.defaultSystemMessage);
            List<int> ids = tokenizer.EncodeAsync(formatted).GetAwaiter().GetResult();   // encodes on a worker thread
            int[] reference = s.Ref32.prompt.token_ids;
            if (ids.SequenceEqual(reference))
                return "1. Tokenizer: Meta's template and tokenizer, with the provider's settings, give " + ids.Count +
                       " prompt IDs, the same as the reference.";
            int at = 0;
            while (at < ids.Count && at < reference.Length && ids[at] == reference[at]) at++;
            string here = at < ids.Count ? s.Show(ids[at]) : "(end)", there = at < reference.Length ? s.Show(reference[at]) : "(end)";
            return "1. Tokenizer: DIFFERENT. Meta's gives " + ids.Count + " prompt IDs, the reference " + reference.Length +
                   "; the first difference is at token " + at + ": here " + here + ", reference " + there + ".";
        }

        string ModelCheck(Setup s, string title, RefFile reference, Func<Model> load,
                          BackendType backend = BackendType.CPU, bool oneByOne = false)
        {
            var text = new StringBuilder(title + ":\n");
            Model model = null;
            try
            {
                EditorUtility.DisplayProgressBar("Model test", title + ": loading", 0.05f);
                var watch = Stopwatch.StartNew();
                model = load();
                text.AppendLine("   loaded in " + Seconds(watch.Elapsed.TotalSeconds));
                text.Append(Greedy(model, s, reference, tokens, (step, fraction) =>
                    EditorUtility.DisplayProgressBar("Model test", title + ": " + step, fraction), backend, oneByOne));
            }
            catch (ModelFiles.ToolError e)
            {
                text.AppendLine("   stopped: " + e.Message);
            }
            catch (Exception e)
            {
                text.AppendLine("   stopped by " + e.GetType().Name + ": " + e.Message);
            }
            finally
            {
                if (model != null) s.Engine.DisposeWeights(model);
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
            return text.ToString();
        }

        /// <summary>
        /// Greedy decoding like grounding/meta_runner.py: the whole prompt first (positions 0 to n-1, a mask of ones, empty
        /// caches), then one token at a time at the next position, with the caches the previous pass returned. With
        /// oneByOne, the prompt too goes in one token at a time, so no pass ever handles more than one token.
        /// </summary>
        internal static string Greedy(Model model, Setup s, RefFile reference, int maxTokens, Action<string, float> progress,
                                      BackendType backend = BackendType.CPU, bool oneByOne = false)
        {
            var text = new StringBuilder();
            int[] prompt = reference.prompt.token_ids, expected = reference.answer.token_ids;
            int n = prompt.Length, eos = s.Arch.eos_token_id;
            var worker = new Worker(model, backend);
            var past = new Tensor<float>[2 * s.Arch.max_layers];
            try
            {
                var empty = new TensorShape(1, s.Arch.num_key_value_heads, 0, s.Arch.head_dim);
                for (int i = 0; i < past.Length; i++) past[i] = new Tensor<float>(empty);

                var watch = Stopwatch.StartNew();
                Scores scores;
                if (oneByOne)
                {
                    scores = null;
                    for (int i = 0; i < n; i++)
                    {
                        if (i % 10 == 0) progress("prompt token " + (i + 1) + " of " + n, 0.1f * i / n);
                        scores = Step(worker, new[] { prompt[i] }, new[] { i }, i + 1, past, s.Arch);
                    }
                    text.AppendLine("   prompt fed one token at a time (" + n + " passes, the reference's IDs): " +
                                    Seconds(watch.Elapsed.TotalSeconds) + "; after the last one " + scores.Describe(s));
                }
                else
                {
                    progress("the prompt pass (" + n + " tokens)", 0.1f);
                    scores = Step(worker, prompt, Enumerable.Range(0, n).ToArray(), n, past, s.Arch);
                    text.AppendLine("   prompt pass (" + n + " tokens, the reference's IDs): " + Seconds(watch.Elapsed.TotalSeconds) +
                                    "; at the last position " + scores.Describe(s));
                }

                watch.Reset();
                watch.Start();
                int firstDifference = -1, steps = 0, generated = 0;
                bool ended = false;
                for (int step = 0; step < maxTokens; step++)
                {
                    int next = scores.ArgMax, want = step < expected.Length ? expected[step] : eos;
                    bool same = next == want;
                    if (!same && firstDifference < 0) firstDifference = step;
                    if (next == eos)
                    {
                        text.AppendLine("   token " + step + ": the end token" + (same ? ", like the reference" : ", but the reference goes on with " + s.Show(want)));
                        ended = true;
                        break;
                    }
                    generated++;
                    text.AppendLine("   token " + step + ": " + s.Show(next) + (same ? "" : "   DIFFERS; the reference has " +
                                    (want == eos ? "the end token" : s.Show(want)) + "; this step's top 5: " + scores.Top(s)));
                    if (firstDifference >= 0 && step - firstDifference >= 3) break;   // a few tokens past the first difference
                    progress("token " + (step + 1), 0.1f + 0.9f * (step + 1) / maxTokens);
                    int length = n + step + 1;
                    scores = Step(worker, new[] { next }, new[] { length - 1 }, length, past, s.Arch);
                    steps++;
                }
                if (steps > 0) text.AppendLine("   one token every " + Seconds(watch.Elapsed.TotalSeconds / steps));
                if (firstDifference < 0)
                    text.AppendLine(ended ? "   Result: the same answer as the reference, " + generated + " tokens and then the end token."
                                          : "   Result: the first " + generated + " tokens equal the reference's.");
                else
                    text.AppendLine("   Result: DIFFERENT from token " + firstDifference + " on.");
            }
            finally
            {
                foreach (var t in past) if (t != null) t.Dispose();
                worker.Dispose();
            }
            return text.ToString();
        }

        /// <summary>One pass of the model; returns the scores at the last position and keeps the new caches.</summary>
        static Scores Step(Worker worker, int[] ids, int[] positions, int maskLength, Tensor<float>[] past, FillChatProvider.Architecture arch)
        {
            var mask = Enumerable.Repeat(1, maskLength).ToArray();
            using (var input = new Tensor<int>(new TensorShape(1, ids.Length), ids))
            using (var attention = new Tensor<int>(new TensorShape(1, maskLength), mask))
            using (var position = new Tensor<int>(new TensorShape(1, positions.Length), positions))
            {
                worker.SetInput("input_ids", input);
                worker.SetInput("attention_mask", attention);
                worker.SetInput("position_ids", position);
                for (int i = 0; i < arch.max_layers; i++)
                {
                    worker.SetInput("past_key_values." + i + ".key", past[2 * i]);
                    worker.SetInput("past_key_values." + i + ".value", past[2 * i + 1]);
                }
                worker.Schedule();
                var logits = worker.PeekOutput("logits") as Tensor<float>;
                float[] data = logits.DownloadToArray();
                int vocab = logits.shape[2], rows = logits.shape[1];
                var scores = new Scores(data, (rows - 1) * vocab, vocab);
                for (int i = 0; i < arch.max_layers; i++)
                {
                    past[2 * i] = Copy(past[2 * i], worker.PeekOutput("present." + i + ".key") as Tensor<float>);
                    past[2 * i + 1] = Copy(past[2 * i + 1], worker.PeekOutput("present." + i + ".value") as Tensor<float>);
                }
                return scores;
            }
        }

        static Tensor<float> Copy(Tensor<float> old, Tensor<float> present)
        {
            var copy = new Tensor<float>(present.shape, present.DownloadToArray());
            old.Dispose();
            return copy;
        }

        /// <summary>The scores of one position: the highest, the top five, and how many aren't finite numbers.</summary>
        sealed class Scores
        {
            public readonly int ArgMax, NotFinite;
            readonly int[] top = new int[5];
            readonly float[] topValues = new float[5];

            public Scores(float[] data, int start, int vocab)
            {
                for (int k = 0; k < 5; k++) { top[k] = -1; topValues[k] = float.NegativeInfinity; }
                for (int i = 0; i < vocab; i++)
                {
                    float v = data[start + i];
                    if (float.IsNaN(v) || float.IsInfinity(v)) { NotFinite++; continue; }
                    for (int k = 0; k < 5; k++)
                    {
                        if (v <= topValues[k]) continue;
                        for (int j = 4; j > k; j--) { top[j] = top[j - 1]; topValues[j] = topValues[j - 1]; }
                        top[k] = i;
                        topValues[k] = v;
                        break;
                    }
                }
                ArgMax = top[0] >= 0 ? top[0] : 0;
            }

            public string Top(Setup s)
            {
                return string.Join(" | ", Enumerable.Range(0, 5).Where(k => top[k] >= 0)
                    .Select(k => s.Show(top[k]) + " " + topValues[k].ToString("F2", CultureInfo.InvariantCulture)).ToArray());
            }

            public string Describe(Setup s)
            {
                return (NotFinite == 0 ? "all scores are finite" : NotFinite + " scores are NaN or infinite") + "; top 5: " + Top(s);
            }
        }

        static string Name(UnityEngine.Object asset)
        {
            return asset != null ? asset.name : "none";
        }

        static string Seconds(double s)
        {
            return s.ToString("F2", CultureInfo.InvariantCulture) + " s";
        }

        static string SaveReport(string run, string text)
        {
            string dir = Path.Combine(ModelFiles.RepoRoot, "runs", run);
            if (!File.Exists(Path.Combine(dir, "config.yaml")))
                return "\nNot saved: there is no run " + run + " in runs/.\n";
            string raw = Path.Combine(dir, "raw");
            Directory.CreateDirectory(raw);
            string file = "model-test_" + DateTime.UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'", CultureInfo.InvariantCulture) + ".txt";
            File.WriteAllText(Path.Combine(raw, file), text, new UTF8Encoding(false));
            return "\nSaved as runs/" + run + "/raw/" + file + ".\n";
        }
    }
}
#endif
