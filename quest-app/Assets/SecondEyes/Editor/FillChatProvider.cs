// Second Eyes > Fill chat provider (A1.7c-1, D37): fills Meta's on-device chat provider from the repository's files.
#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace SecondEyes.EditorTools
{
    /// <summary>
    /// Fills the selected Unity Inference Engine provider asset (Meta's AI Building Blocks) for on-device chat, so that
    /// no value is typed by hand. The asset must be in Assets/SecondEyes/Models/NAME/, next to the files that
    /// grounding/copy_to_unity.py wrote there. The values come from:
    ///   grounding/models/NAME.json        the model's shape, chat template, token limits and provider settings
    ///   grounding/prompts/a17-fixed.json  the system message (A1.7c's fixed prompt)
    ///   Assets/SecondEyes/Models/NAME/    the tokenizer files, checked against model.json's fingerprints
    ///   Assets/StreamingAssets/           the .sentis file that model.json names (D34), made by Second Eyes > Convert model
    /// Everything is checked before anything is written, and the Console lists every setting before and after.
    /// The setting names are those of Meta XR SDK v207 (docs/meta-ai.md); if Meta renames one, this stops and says so.
    /// </summary>
    public static class FillChatProvider
    {
        const string MenuPath = "Second Eyes/Fill chat provider";
        const string ProviderClass = "UnityInferenceEngineProvider";
        const string PromptFile = "grounding/prompts/a17-fixed.json";   // A1.7c: its system message goes into the provider
        static readonly string[] TokenizerFiles = { "vocab.json", "merges.txt", "tokenizer_config.json" };
        static readonly string[] NotForChat = { "backend", "splitOverFrames", "layersPerFrame" };   // reported, not changed

        // The parts of our JSON files that are read here (JsonUtility ignores the rest).
        [Serializable] public class Architecture { public int max_layers, num_key_value_heads, head_dim, eos_token_id; }
        [Serializable] public class ProviderSettings { public string backend, execution_mode; public int steps_per_frame; }
        [Serializable] public class Description
        {
            public string name, chat_template;
            public int max_new_tokens, max_prompt_length;
            public Architecture architecture;
            public ProviderSettings provider;
        }
        [Serializable] public class Prompt { public string id, system; }

        sealed class Setting
        {
            public string Key, Before, After;
            public Action Write;
        }

        [MenuItem(MenuPath, false, 2)]
        static void FillSelected()
        {
            try
            {
                Debug.Log(Fill(Selection.activeObject));
                EditorUtility.DisplayDialog("Fill chat provider", "Done. The Console lists every setting.", "OK");
            }
            catch (ModelFiles.ToolError e)
            {
                Debug.LogError("[Fill chat provider] " + e.Message);
                EditorUtility.DisplayDialog("Fill chat provider", e.Message, "OK");
            }
        }

        /// <summary>Fills the provider and returns the report. Throws ToolError, having changed nothing, if anything is wrong.</summary>
        public static string Fill(Object selected)
        {
            if (selected == null || selected.GetType().Name != ProviderClass)
                throw new ModelFiles.ToolError("Select the provider asset first: Meta's Unity Inference Engine provider, in " +
                                    ModelFiles.ModelsFolder + "/<model name>/.");
            string assetPath = AssetDatabase.GetAssetPath(selected);
            string folder = (Path.GetDirectoryName(assetPath) ?? "").Replace('\\', '/');
            string name = Path.GetFileName(folder);
            if ((Path.GetDirectoryName(folder) ?? "").Replace('\\', '/') != ModelFiles.ModelsFolder)
                throw new ModelFiles.ToolError("The provider asset must be in " + ModelFiles.ModelsFolder + "/<model name>/, next to that model's " +
                                    "files; this one is in " + folder + "/.");
            var so = new SerializedObject(selected);
            if (so.FindProperty("mode") == null || so.FindProperty("llmConfig") == null)
                throw new ModelFiles.ToolError("This provider has no chat settings. Meta's chat code only switches on when Unity's " +
                                    "Inference Engine package (com.unity.ai.inference) 2.2.1 or newer is installed (D36).");

            string project = ModelFiles.ProjectRoot, repo = ModelFiles.RepoRoot;
            string descFile = "grounding/models/" + name + ".json";
            string copyHint = "run: python grounding/copy_to_unity.py " + descFile;
            var desc = ModelFiles.ReadJson<Description>(Path.Combine(repo, descFile), descFile, "Every model needs a description there.");
            var prompt = ModelFiles.ReadJson<Prompt>(Path.Combine(repo, PromptFile), PromptFile, "");
            var model = ModelFiles.ReadModelJson(folder);

            // Check every input before touching the asset.
            var problems = new List<string>();
            if (desc.name != name)
                problems.Add(descFile + " names the model '" + desc.name + "', but the asset's folder is '" + name + "'");
            if (model.name != name)
                problems.Add(folder + "/model.json is for '" + model.name + "'; " + copyHint);
            var arch = desc.architecture;
            if (arch == null || arch.max_layers <= 0 || arch.num_key_value_heads <= 0 || arch.head_dim <= 0 || arch.eos_token_id <= 0)
                problems.Add(descFile + " has no complete architecture section");
            var run = desc.provider;
            if (run == null || string.IsNullOrEmpty(run.backend) || string.IsNullOrEmpty(run.execution_mode) || run.steps_per_frame <= 0)
                problems.Add(descFile + " has no complete provider section (backend, execution_mode, steps_per_frame)");
            if (desc.max_new_tokens <= 0 || desc.max_prompt_length <= 0)
                problems.Add(descFile + " has no max_new_tokens or max_prompt_length");
            if (!TemplateWorks(desc.chat_template))
                problems.Add(descFile + "'s chat_template needs {0} for the user text and {1} for the system message, and no other braces");
            if (string.IsNullOrEmpty(prompt.system))
                problems.Add(PromptFile + " has no system message");

            Object sentis = null;
            string sentisPath = "Assets/StreamingAssets/" + model.sentis_file;
            if (string.IsNullOrEmpty(model.sentis_file) || model.sentis_file.IndexOfAny(new[] { '/', '\\' }) >= 0)
                problems.Add(folder + "/model.json names no .sentis file; " + copyHint);
            else if ((sentis = AssetDatabase.LoadAssetAtPath<Object>(sentisPath)) == null)
                problems.Add(sentisPath + " isn't there. Convert the model first: select " + folder +
                             " and run Second Eyes > Convert model (docs/setup/quest-model.md)");

            var texts = new Dictionary<string, TextAsset>();
            foreach (string file in TokenizerFiles)
            {
                string path = folder + "/" + file;
                string expected = null;
                if (model.tokenizer_files != null)
                    foreach (var t in model.tokenizer_files)
                        if (t != null && t.file == file) expected = t.sha256;
                var text = AssetDatabase.LoadAssetAtPath<TextAsset>(path);
                if (text == null)
                    problems.Add(path + " isn't there; " + copyHint);
                else if (expected == null)
                    problems.Add(folder + "/model.json has no fingerprint for " + file + "; " + copyHint);
                else if (ModelFiles.Sha256(Path.Combine(project, path)) != expected)
                    problems.Add(path + " isn't the file copy_to_unity.py wrote (its fingerprint differs from model.json); " +
                                 copyHint + " instead of editing it");
                else
                    texts[file] = text;
            }

            var settings = new List<Setting>();
            SetEnum(so, settings, problems, "mode", "Chat");
            SetObject(so, settings, problems, "modelFile", null);   // an embedded model would go into the app a second time
            SetBool(so, settings, problems, "useStreamingAsset", true);
            SetObject(so, settings, problems, "streamingAssetModel", sentis);
            SetString(so, settings, problems, "streamingAssetFileName", model.sentis_file);   // hidden; Meta's inspector sets the same
            if (run != null)
            {
                SetEnum(so, settings, problems, "llmConfig.backendType", run.backend);
                SetEnum(so, settings, problems, "llmConfig.inferenceExecutionMode", run.execution_mode);
                SetInt(so, settings, problems, "llmConfig.stepsPerFrame", run.steps_per_frame);
            }
            SetObject(so, settings, problems, "llmConfig.vocabFile", Get(texts, "vocab.json"));
            SetObject(so, settings, problems, "llmConfig.mergesFile", Get(texts, "merges.txt"));
            SetObject(so, settings, problems, "llmConfig.tokenizerConfigFile", Get(texts, "tokenizer_config.json"));
            SetString(so, settings, problems, "llmConfig.chatTemplateFormat", desc.chat_template);
            SetString(so, settings, problems, "llmConfig.defaultSystemMessage", prompt.system);
            if (arch != null)
            {
                SetInt(so, settings, problems, "llmConfig.maxLayers", arch.max_layers);
                SetInt(so, settings, problems, "llmConfig.numKeyValueHeads", arch.num_key_value_heads);
                SetInt(so, settings, problems, "llmConfig.headDim", arch.head_dim);
                SetInt(so, settings, problems, "llmConfig.eosTokenId", arch.eos_token_id);
            }
            SetInt(so, settings, problems, "llmConfig.maxNewTokens", desc.max_new_tokens);
            SetInt(so, settings, problems, "llmConfig.maxPromptLength", desc.max_prompt_length);

            if (problems.Count > 0)
                throw new ModelFiles.ToolError("Nothing was changed. Fix these first:\n- " + string.Join("\n- ", problems.ToArray()));

            foreach (var s in settings) s.Write();
            so.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();

            var report = new StringBuilder();
            report.Append("[Fill chat provider] Filled ").Append(assetPath).Append(" from ").Append(descFile).Append(", ")
                  .Append(PromptFile).Append(" and ").Append(folder).Append("/model.json:");
            foreach (var s in settings)
                report.Append("\n  ").Append(s.Key).Append(": ")
                      .Append(s.Before == s.After ? s.After + " (unchanged)" : s.Before + " -> " + s.After);
            var left = new List<string>();
            foreach (string key in NotForChat)
            {
                var p = so.FindProperty(key);
                if (p != null) left.Add(key + " = " + Show(p));
            }
            if (left.Count > 0)
                report.Append("\nNot used for chat, left as they are: ").Append(string.Join(", ", left.ToArray()));
            return report.ToString();
        }

        static bool TemplateWorks(string template)
        {
            if (string.IsNullOrEmpty(template) || !template.Contains("{0}") || !template.Contains("{1}")) return false;
            try { string.Format(template, "user", "system"); return true; }
            catch (FormatException) { return false; }
        }

        static TextAsset Get(Dictionary<string, TextAsset> texts, string file)
        {
            TextAsset text;
            texts.TryGetValue(file, out text);
            return text;
        }

        static string Quote(string s) { return s == null ? "none" : "\"" + s.Replace("\n", "\\n") + "\""; }

        static string Describe(Object o) { return o == null ? "none" : AssetDatabase.GetAssetPath(o); }

        static string Show(SerializedProperty p)
        {
            switch (p.propertyType)
            {
                case SerializedPropertyType.Enum:
                    return p.enumValueIndex >= 0 && p.enumValueIndex < p.enumNames.Length ? p.enumNames[p.enumValueIndex] : "?";
                case SerializedPropertyType.Integer: return p.intValue.ToString();
                case SerializedPropertyType.Boolean: return p.boolValue ? "true" : "false";
                case SerializedPropertyType.String: return Quote(p.stringValue);
                case SerializedPropertyType.ObjectReference: return Describe(p.objectReferenceValue);
                default: return p.propertyType.ToString();
            }
        }

        static SerializedProperty Find(SerializedObject so, List<string> problems, string key, SerializedPropertyType type)
        {
            var p = so.FindProperty(key);
            if (p == null)
                problems.Add("Meta's provider has no setting '" + key + "'; the SDK may have changed since v207 (docs/meta-ai.md)");
            else if (p.propertyType != type)
                problems.Add("'" + key + "' is a " + p.propertyType + " setting, not " + type + "; the SDK may have changed since v207");
            else
                return p;
            return null;
        }

        static void SetEnum(SerializedObject so, List<Setting> settings, List<string> problems, string key, string value)
        {
            var p = Find(so, problems, key, SerializedPropertyType.Enum);
            if (p == null) return;
            int index = Array.IndexOf(p.enumNames, value);
            if (index < 0)
            {
                problems.Add("'" + key + "' has no option '" + value + "' (it has " + string.Join(", ", p.enumNames) + ")");
                return;
            }
            settings.Add(new Setting { Key = key, Before = Show(p), After = value, Write = () => p.enumValueIndex = index });
        }

        static void SetInt(SerializedObject so, List<Setting> settings, List<string> problems, string key, int value)
        {
            var p = Find(so, problems, key, SerializedPropertyType.Integer);
            if (p != null)
                settings.Add(new Setting { Key = key, Before = Show(p), After = value.ToString(), Write = () => p.intValue = value });
        }

        static void SetBool(SerializedObject so, List<Setting> settings, List<string> problems, string key, bool value)
        {
            var p = Find(so, problems, key, SerializedPropertyType.Boolean);
            if (p != null)
                settings.Add(new Setting { Key = key, Before = Show(p), After = value ? "true" : "false", Write = () => p.boolValue = value });
        }

        static void SetString(SerializedObject so, List<Setting> settings, List<string> problems, string key, string value)
        {
            var p = Find(so, problems, key, SerializedPropertyType.String);
            if (p != null)
                settings.Add(new Setting { Key = key, Before = Show(p), After = Quote(value), Write = () => p.stringValue = value });
        }

        static void SetObject(SerializedObject so, List<Setting> settings, List<string> problems, string key, Object value)
        {
            var p = Find(so, problems, key, SerializedPropertyType.ObjectReference);
            if (p != null)
                settings.Add(new Setting { Key = key, Before = Show(p), After = Describe(value), Write = () => p.objectReferenceValue = value });
        }
    }
}
#endif
