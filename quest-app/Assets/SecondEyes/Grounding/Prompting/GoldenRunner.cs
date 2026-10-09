using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SecondEyes.Grounding.Replay;
using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.Grounding.Prompting
{
    /// <summary>
    /// A2.5 delivery 2 (D104(4)): the on-device golden self-check. The pushed goldens in files/prompting/goldens are
    /// checked against their golden manifest, file by file. Then, for every golden, the document, mapping, mapping hash and
    /// prompt are rebuilt from the shipped snapshot and command records and compared with the PC's, and the runtime's own
    /// tokenization of the prompt is compared with the PC's token IDs. The model is loaded only for its tokenizer: nothing
    /// is evaluated. Everything goes into a new folder files/prompting/results/(UTC time): results.jsonl, identity.json
    /// and done.json, written last.
    /// </summary>
    public sealed class GoldenRunner : IDisposable
    {
        public const int ContextTokens = 1024;   // tokenization only

        public static string GoldenDir { get { return Path.Combine(Application.persistentDataPath, "prompting", "goldens"); } }
        public static string ResultsRoot { get { return Path.Combine(Application.persistentDataPath, "prompting", "results"); } }
        public static bool GoldensPresent { get { return File.Exists(Path.Combine(GoldenDir, "golden-manifest.json")); } }

        private readonly SynchronizationContext main;
        private readonly Action<string> status;
        private LlamaRuntime runtime;

        public GoldenRunner(SynchronizationContext main, Action<string> status)
        {
            this.main = main;
            this.status = status;
        }

        private void Say(string text) { main.Post(_ => status(text), null); }
        private void Event(string ev, string json) { main.Post(_ => EventLog.Write(ev, json), null); }

        internal sealed class Goldens
        {
            public string ManifestSha256, AssetSha256;
            public PromptAsset Asset;
            public List<JNode> Rows = new List<JNode>();
            public Dictionary<string, string> Texts = new Dictionary<string, string>();
        }

        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false, true);

        /// <summary>Every file against the golden manifest; throws with the reason.</summary>
        internal static Goldens Read(string dir)
        {
            var g = new Goldens();
            byte[] mb = File.ReadAllBytes(Path.Combine(dir, "golden-manifest.json"));
            g.ManifestSha256 = ReplayHash.Sha256Hex(mb);
            JNode man = JParse.Parse(Utf8.GetString(mb));
            foreach (var f in man["files"].Fields)
            {
                string path = Path.Combine(dir, f.Key);
                if (!File.Exists(path) || ReplayHash.Sha256Hex(File.ReadAllBytes(path)) != f.Value.Text)
                    throw new InvalidOperationException(f.Key + " is missing or differs from the golden manifest: push the goldens again");
            }
            byte[] ab = File.ReadAllBytes(Path.Combine(dir, "prompt-asset.json"));
            g.AssetSha256 = ReplayHash.Sha256Hex(ab);
            if (g.AssetSha256 != man["prompt_asset_sha256"].Text)
                throw new InvalidOperationException("the prompt asset is not the one the golden manifest pins");
            g.Asset = PromptAsset.From(JParse.Parse(Utf8.GetString(ab)));
            foreach (string line in Utf8.GetString(File.ReadAllBytes(Path.Combine(dir, "goldens.jsonl"))).Split('\n'))
            {
                if (line.Trim().Length == 0) continue;
                JNode row = JParse.Parse(line);
                g.Rows.Add(row);
                foreach (string key in new[] { "scene_file", "command_file" })
                {
                    string rel = row[key].Text;
                    if (!g.Texts.ContainsKey(rel)) g.Texts[rel] = Utf8.GetString(File.ReadAllBytes(Path.Combine(dir, rel)));
                }
            }
            if (g.Rows.Count.ToString(CultureInfo.InvariantCulture) != man["goldens"].Text)
                throw new InvalidOperationException("goldens.jsonl does not hold the manifest's number of goldens");
            return g;
        }

        private static int[] Tokenize(IntPtr s, byte[] text)
        {
            var z = new byte[text.Length + 1];
            Buffer.BlockCopy(text, 0, z, 0, text.Length);
            var buf = new int[text.Length + 16];
            int count = LlamaNative.se_tokenize(s, z, buf, buf.Length);
            if (count < 0)
            {
                buf = new int[-count];
                count = LlamaNative.se_tokenize(s, z, buf, buf.Length);
            }
            if (count < 0) throw new InvalidOperationException("se_tokenize failed: " + LlamaNative.LastError());
            var ids = new int[count];
            Array.Copy(buf, ids, count);
            return ids;
        }

        private static void WriteJson(string path, string json) { File.WriteAllText(path, json + "\n", new UTF8Encoding(false)); }

        /// <summary>Runs the whole check and returns a one-line summary for the panel. Call on the main thread.</summary>
        public async Task<string> RunAsync(string modelPath)
        {
            string stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            string goldenDir = GoldenDir, dir = Path.Combine(ResultsRoot, stamp), appVersion = Application.version;
            if (Directory.Exists(dir))
                return "Golden check not started: results folder " + stamp + " already exists. Nothing was written; press again.";
            Directory.CreateDirectory(dir);
            var total = Stopwatch.StartNew();
            Say("Golden check: reading the goldens...");
            Goldens g = await Task.Run(() => Read(goldenDir));
            var checks = PromptSelfChecks.Run();
            bool checksOk = checks.TrueForAll(c => c.Passed);
            var sj = new JsonWriter().BeginObj().Key("record_type").S("a25_prompt_selfchecks").Key("checks").BeginArr();
            foreach (var c in checks) sj.BeginObj().Key("name").S(c.Name).Key("passed").B(c.Passed).Key("detail").S(c.Detail).End();
            WriteJson(Path.Combine(dir, "selfchecks.json"), sj.End().Key("all_passed").B(checksOk).End().ToString());
            if (!checksOk)
                return "Golden check stopped: a prompt self-check failed (prompting/results/" + stamp + "/selfchecks.json).";
            Event("prompting.golden.start", new JsonWriter().BeginObj().Key("results").S(stamp).Key("goldens").I(g.Rows.Count)
                .Key("golden_manifest_sha256").S(g.ManifestSha256).End().ToString());
            Say("Golden check: loading the model for its tokenizer...");
            runtime = new LlamaRuntime(main);
            double loadMs = await runtime.LoadAsync(modelPath, ContextTokens, ReplayRunner.Threads, ReplayRunner.Flags);
            Say("Golden check: rebuilding " + g.Rows.Count + " documents and prompts...");
            var results = await runtime.WithModel(s =>
            {
                var list = new List<GoldenResult>();
                foreach (JNode row in g.Rows)
                    list.Add(GoldenCheck.Check(g.Asset, row, g.Texts[row["scene_file"].Text], g.Texts[row["command_file"].Text],
                                               bytes => Tokenize(s, bytes)));
                return list;
            });
            int ok = 0;
            var lines = new StringBuilder();
            foreach (GoldenResult r in results)
            {
                if (r.AllOk) ok++;
                lines.Append(new JsonWriter().BeginObj().Key("record_type").S("a25_golden_result").Key("request_id").S(r.RequestId)
                    .Key("kind").S(r.Kind).Key("all_ok").B(r.AllOk).Key("document_ok").B(r.DocumentOk).Key("mapping_ok").B(r.MappingOk)
                    .Key("mapping_hash_ok").B(r.MappingHashOk).Key("prompt_ok").B(r.PromptOk)
                    .Key("tokens_ok").S(r.TokensOk == null ? "skipped" : r.TokensOk == true ? "yes" : "no")
                    .Key("runtime_tokens").I(r.RuntimeTokens).Key("first_token_difference").I(r.FirstTokenDifference)
                    .Key("built_document_sha256").S(r.BuiltDocumentSha256).Key("built_prompt_sha256").S(r.BuiltPromptSha256)
                    .Key("error").S(r.Error).End().ToString()).Append('\n');
            }
            File.WriteAllText(Path.Combine(dir, "results.jsonl"), lines.ToString(), new UTF8Encoding(false));
            WriteJson(Path.Combine(dir, "identity.json"), new JsonWriter().BeginObj().Key("record_type").S("a25_golden_identity")
                .Key("results").S(stamp).Key("golden_manifest_sha256").S(g.ManifestSha256).Key("prompt_asset_sha256").S(g.AssetSha256)
                .Key("goldens").I(g.Rows.Count).Key("model").S(modelPath).Key("context_tokens").I(ContextTokens)
                .Key("threads").S("requested and passed to the loader; not read back by the runtime")
                .Key("llama_cpp").S(runtime.Version).Key("load_ms").D(loadMs).Key("app_version").S(appVersion)
                .Key("event_log").S(EventLog.FilePath).Key("session").S(EventLog.SessionId).End().ToString());
            WriteJson(Path.Combine(dir, "done.json"), new JsonWriter().BeginObj().Key("record_type").S("a25_golden_done")
                .Key("results").S(stamp).Key("goldens").I(g.Rows.Count).Key("checked").I(results.Count).Key("all_ok").I(ok)
                .Key("self_checks_passed").B(checksOk)
                .Key("total_ms").D(total.Elapsed.TotalMilliseconds)
                .Key("finished_utc").S(DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture)).End().ToString());
            Event("prompting.golden.end", new JsonWriter().BeginObj().Key("results").S(stamp).Key("checked").I(results.Count)
                .Key("all_ok").I(ok).End().ToString());
            return $"Golden check: {ok}/{results.Count} goldens equal the PC's (document, prompt, mapping, tokens). "
                 + "Results: prompting/results/" + stamp;
        }

        public void Dispose()
        {
            if (runtime != null) runtime.Dispose();
            runtime = null;
        }
    }
}
