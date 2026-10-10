using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Replay;
using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.Grounding.Interactive
{
    /// <summary>The pipeline's native calls on the loaded model.</summary>
    internal sealed class LlamaAdapter : INative
    {
        private readonly IntPtr s;
        private readonly int[] buf = new int[16384];

        public LlamaAdapter(IntPtr s) { this.s = s; }

        public int[] Tokenize(byte[] utf8)
        {
            var z = new byte[utf8.Length + 1];
            Buffer.BlockCopy(utf8, 0, z, 0, utf8.Length);
            int n = LlamaNative.se_tokenize(s, z, buf, buf.Length);
            if (n <= 0) return new int[0];
            var t = new int[n];
            Array.Copy(buf, t, n);
            return t;
        }

        public int Eval(int[] tokens, int keep) { return LlamaNative.se_eval(s, tokens, tokens.Length, keep); }
        public int Cached() { return LlamaNative.se_n_cached(s); }
        public int Logits(float[] row) { return LlamaNative.se_logits(s, row, row.Length); }
        public double LogProb(int token) { return LlamaNative.se_logprob(s, token); }
        public string LastError() { return LlamaNative.LastError(); }
    }

    /// <summary>A2.5 delivery 3 on the headset: one model load, the pipeline on the worker thread, every outcome appended
    /// to a JSONL file as it completes. The Interactive check sends the 32 verified golden requests through the pipeline
    /// twice, cache off and then exact prefix. Everything goes into a new folder files/interactive/results/(UTC time):
    /// identity.json, outcomes-off.jsonl, outcomes-prefix.jsonl and done.json, written last.</summary>
    public sealed class InteractiveRunner
    {
        public static string ResultsRoot { get { return Path.Combine(Application.persistentDataPath, "interactive", "results"); } }

        private readonly SynchronizationContext main;
        private readonly Action<string> status;
        private LlamaRuntime runtime;

        public InteractiveRunner(SynchronizationContext main, Action<string> status)
        {
            this.main = main;
            this.status = status;
        }

        private void Say(string text) { main.Post(_ => status(text), null); }
        private void Event(string ev, string json) { main.Post(_ => EventLog.Write(ev, json), null); }
        private static void WriteJson(string path, string json) { File.WriteAllText(path, json + "\n", new UTF8Encoding(false)); }

        private static void WriteJson(string path, JsonWriter open, StartupGuard.Result guard) { WriteJson(path, guard.WriteTo(open).End().ToString()); }

        public void Dispose() { if (runtime != null) runtime.Dispose(); runtime = null; }

        private async Task<List<InteractiveOutcome>> Pass(GoldenRunner.Goldens g, CacheMode mode, string file, string stamp)
        {
            var cache = new PrefixCache(mode);
            var outcomes = new List<InteractiveOutcome>();
            using (var w = new StreamWriter(file, false, new UTF8Encoding(false)))
            {
                foreach (JNode row in g.Rows)
                {
                    string scene = g.Texts[row["scene_file"].Text];
                    var r = new InteractiveRequest {
                        RequestId = row["request_id"].Text, Source = "golden-check", SnapshotId = row["snapshot_id"].Text,
                        SnapshotSha256 = ReplayHash.Sha256Hex(Encoding.UTF8.GetBytes(scene)), SceneText = scene,
                        CommandText = g.Texts[row["command_file"].Text] };
                    InteractiveOutcome o = await runtime.WithModel(s => {
                        var vec = new float[LlamaNative.se_n_vocab(s)];
                        return Pipeline.Run(r, g.Asset, new LlamaAdapter(s), cache, ExecutionPurpose.Diagnostic, vec, null);
                    });
                    w.WriteLine(o.ToJson());
                    w.Flush();
                    outcomes.Add(o);
                    Event("interactive.outcome", new JsonWriter().BeginObj().Key("results").S(stamp).Key("request_id").S(o.RequestId)
                        .Key("cache_mode").S(o.CacheMode).Key("status").S(o.Status).Key("kept_tokens").I(o.KeptTokens)
                        .Key("total_ms").D(o.TotalMs).End().ToString());
                    Say("Interactive check (" + mode + "): " + outcomes.Count + "/" + g.Rows.Count + " ...");
                }
            }
            return outcomes;
        }

        public async Task<string> CheckAsync(string modelPath)
        {
            string stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            string dir = Path.Combine(ResultsRoot, stamp), goldenDir = GoldenRunner.GoldenDir, appVersion = Application.version;
            if (Directory.Exists(dir))
                return "Interactive check not started: results folder " + stamp + " already exists. Nothing was written; press again.";
            Directory.CreateDirectory(dir);
            var total = Stopwatch.StartNew();
            Say("Interactive check: reading the goldens...");
            GoldenRunner.Goldens g = await Task.Run(() => GoldenRunner.Read(goldenDir));
            Event("interactive.start", new JsonWriter().BeginObj().Key("results").S(stamp).Key("requests").I(g.Rows.Count)
                .Key("golden_manifest_sha256").S(g.ManifestSha256).End().ToString());
            Say("Interactive check: loading the model...");
            runtime = new LlamaRuntime(main);
            double loadMs = await runtime.LoadAsync(modelPath, Pipeline.ContextTokens, ReplayRunner.Threads, ReplayRunner.Flags);
            StartupGuard.Result guard = await runtime.WithModel(s => StartupGuard.Run(g.Asset, g.Rows, g.Texts, b => new LlamaAdapter(s).Tokenize(b)));
            Event("interactive.startup_check", guard.WriteTo(new JsonWriter().BeginObj().Key("results").S(stamp)).End().ToString());
            if (!guard.Ok)
            {
                WriteJson(Path.Combine(dir, "identity.json"), guard.WriteTo(new JsonWriter().BeginObj().Key("record_type").S("a25_interactive_identity")
                    .Key("results").S(stamp).Key("golden_manifest_sha256").S(g.ManifestSha256)).End().ToString());
                return "Interactive check refused: the startup check failed (" + guard.FirstFailure + "); nothing was evaluated.";
            }
            var off = await Pass(g, CacheMode.Off, Path.Combine(dir, "outcomes-off.jsonl"), stamp);
            var prefix = await Pass(g, CacheMode.ExactPrefix, Path.Combine(dir, "outcomes-prefix.jsonl"), stamp);
            int promptsEqual = 0, sameChoice = 0, sameOffered = 0, kept = 0, completedOrAsk = 0;
            Func<List<InteractiveOutcome>, string, int> n = (list, f) => list.FindAll(x => x.Status == f || x.AskBasis == f
                                                                                    || "path_" + x.ExecutionPath == f).Count;
            for (int k = 0; k < g.Rows.Count; k++)
            {
                if (off[k].PromptSha256 == g.Rows[k]["prompt_sha256"].Text && prefix[k].PromptSha256 == off[k].PromptSha256) promptsEqual++;
                if (off[k].ChoiceCode != null && off[k].ChoiceCode == prefix[k].ChoiceCode) sameChoice++;
                if (off[k].Logits.Count > 0 && System.Linq.Enumerable.SequenceEqual(off[k].Logits, prefix[k].Logits)) sameOffered++;
                if (prefix[k].KeptTokens > 0) kept++;
                if ((off[k].Status == "completed" || off[k].Status == "ask") && (prefix[k].Status == "completed" || prefix[k].Status == "ask")) completedOrAsk++;
            }
            WriteJson(Path.Combine(dir, "identity.json"), new JsonWriter().BeginObj().Key("record_type").S("a25_interactive_identity")
                .Key("results").S(stamp).Key("golden_manifest_sha256").S(g.ManifestSha256).Key("prompt_asset_sha256").S(g.AssetSha256)
                .Key("model").S(modelPath).Key("context_tokens").I(Pipeline.ContextTokens)
                .Key("threads").S("requested and passed to the loader; not read back by the runtime").Key("flags").I(ReplayRunner.Flags)
                .Key("llama_cpp").S(runtime.Version).Key("load_ms").D(loadMs).Key("app_version").S(appVersion)
                .Key("event_log").S(EventLog.FilePath).Key("session").S(EventLog.SessionId), guard);
            WriteJson(Path.Combine(dir, "done.json"), new JsonWriter().BeginObj().Key("record_type").S("a25_interactive_done")
                .Key("results").S(stamp).Key("purpose").S("diagnostic").Key("requests").I(g.Rows.Count).Key("off").I(off.Count).Key("prefix").I(prefix.Count)
                .Key("off_completed").I(n(off, "completed")).Key("off_ask_model_selected").I(n(off, "model_selected"))
                .Key("off_ask_exact_tie").I(n(off, "exact_tie")).Key("off_refused").I(n(off, "refused")).Key("off_failed").I(n(off, "failed"))
                .Key("off_path_u").I(n(off, "path_U")).Key("prefix_path_u").I(n(prefix, "path_U")).Key("prefix_path_p").I(n(prefix, "path_P"))
                .Key("prefix_completed").I(n(prefix, "completed")).Key("prefix_ask_model_selected").I(n(prefix, "model_selected"))
                .Key("prefix_ask_exact_tie").I(n(prefix, "exact_tie")).Key("prefix_refused").I(n(prefix, "refused")).Key("prefix_failed").I(n(prefix, "failed"))
                .Key("completed_or_ask_both").I(completedOrAsk).Key("prompts_equal_goldens").I(promptsEqual)
                .Key("prefix_kept_some").I(kept).Key("same_choice").I(sameChoice).Key("same_offered_logits").I(sameOffered)
                .Key("total_ms").D(total.Elapsed.TotalMilliseconds)
                .Key("finished_utc").S(DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture)).End().ToString());
            Event("interactive.end", new JsonWriter().BeginObj().Key("results").S(stamp).Key("prompts_equal_goldens").I(promptsEqual)
                .Key("same_choice").I(sameChoice).End().ToString());
            return "Interactive check done: prompts " + promptsEqual + "/" + g.Rows.Count + " equal to the goldens; choices "
                   + sameChoice + "/" + g.Rows.Count + " equal with and without the cache; results " + stamp + ".";
        }
    }
}
