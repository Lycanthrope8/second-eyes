using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.Grounding.Replay
{
    /// <summary>
    /// A2.5 delivery 1, step 2 (D100, D103): replays the pushed frozen requests with llama.cpp on the headset.
    ///
    /// The bundle's headset part (replay-manifest.json, requests.jsonl, fixtures-written.jsonl) is read from
    /// files/replay/bundle and checked against its own hashes. The self-checks run first. Then the model loads in this
    /// runner's own runtime: 8,192 tokens of context requested, 2 threads, the accepted settings, with llama.cpp's
    /// startup lines captured from stderr around the load only.
    ///
    /// The written fixtures go through the ordered input checks and must stop before any evaluation. Each dataset
    /// request, in the bundle's order, passes the same checks (including the native tokenization of the complete prompt
    /// against the frozen IDs) and is then scored on three paths from slices of its frozen token list:
    /// - U: all tokens with keep 0;
    /// - R: right after U, keep n - 1 and evaluate the last token;
    /// - P: clear, the tokens before the command line with keep 0, then the rest with keep k.
    /// se_n_cached is checked after every evaluation, and the intermediate prefix is never scored.
    ///
    /// Everything goes into a new folder files/replay/results/(UTC time): identity.json, startup-log.txt,
    /// selfchecks.json, fixtures.jsonl, results.jsonl (one line per request, written as it completes) and done.json,
    /// written last. The D101 comparisons are made on the laptop, never here.
    /// </summary>
    public sealed class ReplayRunner : IDisposable
    {
        public const int RequestedContext = 8192;
        public const int Threads = 2;   // D59
        public const int Flags = 0;     // the accepted settings: flash attention automatic, repacking and mapping on

        public static string BundleDir { get { return Path.Combine(Application.persistentDataPath, "replay", "bundle"); } }
        public static string ResultsRoot { get { return Path.Combine(Application.persistentDataPath, "replay", "results"); } }
        public static bool BundlePresent { get { return File.Exists(Path.Combine(BundleDir, "replay-manifest.json")); } }

        private readonly SynchronizationContext main;
        private readonly Action<string> status;
        private LlamaRuntime runtime;

        public ReplayRunner(SynchronizationContext main, Action<string> status)
        {
            this.main = main;
            this.status = status;
        }

        private void Say(string text) { main.Post(_ => status(text), null); }
        private void Event(string ev, string json) { main.Post(_ => EventLog.Write(ev, json), null); }

        private sealed class Bundle
        {
            public ReplayManifest Manifest;
            public string ManifestSha256, RequestsSha256, FixturesSha256;
            public List<ReplayRecord> Requests = new List<ReplayRecord>();
            public List<ReplayRecord> Fixtures = new List<ReplayRecord>();
        }

        private static List<ReplayRecord> Lines(byte[] data)
        {
            var list = new List<ReplayRecord>();
            foreach (string line in new UTF8Encoding(false, true).GetString(data).Split('\n'))
                if (line.Trim().Length > 0) list.Add(JsonUtility.FromJson<ReplayRecord>(line));
            return list;
        }

        /// <summary>The bundle's headset part, checked against its own manifest; throws with the reason.</summary>
        private static Bundle ReadBundle(string dir)
        {
            var b = new Bundle();
            byte[] mb = File.ReadAllBytes(Path.Combine(dir, "replay-manifest.json"));
            b.ManifestSha256 = ReplayHash.Sha256Hex(mb);
            b.Manifest = JsonUtility.FromJson<ReplayManifest>(new UTF8Encoding(false, true).GetString(mb));
            ReplayManifest m = b.Manifest;
            if (m == null || m.record_type != "a25_replay_headset_manifest")
                throw new InvalidOperationException("replay-manifest.json is not a replay bundle's headset manifest");
            if (m.input_checks == null || string.Join(",", m.input_checks) != string.Join(",", ReplayChecks.Order))
                throw new InvalidOperationException("the bundle expects another order of input checks than this app");
            byte[] rb = File.ReadAllBytes(Path.Combine(dir, Path.GetFileName(m.requests_file)));
            byte[] fb = File.ReadAllBytes(Path.Combine(dir, Path.GetFileName(m.fixtures_file)));
            b.RequestsSha256 = ReplayHash.Sha256Hex(rb);
            b.FixturesSha256 = ReplayHash.Sha256Hex(fb);
            if (b.RequestsSha256 != m.requests_sha256 || b.FixturesSha256 != m.fixtures_sha256)
                throw new InvalidOperationException("a bundle file differs from its manifest's SHA-256: push the bundle again");
            b.Requests = Lines(rb);
            b.Fixtures = Lines(fb);
            if (b.Requests.Count != m.requests_count || b.Fixtures.Count != m.fixtures_count)
                throw new InvalidOperationException("the bundle's record counts differ from its manifest");
            return b;
        }

        private static int[] Tokenize(IntPtr s, byte[] text)
        {
            var z = new byte[text.Length + 1];   // se_tokenize reads up to the NUL
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

        private static int[] Slice(int[] t, int offset, int count)
        {
            var s = new int[count];
            Array.Copy(t, offset, s, 0, count);
            return s;
        }

        /// <summary>One evaluation: the slice, keep, return code, se_n_cached against the expected count, time.
        /// Returns whether it succeeded with the expected cache.</summary>
        private static bool Eval(JsonWriter w, IntPtr s, int[] t, int offset, int count, int keep, int expect)
        {
            var clock = Stopwatch.StartNew();
            int rc = LlamaNative.se_eval(s, Slice(t, offset, count), count, keep);
            double ms = clock.Elapsed.TotalMilliseconds;
            int cached = LlamaNative.se_n_cached(s);
            w.BeginObj().Key("offset").I(offset).Key("count").I(count).Key("keep").I(keep).Key("status").I(rc)
             .Key("error").S(rc != 0 ? LlamaNative.LastError() : null).Key("n_cached").I(cached).Key("expected_n_cached").I(expect)
             .Key("ms").D(ms).End();
            return rc == 0 && cached == expect;
        }

        /// <summary>The final position's offered scores, written raw: logits as float32, log-probabilities (reported
        /// only), shares over every offered code (K included) and the decision with exact ties to K.</summary>
        private static string Scored(JsonWriter w, IntPtr s, float[] row, ReplayRecord r, ReplayManifest m)
        {
            int got = LlamaNative.se_logits(s, row, row.Length);
            PathScore p = ReplayScoring.Score(row, got, r.codes, r.code_token_ids, m.ask_code, id => LlamaNative.se_logprob(s, id));
            w.Key("logits_returned").I(got).Key("error").S(p.Error).Key("non_finite").I(p.NonFinite)
             .Key("first_non_finite").I(p.FirstNonFinite).Key("choice_code").S(p.Choice).Key("selection_reason").S(p.Reason)
             .Key("tied_codes").Strs(p.Tied).Key("offered").BeginArr();
            if (p.Offered != null)
                for (int i = 0; i < r.codes.Length; i++)
                    w.BeginObj().Key("code").S(r.codes[i]).Key("target").S(r.targets[i]).Key("token_id").I(r.code_token_ids[i])
                     .Key("logit").F(p.Offered[i]).Key("log_prob").D(p.LogProbs[i]).Key("restricted_share").D(p.Shares[i]).End();
            w.End();
            return p.Error == null ? p.Choice : "error";
        }

        private static string RequestLine(IntPtr s, float[] row, ReplayRecord r, ReplayManifest m, string[] choices)
        {
            var w = new JsonWriter().BeginObj();
            var clock = Stopwatch.StartNew();
            w.Key("record_type").S("a25_replay_result").Key("request_id").S(r.request_id).Key("kind").S(r.kind);
            InputCheck c = ReplayChecks.Run(r, m, bytes => Tokenize(s, bytes));
            w.Key("input").BeginObj().Key("outcome").S(c.Outcome).Key("check").S(c.Check).Key("reason").S(c.Reason)
             .Key("detail").S(c.Detail).Key("runtime_tokens").I(c.RuntimeTokens).Key("first_token_difference").I(c.FirstTokenDifference)
             .Key("prompt_sha256").S(c.PromptSha256).Key("token_ids_sha256").S(c.TokenIdsSha256)
             .Key("mapping_sha256").S(c.MappingSha256).Key("ms").D(clock.Elapsed.TotalMilliseconds).End();
            int n = r.token_ids != null ? r.token_ids.Length : 0, k = r.keep_before_command_line;
            w.Key("boundaries").BeginObj().Key("n").I(n).Key("keep_before_last_token").I(r.keep_before_last_token)
             .Key("keep_before_command_line").I(k).End();
            w.Key("paths").BeginObj();
            if (c.Outcome == "passed" && (k <= 0 || k >= n - 1 || r.keep_before_last_token != n - 1))
            {
                w.Key("error").S("boundaries outside 0 < keep_before_command_line < keep_before_last_token = n - 1");
            }
            else if (c.Outcome == "passed")
            {
                int[] t = r.token_ids;
                // U: all tokens from an empty cache
                w.Key("U").BeginObj().Key("evals").BeginArr();
                bool ok = Eval(w, s, t, 0, n, 0, n);
                w.End();
                choices[0] = ok ? Scored(w, s, row, r, m) : "error";
                if (!ok) w.Key("error").S("eval_failed_or_cache_mismatch");
                w.End();
                // R: right after U, the last token re-evaluated over the cached n - 1
                w.Key("R").BeginObj();
                if (ok)
                {
                    w.Key("evals").BeginArr();
                    bool okR = Eval(w, s, t, n - 1, 1, n - 1, n);
                    w.End();
                    choices[1] = okR ? Scored(w, s, row, r, m) : "error";
                    if (!okR) w.Key("error").S("eval_failed_or_cache_mismatch");
                }
                else { choices[1] = "skipped"; w.Key("error").S("skipped: U did not complete"); }
                w.End();
                // P: split evaluation of the same request, the prefix built in its own pass
                w.Key("P").BeginObj().Key("evals").BeginArr();
                bool okP = Eval(w, s, t, 0, k, 0, k) && Eval(w, s, t, k, n - k, k, n);
                w.End();
                choices[2] = okP ? Scored(w, s, row, r, m) : "error";
                if (!okP) w.Key("error").S("eval_failed_or_cache_mismatch");
                w.End();
            }
            w.End().Key("ms").D(clock.Elapsed.TotalMilliseconds).End();
            return w.ToString();
        }

        private static string FixtureLine(IntPtr s, ReplayRecord r, ReplayManifest m, out bool asExpected)
        {
            InputCheck c = ReplayChecks.Run(r, m, bytes => Tokenize(s, bytes));
            asExpected = c.Outcome == r.expected_outcome && c.Reason == r.expected_reason;
            return new JsonWriter().BeginObj().Key("record_type").S("a25_replay_fixture").Key("request_id").S(r.request_id)
                .Key("kind").S(r.kind).Key("expected_outcome").S(r.expected_outcome).Key("expected_reason").S(r.expected_reason)
                .Key("outcome").S(c.Outcome).Key("check").S(c.Check).Key("reason").S(c.Reason).Key("detail").S(c.Detail)
                .Key("runtime_tokens").I(c.RuntimeTokens).Key("evaluated").B(false).Key("as_expected").B(asExpected).End().ToString();
        }

        private static void WriteJson(string path, string json) { File.WriteAllText(path, json + "\n", new UTF8Encoding(false)); }

        /// <summary>Runs the whole replay and returns a one-line summary for the panel. Call on the main thread.</summary>
        public async Task<string> RunAsync(string modelPath)
        {
            string stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            string bundleDir = BundleDir, dir = Path.Combine(ResultsRoot, stamp);
            string appVersion = Application.version, unity = Application.unityVersion, device = SystemInfo.deviceModel;
            if (Directory.Exists(dir))   // never written into: a second press within the same second gets refused
                return "Replay not started: results folder " + stamp + " already exists. Nothing was written; press Replay again.";
            Directory.CreateDirectory(dir);
            var total = Stopwatch.StartNew();
            Say("Replay: reading the bundle...");
            Bundle b = await Task.Run(() => ReadBundle(bundleDir));
            ReplayManifest m = b.Manifest;
            Event("replay.start", new JsonWriter().BeginObj().Key("results").S(stamp).Key("requests").I(b.Requests.Count)
                .Key("fixtures").I(b.Fixtures.Count).Key("requests_sha256").S(b.RequestsSha256).End().ToString());

            var checks = ReplaySelfChecks.Run();
            var sj = new JsonWriter().BeginObj().Key("record_type").S("a25_replay_selfchecks").Key("checks").BeginArr();
            bool allPassed = true;
            foreach (var sc in checks)
            {
                allPassed &= sc.Passed;
                sj.BeginObj().Key("name").S(sc.Name).Key("passed").B(sc.Passed).Key("detail").S(sc.Detail).End();
            }
            WriteJson(Path.Combine(dir, "selfchecks.json"), sj.End().Key("all_passed").B(allPassed).End().ToString());
            if (!allPassed) return "Replay stopped: a self-check failed (replay/results/" + stamp + "/selfchecks.json).";

            Say("Replay: loading the model (8,192 tokens of context, 2 threads)...");
            long rssBefore = LlamaRuntime.MemoryKb(false), peakBefore = LlamaRuntime.MemoryKb(true);
            string captureError = null;
            runtime = new LlamaRuntime(main);
            string capturePath = Path.Combine(dir, "startup-log.txt");
            double loadMs = await runtime.LoadCapturedAsync(modelPath, RequestedContext, Threads, Flags, capturePath,
                                                            e => captureError = captureError == null ? e : captureError + "; " + e);
            long rssAfter = LlamaRuntime.MemoryKb(false), peakAfter = LlamaRuntime.MemoryKb(true);
            int[] info = await runtime.WithModel(s => new[] { LlamaNative.se_n_ctx(s), LlamaNative.se_n_vocab(s),
                                                              LlamaNative.se_n_seq(s), LlamaNative.se_flags(s) });
            long captureBytes = File.Exists(capturePath) ? new FileInfo(capturePath).Length : -1;
            var id = new JsonWriter().BeginObj().Key("record_type").S("a25_replay_identity").Key("results").S(stamp)
                .Key("started_utc").S(DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture))
                .Key("bundle").BeginObj().Key("policy_id").S(m.policy_id).Key("manifest_sha256").S(b.ManifestSha256)
                .Key("requests_sha256").S(b.RequestsSha256).Key("requests").I(b.Requests.Count)
                .Key("fixtures_sha256").S(b.FixturesSha256).Key("fixtures").I(b.Fixtures.Count).End()
                .Key("model").BeginObj().Key("path").S(modelPath).Key("bytes").I(new FileInfo(modelPath).Length).End()
                .Key("requested").BeginObj().Key("n_ctx").I(RequestedContext).Key("threads").I(Threads).Key("n_seq").I(LlamaRuntime.Sequences)
                .Key("flags").I(Flags).End()
                .Key("runtime").BeginObj().Key("n_ctx").I(info[0]).Key("n_vocab").I(info[1]).Key("n_seq").I(info[2]).Key("flags").I(info[3])
                .Key("version").S(runtime.Version).Key("system_info").S(runtime.SystemInfo).Key("load_ms").D(loadMs).End()
                .Key("memory_kb").BeginObj().Key("before_load_rss").I(rssBefore).Key("before_load_peak").I(peakBefore)
                .Key("after_load_rss").I(rssAfter).Key("after_load_peak").I(peakAfter).End()
                .Key("capture").BeginObj().Key("scope").S("process-wide stderr (file descriptor 2), redirected around se_load_ex only")
                .Key("verbose").S("on during the load only").Key("file").S("startup-log.txt").Key("bytes").I(captureBytes)
                .Key("error").S(captureError).End()
                .Key("app").BeginObj().Key("version").S(appVersion).Key("unity").S(unity).Key("device_model").S(device)
                .Key("event_log").S(EventLog.FilePath).Key("session").S(EventLog.SessionId).End()
                .End().ToString();
            WriteJson(Path.Combine(dir, "identity.json"), id);
            Event("replay.load", new JsonWriter().BeginObj().Key("results").S(stamp).Key("ms").D(loadMs).Key("n_ctx_requested").I(RequestedContext)
                .Key("n_ctx").I(info[0]).Key("threads").I(Threads).Key("llama_cpp").S(runtime.Version)
                .Key("capture_bytes").I(captureBytes).Key("capture_error").S(captureError).End().ToString());
            if (info[1] != m.vocab_size)
                return "Replay stopped: the model's vocabulary (" + info[1] + ") is not the bundle's (" + m.vocab_size + ").";

            int fixturesOk = await runtime.WithModel(s =>
            {
                int ok = 0;
                using (var f = new StreamWriter(Path.Combine(dir, "fixtures.jsonl"), false, new UTF8Encoding(false)))
                    foreach (ReplayRecord r in b.Fixtures)
                    {
                        string line = FixtureLine(s, r, m, out bool asExpected);
                        if (asExpected) ok++;
                        f.Write(line + "\n");
                    }
                return ok;
            });

            int vocab = info[1];
            int completed = await runtime.WithModel(s =>
            {
                var row = new float[vocab];
                int done = 0;
                using (var f = new StreamWriter(Path.Combine(dir, "results.jsonl"), false, new UTF8Encoding(false)))
                    for (int i = 0; i < b.Requests.Count; i++)
                    {
                        ReplayRecord r = b.Requests[i];
                        Say($"Replay: {r.request_id} ({i + 1}/{b.Requests.Count}), U, R and P...");
                        var choices = new[] { "-", "-", "-" };
                        f.Write(RequestLine(s, row, r, m, choices) + "\n");
                        f.Flush();
                        done++;
                        Event("replay.request", new JsonWriter().BeginObj().Key("request_id").S(r.request_id).Key("U").S(choices[0])
                            .Key("R").S(choices[1]).Key("P").S(choices[2]).End().ToString());
                    }
                return done;
            });

            WriteJson(Path.Combine(dir, "done.json"), new JsonWriter().BeginObj().Key("record_type").S("a25_replay_done").Key("results").S(stamp)
                .Key("requests").I(b.Requests.Count).Key("written").I(completed).Key("fixtures").I(b.Fixtures.Count)
                .Key("fixtures_as_expected").I(fixturesOk).Key("self_checks_passed").B(allPassed).Key("total_ms").D(total.Elapsed.TotalMilliseconds)
                .Key("end_rss_kb").I(LlamaRuntime.MemoryKb(false)).Key("end_peak_kb").I(LlamaRuntime.MemoryKb(true))
                .Key("finished_utc").S(DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture)).End().ToString());
            Event("replay.end", new JsonWriter().BeginObj().Key("results").S(stamp).Key("written").I(completed)
                .Key("fixtures_as_expected").I(fixturesOk).Key("ms").D(total.Elapsed.TotalMilliseconds).End().ToString());
            return $"Replay done: {completed}/{b.Requests.Count} requests written, {fixturesOk}/{b.Fixtures.Count} fixtures as expected. "
                 + "Results: replay/results/" + stamp;
        }

        public void Dispose()
        {
            if (runtime != null) runtime.Dispose();
            runtime = null;
        }
    }
}
