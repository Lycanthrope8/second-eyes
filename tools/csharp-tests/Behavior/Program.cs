using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SecondEyes.Grounding.Interactive;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Replay;

/// Labelled test double: one token per 4 prompt bytes; a cache of tokens; a 100-value row derived from the cache.
sealed class FakeNative : INative
{
    public List<int> Cache = new List<int>();
    public List<(int keep, int count)> Calls = new List<(int, int)>();
    public bool Tie, NaN, ThrowTok, ThrowEval, ThrowLogits; public int FailNext, KTop = -1;
    public int[] Tokenize(byte[] b) { if (ThrowTok) throw new InvalidOperationException("tokenizer fault"); var t = new List<int>(); for (int i = 0; i < b.Length; i += 4) { int v = 0; for (int j = i; j < Math.Min(i + 4, b.Length); j++) v = v * 257 + b[j]; t.Add(v & 0x7fffffff); } return t.ToArray(); }
    public int Eval(int[] tokens, int keep) { if (ThrowEval) throw new InvalidOperationException("eval fault"); Calls.Add((keep, tokens.Length)); if (FailNext > 0) { FailNext--; return -3; } if (keep > Cache.Count) return -1; Cache = Cache.Take(keep).Concat(tokens).ToList(); return 0; }
    public int Cached() => Cache.Count;
    public int Logits(float[] row) { if (ThrowLogits) throw new InvalidOperationException("logits fault"); int h = 17; foreach (int x in Cache) h = unchecked(h * 31 + x); for (int i = 0; i < 100; i++) row[i] = (float)(((h ^ (i * 7919)) & 1023) / 100.0); if (Tie) { row[32] = 50; row[33] = 50; } if (NaN) row[5] = float.NaN; if (KTop >= 0) row[KTop] = 99; return 100; }
    public double LogProb(int t) => -1.0;
    public string LastError() => "fake error";
}

static class Program
{
    static int pass, fail;
    static void Check(string n, bool ok, string d = "") { if (ok) { pass++; Console.WriteLine("  ok    " + n); } else { fail++; Console.WriteLine("  FAIL  " + n + (d.Length > 0 ? "  [" + d + "]" : "")); } }

    static int Main(string[] args)
    {
        string h = args[0];
        var asset = PromptAsset.From(JParse.Parse(File.ReadAllText(Path.Combine(h, "prompt-asset.json"), Encoding.UTF8)));
        var goldens = File.ReadAllLines(Path.Combine(h, "goldens.jsonl"), Encoding.UTF8).Where(l => l.Trim().Length > 0).Select(JParse.Parse).ToList();
        InteractiveRequest Req(JNode g, string id) {
            string scene = File.ReadAllText(Path.Combine(h, g["scene_file"].Text), Encoding.UTF8);
            return new InteractiveRequest { RequestId = id, Source = "test", SnapshotId = g["snapshot_id"].Text,
                SnapshotSha256 = ReplayHash.Sha256Hex(Encoding.UTF8.GetBytes(scene)), SceneText = scene,
                CommandText = File.ReadAllText(Path.Combine(h, g["command_file"].Text), Encoding.UTF8) }; }
        var row = new float[100];
        // 1. every golden's prompt and mapping, through the pipeline
        int same = 0;
        foreach (var g in goldens) {
            var o = Pipeline.Run(Req(g, g["request_id"].Text), asset, new FakeNative(), new PrefixCache(CacheMode.Off), ExecutionPurpose.Diagnostic, row, null);
            if (o.PromptSha256 == g["prompt_sha256"].Text && o.MappingSha256 == g["mapping_sha256"].Text && (o.Status == "completed" || o.Status == "ask")) same++;
        }
        Check($"the pipeline builds every golden's prompt and mapping ({same} of {goldens.Count})", same == goldens.Count);
        var g1 = goldens[0]; var g2 = goldens.First(g => g["snapshot_id"].Text == g1["snapshot_id"].Text && g["request_id"].Text != g1["request_id"].Text);
        var gOther = goldens.First(g => g["snapshot_id"].Text != g1["snapshot_id"].Text);
        // 2. cache off
        var nat = new FakeNative(); var off = new PrefixCache(CacheMode.Off);
        var a = Pipeline.Run(Req(g1, "a"), asset, nat, off, ExecutionPurpose.Diagnostic, row, null); var b = Pipeline.Run(Req(g1, "b"), asset, nat, off, ExecutionPurpose.Diagnostic, row, null);
        Check("cache off: a repeat is evaluated in full, with the same outcome", a.KeptTokens == 0 && b.KeptTokens == 0 && b.EvaluatedTokens == b.Tokens
              && a.ChoiceCode == b.ChoiceCode && a.Logits.SequenceEqual(b.Logits));
        // 3. exact prefix
        nat = new FakeNative(); var pc = new PrefixCache(CacheMode.ExactPrefix);
        a = Pipeline.Run(Req(g1, "a"), asset, nat, pc, ExecutionPurpose.Diagnostic, row, null); b = Pipeline.Run(Req(g1, "b"), asset, nat, pc, ExecutionPurpose.Diagnostic, row, null);
        Check("exact prefix: the first request in full, its repeat keeps n - 1 and evaluates the final position",
              a.KeptTokens == 0 && b.KeptTokens == b.Tokens - 1 && b.EvaluatedTokens == 1 && b.Status == a.Status);
        var p1 = Encoding.UTF8.GetBytes(File.ReadAllText(Path.Combine(h, g1["scene_file"].Text)));
        var c = Pipeline.Run(Req(g2, "c"), asset, nat, pc, ExecutionPurpose.Diagnostic, row, null);
        byte[] pa = PromptBytes(asset, Req(g1, "x")), pb2 = PromptBytes(asset, Req(g2, "y"));
        int diff = Enumerable.Range(0, Math.Min(pa.Length, pb2.Length)).First(k => pa[k] != pb2[k]);
        Check($"another command or pose on the same snapshot keeps exactly the shared prefix ({c.KeptTokens} tokens, first difference at byte {diff})",
              c.KeptTokens == Math.Min(diff / 4, c.Tokens - 1) && c.KeptTokens > 0);
        var d = Pipeline.Run(Req(gOther, "d"), asset, nat, pc, ExecutionPurpose.Diagnostic, row, null);
        Check("another snapshot invalidates the cache: kept 0", d.KeptTokens == 0 && d.EvaluatedTokens == d.Tokens);
        // 4. refusals
        var bad = Req(g1, "e"); bad.SnapshotSha256 = new string('0', 64);
        var e = Pipeline.Run(bad, asset, new FakeNative(), pc, ExecutionPurpose.Diagnostic, row, null);
        Check("a scene that is not the registered snapshot is refused", e.Status == "refused" && e.Reason == "snapshot_hash_mismatch");
        var stale = Req(g1, "f"); var sc = JParse.Parse(stale.CommandText);
        int ri = sc.Fields.FindIndex(x => x.Key == "scene_revision");
        sc.Fields[ri] = new KeyValuePair<string, JNode>("scene_revision", new JNode { Kind = JKind.Number, Text = "999", IsInteger = true, Num = 999 });
        stale.CommandText = Canon.Enc(sc);
        var f = Pipeline.Run(stale, asset, new FakeNative(), pc, ExecutionPurpose.Diagnostic, row, null);
        Check("a command for another scene revision is refused as stale", f.Status == "refused" && f.Reason == "stale_scene", f.Reason + " " + f.Detail);
        var big = Req(g1, "g"); var cmd = JParse.Parse(big.CommandText);
        big.CommandText = big.CommandText.Replace(cmd["text"].Text.Length > 0 ? Canon.Enc(cmd["text"]) : "\"\"", Canon.Enc(JNode.Str(new string('x', 40000))));
        var nb = new FakeNative(); var gq = Pipeline.Run(big, asset, nb, new PrefixCache(CacheMode.Off), ExecutionPurpose.Diagnostic, row, null);
        Check("a prompt over the context ceiling is refused before any evaluation", gq.Status == "refused" && gq.Reason == "context_budget_exceeded" && nb.Calls.Count == 0, gq.Reason);
        // 5. failures, ASK, cancellation
        nat = new FakeNative { FailNext = 1 }; pc = new PrefixCache(CacheMode.ExactPrefix);
        Pipeline.Run(Req(g1, "h0"), asset, nat, pc, ExecutionPurpose.Diagnostic, row, null); nat.FailNext = 1;
        var h1 = Pipeline.Run(Req(g1, "h1"), asset, nat, pc, ExecutionPurpose.Diagnostic, row, null); var h2 = Pipeline.Run(Req(g1, "h2"), asset, nat, pc, ExecutionPurpose.Diagnostic, row, null);
        Check("a failed evaluation is a failed outcome and resets the cache: the next request keeps 0",
              h1.Status == "failed" && h1.Reason == "eval_failed" && h2.KeptTokens == 0 && h2.Status != "failed");
        var tie = Pipeline.Run(Req(g1, "i"), asset, new FakeNative { Tie = true }, new PrefixCache(CacheMode.Off), ExecutionPurpose.Diagnostic, row, null);
        Check("an exact tie at the top is ASK, with no target and both codes listed",
              tie.Status == "ask" && tie.TargetObjectId == null && tie.ChoiceCode == asset.AskCode && tie.TiedCodes.Length == 2);
        var nc = new FakeNative(); var can = Pipeline.Run(Req(g1, "j"), asset, nc, new PrefixCache(CacheMode.Off), ExecutionPurpose.Diagnostic, row, () => true);
        Check("a cancelled request stops before any evaluation", can.Status == "cancelled" && nc.Calls.Count == 0);
        var nan = Pipeline.Run(Req(g1, "k"), asset, new FakeNative { NaN = true }, new PrefixCache(CacheMode.Off), ExecutionPurpose.Diagnostic, row, null);
        Check("a non-finite row is a failed outcome, never a choice", nan.Status == "failed" && nan.Reason == "non_finite_output" && nan.ChoiceCode == null);
        // ChatGPT's review: prefix reuse only in diagnostic execution; the actual path on every outcome; ASK kept apart from failures
        var on = new FakeNative();
        var op = Pipeline.Run(Req(g1, "o1"), asset, on, new PrefixCache(CacheMode.ExactPrefix), ExecutionPurpose.Operational, row, null);
        Check("an operational request with prefix reuse is refused before any evaluation, with an explicit reason",
              op.Status == "refused" && op.Reason == "prefix_reuse_not_accepted" && on.Calls.Count == 0 && op.ExecutionPath == "none");
        var uo = new PrefixCache(CacheMode.Off);
        var u1 = Pipeline.Run(Req(g1, "u1"), asset, on, uo, ExecutionPurpose.Operational, row, null);
        var u2 = Pipeline.Run(Req(g2, "u2"), asset, on, uo, ExecutionPurpose.Operational, row, null);
        Check("an operational request clears the prior KV and evaluates its complete prompt: path U, prior cache recorded",
              u1.ExecutionPath == "U" && u2.ExecutionPath == "U" && u2.PriorCachedTokens == u1.Tokens && u2.KeptTokens == 0
              && on.Calls[1].keep == 0 && on.Calls[1].count == u2.Tokens && u2.Purpose == "operational");
        var dn = new FakeNative(); var dc = new PrefixCache(CacheMode.ExactPrefix);
        var d1 = Pipeline.Run(Req(g1, "d1"), asset, dn, dc, ExecutionPurpose.Diagnostic, row, null);
        var d2 = Pipeline.Run(Req(g1, "d2"), asset, dn, dc, ExecutionPurpose.Diagnostic, row, null);
        Check("diagnostic prefix reuse records the path that actually ran: U first, then P", d1.ExecutionPath == "U" && d2.ExecutionPath == "P" && d2.Purpose == "diagnostic");
        Check("an exact tie is ASK by the tie rule", tie.AskBasis == "exact_tie");
        var kt = Pipeline.Run(Req(g1, "kt"), asset, new FakeNative { KTop = asset.CodeTokenIds[asset.AskCode] }, new PrefixCache(CacheMode.Off),
                              ExecutionPurpose.Operational, row, null);
        Check("ASK as the top offered score is model-selected", kt.Status == "ask" && kt.AskBasis == "model_selected" && kt.TargetObjectId == null);
        Check("a technical failure keeps its reason and is never an ASK",
              h1.Status == "failed" && h1.AskBasis == null && h1.ChoiceCode == null && h1.ExecutionPath != "none"
              && nan.Status == "failed" && nan.AskBasis == null);
        var json = System.Text.Json.JsonDocument.Parse(a.ToJson()).RootElement;
        Check("an outcome serializes to JSON with its scores and times", json.GetProperty("offered").GetArrayLength() == a.Codes.Count
              && json.GetProperty("ms").GetProperty("total").GetDouble() > 0 && json.GetProperty("cache_mode").GetString() == "ExactPrefix" && json.GetProperty("execution_path").GetString() == "U"
              && json.GetProperty("purpose").GetString() == "diagnostic");
        FailureTests(goldens, h, asset);
        GuardTests(goldens, h, asset);
        IntakeTests(goldens, h);
        LifecycleTests();
        ShutdownTests();
        PresentationTests(goldens, h, asset);
        Console.WriteLine($"{pass} passed, {fail} failed");
        return fail == 0 ? 0 : 1;
    }

    static byte[] PromptBytes(PromptAsset a, InteractiveRequest r)
    {
        var scene = JParse.Parse(r.SceneText); var cmd = JParse.Parse(r.CommandText);
        var m = Choices.SerializedOrder(scene, a.ObjectCodes, a.AskCode, a.AskTarget);
        return Encoding.UTF8.GetBytes(Choices.Prompt(a.BeforeSystem, a.SystemMessage, a.Between, a.AfterUser, m, Coordinates.Document(a.HeaderLine, a.SemanticsLine, scene, cmd)));
    }

    static void IntakeTests(List<JNode> goldens, string h)
    {
        var g = goldens[0];
        string scene = File.ReadAllText(Path.Combine(h, g["scene_file"].Text), Encoding.UTF8);
        var sc = JParse.Parse(scene);
        var cmd = JParse.Parse(File.ReadAllText(Path.Combine(h, g["command_file"].Text), Encoding.UTF8));
        string sha = ReplayHash.Sha256Hex(Encoding.UTF8.GetBytes(scene));
        var bind = new SceneBinding { SnapshotId = g["snapshot_id"].Text, SceneId = sc["scene_id"].Text, SceneRevision = sc["scene_revision"].Text,
                                      SnapshotSha256 = sha, SceneText = scene, Epoch = 1 };
        string Req(string id, string sid = null, string rev = null, string text = null, string rt = "a25_interactive_request") {
            string c = Canon.Enc(cmd);
            if (text != null) { var cn = JParse.Parse(c); int i = cn.Fields.FindIndex(x => x.Key == "text"); cn.Fields[i] = new KeyValuePair<string, JNode>("text", JNode.Str(text)); c = Canon.Enc(cn); }
            string env = ("{'format_version':1,'record_type':'" + rt + "','request_id':'" + id + "','expected_scene':{'snapshot_id':'"
                          + (sid ?? bind.SnapshotId) + "','scene_id':'" + bind.SceneId + "','scene_revision':" + (rev ?? bind.SceneRevision)
                          + ",'snapshot_sha256':'" + sha + "'},'command':").Replace('\'', '"');
            return env + c + "}";
        }
        JNode r; string id;
        Check("intake: a well-formed request validates", RequestFile.Validate(Req("q-1"), out r, out id) == null && id == "q-1");
        var bad = new (string label, string text, string want)[] {
            ("not JSON", "{oops", "not valid JSON"), ("another record type", Req("q-2", rt: "x"), "record_type"),
            ("a request ID with a path", Req("../q"), "request_id"), ("an empty command text", Req("q-3", text: "   "), "non-empty"),
            ("a command for another revision", Req("q-4", rev: (long.Parse(bind.SceneRevision) + 1).ToString()), "differs") };
        foreach (var b in bad) {
            string why = RequestFile.Validate(b.text, out r, out id);
            Check("intake: refuses " + b.label, why != null && why.Contains(b.want), why ?? "accepted");
        }
        RequestFile.Validate(Req("q-5"), out r, out id);
        Check("intake: the current scene matches; another snapshot or revision names the field", bind.Mismatch(r["expected_scene"]) == null
              && bind.Mismatch(JParse.Parse(Req("q-6", sid: "other"))["expected_scene"]) == "snapshot_id");
        var s = new SessionCore();
        Check("intake: a request ID is processed once; its duplicate is not", s.Claim("q-7") && !s.Claim("q-7") && s.Processed == 1);
        var t1 = new Ticket { RequestId = "a", SceneEpoch = 1 }; var t2 = new Ticket { RequestId = "b", SceneEpoch = 1 }; var t3 = new Ticket { RequestId = "c", SceneEpoch = 1 };
        s.Admit(t1); s.Admit(t2); s.Admit(t3);
        var run = s.Next(); var cancelled = s.CancelAll();
        Check("intake: cancel answers the queued requests at once and flags the running one",
              run == t1 && cancelled.Count == 2 && cancelled[0] == t2 && t1.CancelRequested && s.Waiting == 0 && s.Next() == null);
        InteractiveOutcome Done(string st) => new InteractiveOutcome { RequestId = "x", Status = st, ChoiceCode = st == "completed" ? "A" : null,
                                                                       TargetObjectId = st == "completed" ? "obj" : null, Reason = st == "failed" ? "eval_failed" : "max_offered_logit" };
        var o1 = s.Finish(t1, Done("completed"), bind);
        Check("intake: a result finished after a cancel is cancelled, its target discarded, its earlier status kept in the detail",
              o1.Status == "cancelled" && o1.Reason == "cancelled_during_evaluation" && o1.TargetObjectId == null && o1.Detail.Contains("completed, choice A"));
        var t4 = new Ticket { RequestId = "d", SceneEpoch = 1 }; s.Admit(t4); s.Next();
        var b2 = new SceneBinding { SnapshotId = bind.SnapshotId, Epoch = 2 };
        var o4 = s.Finish(t4, Done("completed"), b2);
        Check("intake: a result for a scene switched meanwhile is refused as stale, its target discarded",
              o4.Status == "refused" && o4.Reason == "stale_result_scene_changed" && o4.TargetObjectId == null);
        var t5 = new Ticket { RequestId = "e", SceneEpoch = 1 }; s.Admit(t5); s.Next(); s.CancelAll();
        var o5 = s.Finish(t5, Done("failed"), b2);
        Check("intake: a technical failure keeps its own reason through cancel and a scene switch", o5.Status == "failed" && o5.Reason == "eval_failed");
        var t6 = new Ticket { RequestId = "f", SceneEpoch = 1 }; s.Admit(t6); s.Next();
        var o6 = s.Finish(t6, Done("completed"), bind);
        Check("intake: an ordinary result passes unchanged", o6.Status == "completed" && o6.TargetObjectId == "obj");
        var st = new StageTimes { Intake = "adb_inbox", Detected = 100, Validated = 101, Queued = 102, InferenceStart = 110, InferenceEnd = 17110, Outcome = 17111 };
        var sj = System.Text.Json.JsonDocument.Parse(st.ToJson()).RootElement;
        Check("intake: the stage times give app-observed latency from detection to outcome, on one clock",
              st.AppObserved == 17011 && sj.GetProperty("app_observed_ms").GetDouble() == 17011 && sj.GetProperty("intake").GetString() == "adb_inbox");
        var w1 = new Ticket { RequestId = "w1", Source = "adb_inbox", SceneEpoch = 1 };
        var early = SessionCore.StaleAtStart(w1, new SceneBinding { Epoch = 2 });
        Check("intake: a request whose scene changed while it waited is refused before evaluation",
              early != null && early.Status == "refused" && early.Reason == "scene_changed_before_evaluation" && early.ExecutionPath == "none");
        Check("intake: a request still bound to the current scene goes on to evaluation", SessionCore.StaleAtStart(w1, new SceneBinding { Epoch = 1 }) == null);
        var an = SessionCore.Answer("q-9", "adb_inbox", "refused", "duplicate_request_id", "");
        Check("intake: an answer without evaluation is operational, uncached, with no path", an.Purpose == "operational" && an.ExecutionPath == "none" && an.CacheMode == "Off");
    }

    static void FailureTests(List<JNode> goldens, string h, PromptAsset asset)
    {
        InteractiveRequest Req(JNode g, string id) {
            string scene = File.ReadAllText(Path.Combine(h, g["scene_file"].Text), Encoding.UTF8);
            return new InteractiveRequest { RequestId = id, Source = "test", SnapshotId = g["snapshot_id"].Text,
                SnapshotSha256 = ReplayHash.Sha256Hex(Encoding.UTF8.GetBytes(scene)), SceneText = scene,
                CommandText = File.ReadAllText(Path.Combine(h, g["command_file"].Text), Encoding.UTF8) }; }
        var row = new float[100]; var g1 = goldens[0];
        var D = ExecutionPurpose.Diagnostic;
        var nat = new FakeNative { ThrowTok = true }; var pc = new PrefixCache(CacheMode.ExactPrefix);
        var a = Pipeline.Run(Req(g1, "f1"), asset, nat, pc, D, row, null);
        nat.ThrowTok = false;
        var b = Pipeline.Run(Req(g1, "f2"), asset, nat, pc, D, row, null);
        Check("failure: a tokenizer exception is an explicit failed outcome, nothing evaluated; the next request recovers",
              a.Status == "failed" && a.Reason == "tokenization_exception" && a.Detail.Contains("tokenizer fault") && nat.Calls.Count == 1 && b.Status == "completed");
        nat.ThrowEval = true;
        var c = Pipeline.Run(Req(g1, "f3"), asset, nat, pc, D, row, null);
        nat.ThrowEval = false;
        var d = Pipeline.Run(Req(g1, "f4"), asset, nat, pc, D, row, null);
        Check("failure: an evaluation exception fails explicitly and resets the cache: the next request keeps 0",
              c.Status == "failed" && c.Reason == "eval_exception" && d.Status == "completed" && d.KeptTokens == 0 && d.ExecutionPath == "U");
        nat.ThrowLogits = true;
        var e = Pipeline.Run(Req(g1, "f5"), asset, nat, pc, D, row, null);
        nat.ThrowLogits = false;
        var f = Pipeline.Run(Req(g1, "f6"), asset, nat, pc, D, row, null);
        Check("failure: a scoring exception fails explicitly and resets the cache: the next request keeps 0",
              e.Status == "failed" && e.Reason == "scoring_exception" && f.KeptTokens == 0 && f.Status == "completed");
        var ok1 = Pipeline.Run(Req(g1, "f7"), asset, nat, pc, D, row, null);
        nat.NaN = true;
        var bad = Pipeline.Run(Req(g1, "f8"), asset, nat, pc, D, row, null);
        nat.NaN = false;
        var after = Pipeline.Run(Req(g1, "f9"), asset, nat, pc, D, row, null);
        Check("failure: an invalid score is never committed: after a non-finite row the next request keeps 0, not n - 1",
              ok1.KeptTokens == ok1.Tokens - 1 && bad.Status == "failed" && bad.Reason == "non_finite_output" && after.KeptTokens == 0);
        Check("failure: no failure is ever an ASK", new[] { a, c, e, bad }.All(o => o.Status == "failed" && o.AskBasis == null && o.ChoiceCode == null && o.TargetObjectId == null));
    }

    static void GuardTests(List<JNode> goldens, string h, PromptAsset asset)
    {
        var texts = new Dictionary<string, string>();
        foreach (var g in goldens)
            foreach (string k in new[] { "scene_file", "command_file" })
                texts[g[k].Text] = File.ReadAllText(Path.Combine(h, g[k].Text), Encoding.UTF8);
        var byPrompt = goldens.ToDictionary(g => g["prompt_sha256"].Text, g => g["token_ids"].Items.Select(x => (int)x.Num).ToArray());
        Func<byte[], int[]> good = bytes => byPrompt[ReplayHash.Sha256Hex(bytes)];
        var r = StartupGuard.Run(asset, goldens, texts, good);
        Check($"guard: the real goldens pass: {r.SelfChecksPassed}/{r.SelfChecks} self-checks, {r.GoldensEqual}/{r.Goldens} goldens",
              r.Ok && r.SelfChecksPassed == 5 && r.GoldensEqual == 32 && r.FirstFailure == null);
        string victim = goldens[7]["prompt_sha256"].Text;
        var r2 = StartupGuard.Run(asset, goldens, texts, bytes => { var t = (int[])good(bytes).Clone(); if (ReplayHash.Sha256Hex(bytes) == victim) t[5]++; return t; });
        Check("guard: one wrong native token refuses processing, naming the golden", !r2.Ok && r2.GoldensEqual == 31 && r2.FirstFailure.Contains(goldens[7]["request_id"].Text));
        var r3 = StartupGuard.Run(asset, goldens, texts, bytes => throw new InvalidOperationException("tokenizer down"));
        Check("guard: a throwing tokenizer refuses processing with its reason", !r3.Ok && r3.GoldensEqual == 0 && r3.FirstFailure.Contains("tokenizer down"));
        var js = System.Text.Json.JsonDocument.Parse(r.WriteTo(new JsonWriter().BeginObj()).End().ToString()).RootElement.GetProperty("startup_check");
        Check("guard: its record says ok, with its counts", js.GetProperty("ok").GetBoolean() && js.GetProperty("goldens_equal").GetInt32() == 32);
    }

    static void LifecycleTests()
    {
        var core = new SessionCore();
        var t1 = new Ticket { RequestId = "t1", Session = "s", SceneEpoch = 1 };
        var t2 = new Ticket { RequestId = "t2", Session = "s", SceneEpoch = 1 };
        var t3 = new Ticket { RequestId = "t3", Session = "s", SceneEpoch = 1 };
        core.Admit(t1);
        var running = core.Next();                     // the worker takes t1
        bool admitted2 = core.Admit(t2);               // an inbox handler admits t2 while t1 runs
        var gone = core.Close();                       // shutdown begins: admission closes
        bool admitted3 = core.Admit(t3);               // a handler that was already running tries to admit t3
        Check("lifecycle: after admission closes, a late handler cannot enqueue (it answers its own request)", admitted2 && !admitted3 && core.Waiting == 0);
        Check("lifecycle: closing returns the queued tickets, flagged, and flags the running one", gone.Count == 1 && gone[0] == t2 && t2.CancelRequested && t1.CancelRequested);
        var bind = new SceneBinding { Epoch = 1 };
        var o = core.Finish(running, new InteractiveOutcome { RequestId = "t1", Status = "completed", ChoiceCode = "A", TargetObjectId = "x" }, bind);
        Check("lifecycle: a finished but unpublished ticket still counts as running: shutdown must wait for its publication",
              core.Running == t1 && !core.Drained && o.Status == "cancelled");
        core.Release(t1);                              // the worker has published t1
        Check("lifecycle: after publication and release, the session is drained", core.Running == null && core.Drained);
        var c2 = new SessionCore();
        Check("lifecycle: an open session is never drained, even when idle", !c2.Drained && !c2.Closed);
    }

    static void PresentationTests(List<JNode> goldens, string h, PromptAsset asset)
    {
        var tk = new Ticket { RequestId = "p1", Session = "s1", SceneEpoch = 3 };
        Check("presentation: the current session and epoch are presented", Presentation.Decide("s1", 3, tk) == Presentability.Present);
        Check("presentation: a result for an earlier epoch is stale, even if the worker passed it", Presentation.Decide("s1", 4, tk) == Presentability.StaleScene);
        Check("presentation: a result from another session is never presented", Presentation.Decide("s2", 3, tk) == Presentability.OtherSession
              && Presentation.Decide("s1", 3, null) == Presentability.OtherSession);
        var gA = goldens.First(g => g["snapshot_id"].Text == goldens[0]["snapshot_id"].Text);
        var gB = goldens.First(g => g["snapshot_id"].Text != goldens[0]["snapshot_id"].Text);
        JNode sceneA = JParse.Parse(File.ReadAllText(Path.Combine(h, gA["scene_file"].Text), Encoding.UTF8));
        JNode sceneB = JParse.Parse(File.ReadAllText(Path.Combine(h, gB["scene_file"].Text), Encoding.UTF8));
        var mapB = Choices.SerializedOrder(sceneB, asset.ObjectCodes, asset.AskCode, asset.AskTarget);
        string targetB = mapB[0].Value;
        var oldOutcome = new InteractiveOutcome { Status = "completed", TargetObjectId = targetB, ChoiceCode = mapB[0].Key };
        oldOutcome.Codes.AddRange(mapB.Select(m => m.Key)); oldOutcome.Targets.AddRange(mapB.Select(m => m.Value));
        var linesA = Presentation.SceneLines(sceneA, asset, oldOutcome, true, null);
        var mapA = Choices.SerializedOrder(sceneA, asset.ObjectCodes, asset.AskCode, asset.AskTarget);
        Check("presentation: the scene view uses the current scene's own mapping, never an old outcome's",
              linesA.Count == mapA.Count && linesA.Zip(mapA, (l, m) => l.Contains(m.Key + "  " + (m.Key == asset.AskCode ? "ASK" : m.Value))).All(x => x)
              && !linesA.Any(l => l.StartsWith("\u25B6")));
        string targetA = mapA[0].Value;
        var cur = new InteractiveOutcome { Status = "completed", TargetObjectId = targetA, ChoiceCode = mapA[0].Key };
        Check("presentation: the target is marked only when presentable", Presentation.SceneLines(sceneA, asset, cur, true, null).Count(l => l.StartsWith("\u25B6")) == 1
              && Presentation.SceneLines(sceneA, asset, cur, false, null).Count(l => l.StartsWith("\u25B6")) == 0);
        var askO = new InteractiveOutcome { Status = "ask", ChoiceCode = asset.AskCode };
        var askLines = Presentation.SceneLines(sceneA, asset, askO, true, null);
        Check("presentation: an ASK marks K, last", askLines.Last().StartsWith("\u25B6") && askLines.Count(l => l.StartsWith("\u25B6")) == 1);
        // the pipeline's top-level boundary: an asset missing a code's token ID
        var broken = PromptAsset.From(JParse.Parse(File.ReadAllText(Path.Combine(h, "prompt-asset.json"), Encoding.UTF8)));
        broken.CodeTokenIds.Remove(broken.ObjectCodes[0]);
        string scene = File.ReadAllText(Path.Combine(h, gA["scene_file"].Text), Encoding.UTF8);
        var req = new InteractiveRequest { RequestId = "pe", Source = "test", SnapshotId = gA["snapshot_id"].Text,
                                          SnapshotSha256 = ReplayHash.Sha256Hex(Encoding.UTF8.GetBytes(scene)), SceneText = scene,
                                          CommandText = File.ReadAllText(Path.Combine(h, gA["command_file"].Text), Encoding.UTF8) };
        var pc = new PrefixCache(CacheMode.ExactPrefix); var nat = new FakeNative(); var row = new float[100];
        Pipeline.Run(req, asset, nat, pc, ExecutionPurpose.Diagnostic, row, null);
        InteractiveOutcome pe = null; Exception escaped = null;
        try { pe = Pipeline.Run(req, broken, nat, pc, ExecutionPurpose.Diagnostic, row, null); } catch (Exception e) { escaped = e; }
        var next = Pipeline.Run(req, asset, nat, pc, ExecutionPurpose.Diagnostic, row, null);
        Check("pipeline: an unanticipated exception is a failed outcome (pipeline_exception), never an escape, and resets the cache",
              escaped == null && pe.Status == "failed" && pe.Reason == "pipeline_exception" && pe.AskBasis == null && next.KeptTokens == 0);
    }

    static void ShutdownTests()
    {
        // 1. a pending worker: two reported timeouts, and still no completion (so the caller cannot dispose)
        var core = new SessionCore();
        var w1 = new Ticket { RequestId = "w1", Session = "s", SceneEpoch = 1 }; core.Admit(w1); core.Next();
        var w2 = new Ticket { RequestId = "w2", Session = "s", SceneEpoch = 1 }; core.Admit(w2);
        var answered = new List<string>(); var progress = new List<string>();
        var inbox = new TaskCompletionSource<bool>(); var worker = new TaskCompletionSource<bool>(); var quiet = new TaskCompletionSource<bool>();
        int delays = 0;
        Func<Task> delay = () => { delays++; return delays <= 2 ? Task.CompletedTask : quiet.Task; };
        var run = SessionShutdown.RunAsync(core, tk => answered.Add(tk.RequestId), () => { }, null, worker.Task, delay, m => progress.Add(m));
        Check("shutdown: with the worker still running, timeouts are only reported; shutdown has not completed",
              !run.IsCompleted && progress.Count == 2 && progress.All(m => m.Contains("worker")) && answered.SequenceEqual(new[] { "w2" }));
        core.Release(w1); worker.SetResult(true);
        var r = run.GetAwaiter().GetResult();
        Check("shutdown: it completes only after the worker has published and released; clean, one queued request answered",
              r.Clean && r.CancelledQueued == 1 && r.ProgressReports == 2 && r.Faults.Count == 0);
        // 2. a late inbox handler: its refusal is published before shutdown completes
        var c2 = new SessionCore(); var order = new List<string>();
        var inbox2 = new TaskCompletionSource<bool>(); var worker2 = new TaskCompletionSource<bool>(); worker2.SetResult(true);
        var run2 = SessionShutdown.RunAsync(c2, tk => order.Add("queued:" + tk.RequestId), () => order.Add("inbox stopped"), inbox2.Task, worker2.Task,
                                            () => new TaskCompletionSource<bool>().Task, null);
        bool admittedLate = c2.Admit(new Ticket { RequestId = "late" });
        if (!admittedLate) order.Add("late refused and published");
        bool doneBefore = run2.IsCompleted;
        inbox2.SetResult(true);
        var r2 = run2.GetAwaiter().GetResult();
        Check("shutdown: a handler already running cannot join the queue; its refusal is published before shutdown completes",
              !admittedLate && !doneBefore && order.SequenceEqual(new[] { "inbox stopped", "late refused and published" }) && r2.Clean);
        // 3. a faulted worker task is not a successful shutdown
        var c3 = new SessionCore(); var w3 = new TaskCompletionSource<bool>(); w3.SetException(new InvalidOperationException("boom"));
        var r3 = SessionShutdown.RunAsync(c3, tk => { }, () => { }, Task.CompletedTask, w3.Task, () => new TaskCompletionSource<bool>().Task, null).GetAwaiter().GetResult();
        Check("shutdown: a faulted worker makes the report not clean, naming the fault", !r3.Clean && r3.Faults.Any(f => f.Contains("worker faulted") && f.Contains("boom")));
        // 4. a cancelled inbox task is not a successful shutdown either
        var c4 = new SessionCore(); var i4 = new TaskCompletionSource<bool>(); i4.SetCanceled();
        var r4 = SessionShutdown.RunAsync(c4, tk => { }, () => { }, i4.Task, Task.CompletedTask, () => new TaskCompletionSource<bool>().Task, null).GetAwaiter().GetResult();
        Check("shutdown: a cancelled inbox task makes the report not clean", !r4.Clean && r4.Faults.Any(f => f.Contains("inbox was cancelled")));
        // 5. a finished but unreleased request: the worker task completed, yet the request is still marked running
        var c5 = new SessionCore(); var t5 = new Ticket { RequestId = "t5" }; c5.Admit(t5); c5.Next();
        var r5 = SessionShutdown.RunAsync(c5, tk => { }, () => { }, Task.CompletedTask, Task.CompletedTask, () => new TaskCompletionSource<bool>().Task, null).GetAwaiter().GetResult();
        Check("shutdown: a request left unreleased (finished, unpublished) makes the report not clean", !r5.Clean && r5.Faults.Any(f => f.Contains("unreleased")));
        // 6. answering the queue fails: recorded, and shutdown still waits for the worker before completing
        var c6 = new SessionCore(); var q6 = new Ticket { RequestId = "q6" }; c6.Admit(q6);
        var w6 = new TaskCompletionSource<bool>();
        var run6 = SessionShutdown.RunAsync(c6, tk => throw new System.IO.IOException("disk"), () => { }, Task.CompletedTask, w6.Task,
                                            () => new TaskCompletionSource<bool>().Task, null);
        bool early6 = run6.IsCompleted;
        w6.SetResult(true);
        var r6 = run6.GetAwaiter().GetResult();
        Check("shutdown: a failure answering the queue is a recorded fault, and the worker is still awaited before completion",
              !early6 && !r6.Clean && r6.Faults.Any(f => f.Contains("answering q6 failed") && f.Contains("disk")));
    }
}
