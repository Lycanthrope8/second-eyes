using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Replay;

namespace SecondEyes.Grounding.Interactive
{
    // A2.5 delivery 3 (D98, D99(6), D104): the interactive pipeline, free of Unity so it is tested on a PC.
    // A request (a versioned snapshot and a typed command) becomes a structured outcome:
    //   1. the prompt, built by delivery 2's verified core;
    //   2. tokenization;
    //   3. a cache plan;
    //   4. evaluation;
    //   5. offered-letter scoring.
    // Off is the default: every request clears the prior KV state and evaluates its complete prompt (the replay's U path).
    // ExactPrefix keeps the longest common token prefix of the same snapshot (the replay's P path). It is confined to
    // diagnostic execution: an operational request with it is refused. Enabling it operationally needs cached numerical
    // acceptance under D56/D101, or an explicit amendment by the project lead. Each outcome records the path that ran.

    /// <summary>The native calls the pipeline needs (LlamaNative on the headset, a fake in tests).</summary>
    public interface INative
    {
        int[] Tokenize(byte[] utf8);
        int Eval(int[] tokens, int keep);    // keep `keep` cached positions, evaluate tokens after them; 0 on success
        int Cached();                         // positions in the cache now
        int Logits(float[] row);              // the final position's row; its length, 0 when there is none
        double LogProb(int token);
        string LastError();
    }

    public enum CacheMode { Off, ExactPrefix }

    /// <summary>Operational requests (the presets and the ADB inbox) may use only Off. Diagnostic execution (the
    /// Interactive check) may also use ExactPrefix.</summary>
    public enum ExecutionPurpose { Operational, Diagnostic }

    /// <summary>What every intake (panel, presets, the ADB inbox) hands the pipeline, so they cannot behave differently.</summary>
    public sealed class InteractiveRequest
    {
        public string RequestId, Source;
        public string SnapshotId, SnapshotSha256;   // the snapshot as registered: its ID and the SHA-256 of its file
        public string SceneText, CommandText;       // the scene and command records as JSON text
    }

    public sealed class InteractiveOutcome
    {
        public string RequestId, Source, Status = "failed", Reason, Detail = "", CacheMode, Purpose;
        public string ExecutionPath = "none";   // "U": prior KV cleared, complete prompt; "P": a cached prefix kept; "none": no evaluation
        public string AskBasis;                 // for ASK only: "model_selected" (K had the top offered score) or "exact_tie" (the tie rule)
        public int PriorCachedTokens = -1;
        public string ChoiceCode, TargetObjectId, PromptSha256, MappingSha256;
        public string[] TiedCodes = new string[0];
        public int Tokens = -1, KeptTokens = 0, EvaluatedTokens = 0;
        public double Margin = double.NaN, BuildMs, TokenizeMs, EvalMs, ScoreMs, TotalMs;
        public List<string> Codes = new List<string>(), Targets = new List<string>();
        public List<float> Logits = new List<float>();
        public List<double> Shares = new List<double>(), LogProbs = new List<double>();

        public string ToJson()
        {
            var w = new JsonWriter().BeginObj().Key("record_type").S("a25_interactive_outcome").Key("request_id").S(RequestId)
                .Key("source").S(Source).Key("status").S(Status).Key("reason").S(Reason).Key("detail").S(Detail)
                .Key("cache_mode").S(CacheMode).Key("purpose").S(Purpose).Key("execution_path").S(ExecutionPath)
                .Key("prior_cached_tokens").I(PriorCachedTokens).Key("ask_basis").S(AskBasis)
                .Key("choice_code").S(ChoiceCode).Key("target_object_id").S(TargetObjectId)
                .Key("tied_codes").Strs(TiedCodes).Key("margin").D(Margin).Key("prompt_sha256").S(PromptSha256)
                .Key("mapping_sha256").S(MappingSha256).Key("tokens").I(Tokens).Key("kept_tokens").I(KeptTokens)
                .Key("evaluated_tokens").I(EvaluatedTokens).Key("offered").BeginArr();
            for (int k = 0; k < Logits.Count; k++)
                w.BeginObj().Key("code").S(Codes[k]).Key("target").S(Targets[k]).Key("logit").F(Logits[k])
                 .Key("log_prob").D(LogProbs[k]).Key("restricted_share").D(Shares[k]).End();
            return w.End().Key("ms").BeginObj().Key("build").D(BuildMs).Key("tokenize").D(TokenizeMs).Key("eval").D(EvalMs)
                .Key("score").D(ScoreMs).Key("total").D(TotalMs).End().End().ToString();
        }
    }

    /// <summary>Bookkeeping of the one cached sequence: the tokens in the cache and the snapshot they belong to.</summary>
    public sealed class PrefixCache
    {
        public CacheMode Mode;
        private int[] tokens = new int[0];
        private string key;

        public PrefixCache(CacheMode mode) { Mode = mode; }

        /// <summary>How many cached positions to keep: 0 when the cache is off or holds another snapshot (an inventory
        /// or scene change invalidates it), otherwise the longest common prefix, at most n - 1 so that the final
        /// position is always evaluated. A pose or command change thus keeps only the prefix before it.</summary>
        public int Plan(int[] next, string snapshotKey, int nativeCached)
        {
            if (Mode == CacheMode.Off || key == null || key != snapshotKey) return 0;
            int lcp = 0, max = Math.Min(Math.Min(tokens.Length, next.Length), nativeCached);
            while (lcp < max && tokens[lcp] == next[lcp]) lcp++;
            return Math.Min(lcp, next.Length - 1);
        }

        public void Commit(int[] next, string snapshotKey) { tokens = (int[])next.Clone(); key = snapshotKey; }
        public void Reset() { tokens = new int[0]; key = null; }
        public int Length { get { return tokens.Length; } }
    }

    public static class Pipeline
    {
        public const int ContextTokens = 8192, Continuation = 1;

        private static InteractiveOutcome Stop(InteractiveOutcome o, string status, string reason, string detail, Stopwatch total)
        {
            o.Status = status;
            o.Reason = reason;
            o.Detail = detail ?? "";
            o.TotalMs = total.Elapsed.TotalMilliseconds;
            return o;
        }

        /// <summary>One request, start to finish. Never throws: every way out is a structured outcome.</summary>
        public static InteractiveOutcome Run(InteractiveRequest r, PromptAsset asset, INative native, PrefixCache cache,
                                             ExecutionPurpose purpose, float[] row, Func<bool> cancelled)
        {
            var o = new InteractiveOutcome { RequestId = r.RequestId, Source = r.Source, CacheMode = cache.Mode.ToString(),
                                             Purpose = purpose == ExecutionPurpose.Diagnostic ? "diagnostic" : "operational" };
            var total = Stopwatch.StartNew();
            if (cache.Mode != CacheMode.Off && purpose != ExecutionPurpose.Diagnostic)
                return Stop(o, "refused", "prefix_reuse_not_accepted", "prefix reuse is confined to diagnostic execution until cached "
                            + "numerical acceptance (D56/D101) or an explicit project-lead amendment", total);
            var clock = Stopwatch.StartNew();
            JNode scene, command;
            List<KeyValuePair<string, string>> mapping;
            byte[] prompt;
            try
            {
                if (ReplayHash.Sha256Hex(Encoding.UTF8.GetBytes(r.SceneText ?? "")) != r.SnapshotSha256)
                    return Stop(o, "refused", "snapshot_hash_mismatch", "the scene text is not the registered snapshot", total);
                scene = JParse.Parse(r.SceneText);
                command = JParse.Parse(r.CommandText);
                if (command["scene_id"].Text != scene["scene_id"].Text || command["scene_revision"].Text != scene["scene_revision"].Text)
                    return Stop(o, "refused", "stale_scene", "the command names another scene or revision than the snapshot", total);
                string doc = Coordinates.Document(asset.HeaderLine, asset.SemanticsLine, scene, command);
                mapping = Choices.SerializedOrder(scene, asset.ObjectCodes, asset.AskCode, asset.AskTarget);
                prompt = Encoding.UTF8.GetBytes(Choices.Prompt(asset.BeforeSystem, asset.SystemMessage, asset.Between, asset.AfterUser,
                                                               mapping, doc));
            }
            catch (Exception e) when (e is FormatException || e is KeyNotFoundException || e is ArgumentException
                                      || e is NullReferenceException || e is InvalidCastException)
            {
                return Stop(o, "refused", "invalid_input", e.GetType().Name + ": " + e.Message, total);
            }
            var codes = new string[mapping.Count];
            var targets = new string[mapping.Count];
            var ids = new int[mapping.Count];
            for (int k = 0; k < mapping.Count; k++)
            {
                codes[k] = mapping[k].Key;
                targets[k] = mapping[k].Value;
                ids[k] = asset.CodeTokenIds[codes[k]];
            }
            o.PromptSha256 = ReplayHash.Sha256Hex(prompt);
            o.MappingSha256 = ReplayHash.Mapping(codes, targets, ids);
            o.BuildMs = clock.Elapsed.TotalMilliseconds;
            if (cancelled != null && cancelled()) return Stop(o, "cancelled", "cancelled_before_evaluation", "", total);
            clock.Restart();
            int[] t = native.Tokenize(prompt);
            o.Tokens = t.Length;
            o.TokenizeMs = clock.Elapsed.TotalMilliseconds;
            if (t.Length == 0) return Stop(o, "failed", "tokenization_failed", native.LastError(), total);
            if (t.Length + Continuation > ContextTokens)
                return Stop(o, "refused", "context_budget_exceeded", t.Length + " tokens + " + Continuation + " > " + ContextTokens, total);
            string key = r.SnapshotId + "|" + r.SnapshotSha256;
            o.PriorCachedTokens = native.Cached();
            int keep = cache.Plan(t, key, o.PriorCachedTokens);
            var rest = new int[t.Length - keep];
            Array.Copy(t, keep, rest, 0, rest.Length);
            if (cancelled != null && cancelled()) return Stop(o, "cancelled", "cancelled_before_evaluation", "", total);
            o.ExecutionPath = keep == 0 ? "U" : "P";
            clock.Restart();
            int rc = native.Eval(rest, keep);
            o.EvalMs = clock.Elapsed.TotalMilliseconds;
            o.KeptTokens = keep;
            o.EvaluatedTokens = rest.Length;
            if (rc != 0)
            {
                cache.Reset();
                return Stop(o, "failed", "eval_failed", "status " + rc + ": " + native.LastError(), total);
            }
            if (native.Cached() != t.Length)
            {
                cache.Reset();
                return Stop(o, "failed", "cache_mismatch", native.Cached() + " cached positions, expected " + t.Length, total);
            }
            cache.Commit(t, key);
            clock.Restart();
            int n = native.Logits(row);
            PathScore s = ReplayScoring.Score(row, n, codes, ids, asset.AskCode, native.LogProb);
            o.ScoreMs = clock.Elapsed.TotalMilliseconds;
            if (s.Error != null) return Stop(o, "failed", s.Error, s.NonFinite > 0 ? s.NonFinite + " non-finite values" : "", total);
            o.Codes.AddRange(codes);
            o.Targets.AddRange(targets);
            for (int k = 0; k < codes.Length; k++)
            {
                o.Logits.Add(s.Offered[k]);
                o.Shares.Add(s.Shares[k]);
                o.LogProbs.Add(s.LogProbs[k]);
            }
            var sorted = new List<float>(s.Offered);
            sorted.Sort((a, b) => b.CompareTo(a));
            o.Margin = sorted.Count > 1 ? sorted[0] - sorted[1] : double.NaN;
            o.ChoiceCode = s.Choice;
            o.TiedCodes = s.Tied;
            o.Reason = s.Reason;
            if (s.Choice == asset.AskCode)
            {
                o.AskBasis = s.Tied != null && s.Tied.Length > 1 ? "exact_tie" : "model_selected";
                return Stop(o, "ask", s.Reason, o.AskBasis == "exact_tie" ? "the top offered scores tied exactly; the tie rule gives ASK"
                                                                          : "the model's top offered score was ASK", total);
            }
            o.TargetObjectId = targets[Array.IndexOf(codes, s.Choice)];
            return Stop(o, "completed", s.Reason, "", total);
        }
    }
}
