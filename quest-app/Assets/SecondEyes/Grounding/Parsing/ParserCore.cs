using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Replay;
using SecondEyes.Grounding.Resolution;

namespace SecondEyes.Grounding.Parsing
{
    // A2.7 delivery 2 (D115): the deployed path, free of Unity so it is tested on a PC. A request (a registered
    // snapshot's scene, command and category map) becomes one structured outcome:
    //   1. the records, strictly parsed;
    //   2. the parser prompt (settings.v1.json's wrapper around the command text), tokenized natively;
    //   3. grammar-constrained greedy decoding, from a fresh cache (se_decode_constrained);
    //   4. the decoded string read by the v1 language (parse_output), then its interpretation;
    //   5. A2.4's checker: the grounding query, its validation, the resolver.
    // Outcome kinds stay distinct (D115): selected (resolved); ask (ambiguous, no_match, insufficient_information: a
    // decline the app turns into a request for clarification); unsupported (a label outside the vocabulary); invalid
    // (invalid_input records or an invalid_label_query); budget (a resolver limit); failed (a technical failure: the
    // native side, decoding that didn't complete, output outside the language, or a resolver fault).

    /// <summary>The native calls the parser needs (LlamaParser on the headset, a fake in tests).</summary>
    public interface IParserNative
    {
        int[] Tokenize(byte[] utf8);
        /// <summary>se_decode_constrained: tokens written (the end token included when chosen), status 0 completed,
        /// 1 length exhausted, 2 empty allowed set; a negative return or status is a technical failure.</summary>
        int Decode(int[] prompt, int maxNew, int[] output, out int status);
        byte[] TokenBytes(int token);
        string LastError();
    }

    /// <summary>The full allowed-token check the headset acceptance needs (LlamaParser on the headset).</summary>
    public interface IParserMasks
    {
        /// <summary>The count of tokens the grammar allows after this prefix and the SHA-256 of their sorted IDs
        /// (decimal, comma-separated); a negative count when the grammar refuses the prefix.</summary>
        int AllowedAfter(int[] prefix, out string idsSha256);
    }

    /// <summary>settings.v1.json's parser_prompt: the chat wrapper around the system message and the command text.</summary>
    public sealed class ParserPrompt
    {
        public string BeforeSystem, SystemMessage, Between, AfterUser;

        public static ParserPrompt From(JNode parserPrompt)
        {
            var w = parserPrompt["wrapper"];
            return new ParserPrompt { BeforeSystem = w["before_system"].Text, SystemMessage = parserPrompt["system_message"].Text,
                                      Between = w["between"].Text, AfterUser = w["after_user"].Text };
        }

        public string Build(string text) { return BeforeSystem + SystemMessage + Between + text + AfterUser; }
    }

    /// <summary>Everything the pipeline loads once, verified at load (the asset manifest's SHA-256 is logged with every
    /// outcome, so each run names the configuration that produced it).</summary>
    public sealed class ParserAssets
    {
        public LanguageSpec Spec;
        public ParserPrompt Prompt;
        public Checker Checker;
        public int EndToken, MaxNewTokens, ContextTokens = 1024;
        public string ManifestSha256;
    }

    public sealed class ParserRequest
    {
        public string RequestId, Source, SnapshotId, SnapshotSha256;
        public string SceneText, CommandText, MapText;   // the scene, command and category-map records as JSON text
    }

    public sealed class ParserOutcome
    {
        public string RequestId, Source, Status = "failed", Outcome, Reason, Detail = "";
        public string TargetObjectId, DecodedText, DecodeStatus, ReasonCode, PromptSha256, AssetsSha256;
        public List<string> CandidateIds = new List<string>();
        public List<int> OutputTokens = new List<int>();
        public bool Conditional;
        public int PromptTokens = -1;
        public double ParseMs, TokenizeMs, DecodeMs, ResolveMs, TotalMs;

        public string ToJson()
        {
            var w = new JsonWriter().BeginObj().Key("record_type").S("a27_parser_outcome").Key("request_id").S(RequestId)
                .Key("source").S(Source).Key("status").S(Status).Key("outcome").S(Outcome).Key("reason").S(Reason).Key("detail").S(Detail)
                .Key("target_object_id").S(TargetObjectId).Key("candidate_ids").Strs(CandidateIds.ToArray()).Key("reason_code").S(ReasonCode)
                .Key("conditional").B(Conditional).Key("decoded_text").S(DecodedText).Key("decode_status").S(DecodeStatus)
                .Key("output_tokens").BeginArr();
            foreach (int t in OutputTokens) w.I(t);
            return w.End().Key("prompt_tokens").I(PromptTokens).Key("prompt_sha256").S(PromptSha256)
                .Key("assets_sha256").S(AssetsSha256).Key("ms").BeginObj().Key("parse").D(ParseMs).Key("tokenize").D(TokenizeMs)
                .Key("decode").D(DecodeMs).Key("resolve").D(ResolveMs).Key("total").D(TotalMs).End().End().ToString();
        }
    }

    /// <summary>The contract's file layer for records arriving as text: no byte-order mark, well-formed JSON, no key
    /// repeated in an object. Numbers too large for a double are left to the contract's E_NONFINITE.</summary>
    public static class StrictJson
    {
        public static JNode Parse(string text, out string code, out string message)
        {
            code = null; message = null;
            if (text == null) { code = "E_RECORD_MISSING"; message = "no record was supplied"; return null; }
            if (text.Length > 0 && text[0] == '\uFEFF') { code = "E_PARSE_ENCODING"; message = "starts with a byte-order mark"; return null; }
            JNode n;
            try { n = JParse.Parse(text); }
            catch (Exception e) { code = "E_PARSE_JSON"; message = e.Message; return null; }
            string dup = Repeated(n);
            if (dup != null) { code = "E_PARSE_DUPLICATE_KEY"; message = "key " + dup + " appears twice in one object"; return null; }
            return n;
        }

        private static string Repeated(JNode n)
        {
            if (n.Kind == JKind.Object)
            {
                var seen = new HashSet<string>(StringComparer.Ordinal);
                foreach (var f in n.Fields)
                {
                    if (!seen.Add(f.Key)) return f.Key;
                    string d = Repeated(f.Value);
                    if (d != null) return d;
                }
            }
            else if (n.Kind == JKind.Array)
                foreach (var x in n.Items) { string d = Repeated(x); if (d != null) return d; }
            return null;
        }
    }

    public static class ParserPipeline
    {
        private static ParserOutcome Stop(ParserOutcome o, string status, string outcome, string reason, string detail, Stopwatch total)
        {
            o.Status = status;
            o.Outcome = outcome;
            o.Reason = reason;
            o.Detail = detail ?? "";
            o.TotalMs = total.Elapsed.TotalMilliseconds;
            return o;
        }

        /// <summary>The checker's outcome as one of the distinct kinds.</summary>
        public static string Kind(string outcome)
        {
            switch (outcome)
            {
                case "resolved": return "selected";
                case "ambiguous": case "no_match": case "insufficient_information": return "ask";
                case "unsupported": return "unsupported";
                case "invalid_input": case "invalid_label_query": return "invalid";
                case "budget_exceeded": return "budget";
                default: return "failed";
            }
        }

        /// <summary>One request, start to finish. Never throws: every way out is a structured outcome.</summary>
        public static ParserOutcome Run(ParserRequest r, ParserAssets a, IParserNative native, Func<bool> cancelled)
        {
            var total = Stopwatch.StartNew();
            var o = new ParserOutcome { RequestId = r.RequestId, Source = r.Source, AssetsSha256 = a.ManifestSha256 };
            try
            {
                var t = Stopwatch.StartNew();
                string code, message;
                JNode scene = StrictJson.Parse(r.SceneText, out code, out message);
                if (scene == null) return Stop(o, "invalid", "invalid_input", code, "scene: " + message, total);
                JNode command = StrictJson.Parse(r.CommandText, out code, out message);
                if (command == null) return Stop(o, "invalid", "invalid_input", code, "command: " + message, total);
                JNode cmap = StrictJson.Parse(r.MapText, out code, out message);
                if (cmap == null) return Stop(o, "invalid", "invalid_input", code, "category map: " + message, total);
                if (command.Kind != JKind.Object || !command.Has("text") || command["text"].Kind != JKind.String)
                    return Stop(o, "invalid", "invalid_input", "E_COMMAND_TEXT", "the command record has no text", total);
                o.ParseMs = t.Elapsed.TotalMilliseconds;
                if (cancelled != null && cancelled()) return Stop(o, "failed", "cancelled", "cancelled", null, total);

                t.Restart();
                string prompt = a.Prompt.Build(command["text"].Text);
                byte[] utf8 = new UTF8Encoding(false).GetBytes(prompt);
                o.PromptSha256 = ReplayHash.Sha256Hex(utf8);
                int[] ids = native.Tokenize(utf8);
                o.TokenizeMs = t.Elapsed.TotalMilliseconds;
                if (ids == null || ids.Length == 0) return Stop(o, "failed", "technical_failure", "tokenize", native.LastError(), total);
                o.PromptTokens = ids.Length;
                if (ids.Length + a.MaxNewTokens > a.ContextTokens)
                    return Stop(o, "failed", "technical_failure", "context", ids.Length + " prompt tokens leave no room to decode", total);
                if (cancelled != null && cancelled()) return Stop(o, "failed", "cancelled", "cancelled", null, total);

                t.Restart();
                var output = new int[a.MaxNewTokens];
                int status;
                int n = native.Decode(ids, a.MaxNewTokens, output, out status);
                o.DecodeMs = t.Elapsed.TotalMilliseconds;
                if (n < 0 || status < 0) return Stop(o, "failed", "technical_failure", "decode", native.LastError(), total);
                var bytes = new List<byte>();
                for (int i = 0; i < n; i++)
                {
                    o.OutputTokens.Add(output[i]);
                    if (output[i] == a.EndToken) continue;
                    byte[] piece = native.TokenBytes(output[i]);
                    if (piece == null) return Stop(o, "failed", "technical_failure", "token_bytes", native.LastError(), total);
                    bytes.AddRange(piece);
                }
                o.DecodedText = Encoding.UTF8.GetString(bytes.ToArray());
                o.DecodeStatus = status == 0 ? "completed" : (status == 1 ? "length_exhausted" : "empty_allowed_set");
                if (status != 0) return Stop(o, "failed", "decode:" + o.DecodeStatus, "decode", null, total);

                Parsed parsed;
                try { parsed = Language.ParseOutput(o.DecodedText, a.Spec); }
                catch (LanguageError e) { return Stop(o, "failed", "invalid_output", "language", e.Message, total); }
                if (cancelled != null && cancelled()) return Stop(o, "failed", "cancelled", "cancelled", null, total);

                t.Restart();
                var co = a.Checker.Resolve(Language.Interpretation(parsed), scene, command, cmap);
                o.ResolveMs = t.Elapsed.TotalMilliseconds;
                o.TargetObjectId = co.TargetId;
                o.CandidateIds = co.CandidateIds;
                o.ReasonCode = co.ReasonCode;
                o.Conditional = co.Run != null && co.Run.Result != null && co.Run.Result.Conditional;
                string detail = null;
                if (co.Outcome == "invalid_label_query" && co.QueryIssues.Count > 0) detail = co.QueryIssues[0].ToString();
                else if (co.Outcome == "invalid_input" && co.Run.Issues.Count > 0) detail = co.Run.Issues[0].ToString();
                else if (co.Outcome == "budget_exceeded") detail = co.Run.Budget.Name;
                return Stop(o, Kind(co.Outcome), co.Outcome, co.Outcome, detail, total);
            }
            catch (ResolverFault e) { return Stop(o, "failed", "technical_failure", "resolver_fault", e.Message, total); }
            catch (Exception e) { return Stop(o, "failed", "technical_failure", "pipeline_exception", e.GetType().Name + ": " + e.Message, total); }
        }
    }
}
