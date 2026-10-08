using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace SecondEyes.Grounding.Replay
{
    // A2.5 delivery 1, step 2 (D100, D103): the headset replay's logic, free of Unity so it can be tested on a PC.
    // grounding/quest/replay_inputs.py is the reference: the records, the hash rules and the order of the input checks
    // must match it exactly.

    /// <summary>One line of the bundle's requests.jsonl or fixtures-written.jsonl.</summary>
    [Serializable]
    public class ReplayRecord
    {
        public int format_version;
        public string record_type;
        public string request_id;
        public string kind;
        public string prompt_b64;
        public int prompt_bytes;
        public string prompt_sha256;
        public int[] token_ids;
        public int input_tokens;
        public string token_ids_sha256;
        public string[] codes;
        public string[] targets;
        public int[] code_token_ids;
        public string mapping_sha256;
        public int keep_before_last_token;
        public int keep_before_command_line;
        public string expected_outcome;
        public string expected_reason;
    }

    /// <summary>The bundle's headset/replay-manifest.json.</summary>
    [Serializable]
    public class ReplayManifest
    {
        public int format_version;
        public string record_type;
        public string policy_id;
        public string requests_file;
        public string requests_sha256;
        public int requests_count;
        public string fixtures_file;
        public string fixtures_sha256;
        public int fixtures_count;
        public int context_limit_tokens;
        public int continuation_tokens;
        public int vocab_size;
        public string[] object_codes;
        public string ask_code;
        public string ask_target;
        public string[] choice_codes;
        public int[] choice_token_ids;
        public string[] input_checks;
    }

    public static class ReplayHash
    {
        public static string Sha256Hex(byte[] data)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] h = sha.ComputeHash(data);
                var sb = new StringBuilder(64);
                foreach (byte b in h) sb.Append(b.ToString("x2", CultureInfo.InvariantCulture));
                return sb.ToString();
            }
        }

        /// <summary>A2.3d's rule: SHA-256 of the decimal IDs joined by commas, in ASCII.</summary>
        public static string TokenIds(int[] ids)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < ids.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(ids[i].ToString(CultureInfo.InvariantCulture));
            }
            return Sha256Hex(Encoding.ASCII.GetBytes(sb.ToString()));
        }

        /// <summary>SHA-256 of the UTF-8 lines "code TAB target TAB token ID LF", in prompt order.</summary>
        public static string Mapping(string[] codes, string[] targets, int[] ids)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < codes.Length; i++)
                sb.Append(codes[i]).Append('\t').Append(targets[i]).Append('\t')
                  .Append(ids[i].ToString(CultureInfo.InvariantCulture)).Append('\n');
            return Sha256Hex(new UTF8Encoding(false).GetBytes(sb.ToString()));
        }
    }

    /// <summary>The result of the ordered input checks; Outcome "passed" means evaluation may run.</summary>
    public sealed class InputCheck
    {
        public string Outcome = "passed";
        public string Check;
        public string Reason;
        public string Detail = "";
        public bool TokenizationSkipped;
        public byte[] Prompt;
        public int RuntimeTokens = -1;
        public int FirstTokenDifference = -1;
        public string PromptSha256, TokenIdsSha256, MappingSha256;   // as computed here
    }

    public static class ReplayChecks
    {
        public static readonly string[] Order = { "prompt_bytes", "mapping", "token_ids", "tokenization", "context" };

        private static InputCheck Stop(InputCheck r, string check, string reason, string detail)
        {
            r.Check = check;
            r.Reason = reason;
            r.Detail = detail ?? "";
            r.Outcome = check == "tokenization" ? "tokenization_mismatch"
                      : check == "context" ? "context_budget_exceeded" : "invalid_input";
            return r;
        }

        private static bool StrictBase64(string s)
        {
            if (s == null || s.Length % 4 != 0) return false;
            int pad = 0;
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                bool alpha = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+' || c == '/';
                if (c == '=') { pad++; if (i < s.Length - 2) return false; }
                else if (!alpha || pad > 0) return false;
            }
            return pad <= 2;
        }

        private static int Code(ReplayManifest m, string code)
        {
            for (int i = 0; i < m.choice_codes.Length; i++) if (m.choice_codes[i] == code) return m.choice_token_ids[i];
            return int.MinValue;
        }

        /// <summary>The five input checks in order; the first failure decides. tokenize: the runtime's own tokenizer
        /// on the complete prompt bytes, or null to skip that check (reported).</summary>
        public static InputCheck Run(ReplayRecord r, ReplayManifest m, Func<byte[], int[]> tokenize)
        {
            var res = new InputCheck();
            // 1. prompt bytes: strict base64 of UTF-8, then count and SHA-256
            if (!StrictBase64(r.prompt_b64)) return Stop(res, "prompt_bytes", "prompt_bytes_invalid", "not strict base64");
            byte[] data;
            try
            {
                data = Convert.FromBase64String(r.prompt_b64);
                new UTF8Encoding(false, true).GetString(data);
            }
            catch (Exception e) { return Stop(res, "prompt_bytes", "prompt_bytes_invalid", e.GetType().Name); }
            res.PromptSha256 = ReplayHash.Sha256Hex(data);
            if (data.Length != r.prompt_bytes)
                return Stop(res, "prompt_bytes", "prompt_bytes_invalid", data.Length + " bytes, recorded " + r.prompt_bytes);
            if (res.PromptSha256 != r.prompt_sha256)
                return Stop(res, "prompt_bytes", "prompt_hash_mismatch", "the bytes' SHA-256 differs from prompt_sha256");
            res.Prompt = data;
            // 2. mapping: structure, K and ASK last, hash, then each code's token ID
            string[] codes = r.codes, targets = r.targets;
            int[] ids = r.code_token_ids;
            int objects = m.object_codes.Length;
            if (codes == null || targets == null || ids == null || codes.Length != targets.Length || codes.Length != ids.Length
                || codes.Length < 2 || codes.Length > objects + 1)
                return Stop(res, "mapping", "mapping_invalid", "codes, targets and code_token_ids must be parallel lists of 2 to " + (objects + 1));
            int last = codes.Length - 1;
            if (codes[last] != m.ask_code || targets[last] != m.ask_target)
                return Stop(res, "mapping", "mapping_invalid", "the last choice must be " + m.ask_code + " for " + m.ask_target);
            var seenCodes = new HashSet<string>();
            var seenTargets = new HashSet<string>();
            for (int i = 0; i < last; i++)
            {
                if (codes[i] == null || Array.IndexOf(m.object_codes, codes[i]) < 0 || !seenCodes.Add(codes[i]))
                    return Stop(res, "mapping", "mapping_invalid", "object codes must be distinct letters of the protocol's object codes");
                if (string.IsNullOrEmpty(targets[i]) || targets[i] == m.ask_target || !seenTargets.Add(targets[i]))
                    return Stop(res, "mapping", "mapping_invalid", "object targets must be distinct, non-empty and not the ASK target");
            }
            res.MappingSha256 = ReplayHash.Mapping(codes, targets, ids);
            if (res.MappingSha256 != r.mapping_sha256)
                return Stop(res, "mapping", "mapping_invalid", "the mapping's SHA-256 differs from mapping_sha256");
            var wrong = new List<string>();
            for (int i = 0; i < codes.Length; i++)
            {
                int want = Code(m, codes[i]);
                if (ids[i] != want) wrong.Add(codes[i] + ": " + ids[i] + " (protocol " + want + ")");
            }
            if (wrong.Count > 0) return Stop(res, "mapping", "code_token_mismatch", string.Join("; ", wrong.ToArray()));
            // 3. token IDs: count and hash, then the vocabulary
            int[] toks = r.token_ids;
            if (toks == null || toks.Length == 0)
                return Stop(res, "token_ids", "token_ids_invalid", "token_ids must be a non-empty list");
            if (r.input_tokens != toks.Length)
                return Stop(res, "token_ids", "token_ids_invalid", toks.Length + " token IDs, recorded " + r.input_tokens);
            res.TokenIdsSha256 = ReplayHash.TokenIds(toks);
            if (res.TokenIdsSha256 != r.token_ids_sha256)
                return Stop(res, "token_ids", "token_ids_invalid", "the IDs' SHA-256 differs from token_ids_sha256");
            int outside = 0, firstPos = -1, firstVal = 0;
            for (int i = 0; i < toks.Length; i++)
                if (toks[i] < 0 || toks[i] >= m.vocab_size) { if (outside++ == 0) { firstPos = i; firstVal = toks[i]; } }
            if (outside > 0)
                return Stop(res, "token_ids", "token_out_of_vocabulary", outside + " ID(s) outside 0.." + (m.vocab_size - 1)
                            + ", first at position " + firstPos + ": " + firstVal);
            // 4. the runtime's own tokenization of the complete prompt equals the frozen IDs, token by token
            if (tokenize == null) res.TokenizationSkipped = true;
            else
            {
                int[] got = tokenize(data);
                res.RuntimeTokens = got.Length;
                int n = Math.Min(got.Length, toks.Length), first = -1;
                for (int i = 0; i < n; i++) if (got[i] != toks[i]) { first = i; break; }
                if (first < 0 && got.Length != toks.Length) first = n;
                if (first >= 0)
                {
                    res.FirstTokenDifference = first;
                    return Stop(res, "tokenization", "tokenization_mismatch", got.Length + " tokens against " + toks.Length
                                + " frozen; first difference at position " + first);
                }
            }
            // 5. the context ceiling, never truncated
            if (toks.Length + m.continuation_tokens > m.context_limit_tokens)
                return Stop(res, "context", "context_budget_exceeded", toks.Length + " tokens + " + m.continuation_tokens
                            + " > " + m.context_limit_tokens);
            return res;
        }
    }

    /// <summary>Offered-letter scores of one final position: every offered code, K included, normalized.</summary>
    public sealed class PathScore
    {
        public float[] Offered;
        public double[] LogProbs;
        public double[] Shares;
        public string Choice, Reason;
        public string[] Tied = new string[0];
        public string Error;
        public int NonFinite;
        public int FirstNonFinite = -1;
    }

    public static class ReplayScoring
    {
        /// <summary>row: the final position's full row, n values of it valid. A non-finite value anywhere is an error,
        /// never a choice. Exact ties at the top go to the ASK code.</summary>
        public static PathScore Score(float[] row, int n, string[] codes, int[] codeIds, string askCode, Func<int, double> logProb)
        {
            var s = new PathScore();
            if (n <= 0) { s.Error = "no_scores"; return s; }
            for (int i = 0; i < n; i++)
                if (float.IsNaN(row[i]) || float.IsInfinity(row[i])) { if (s.NonFinite++ == 0) s.FirstNonFinite = i; }
            if (s.NonFinite > 0) { s.Error = "non_finite_output"; return s; }
            int k = codes.Length;
            s.Offered = new float[k];
            s.LogProbs = new double[k];
            s.Shares = new double[k];
            for (int i = 0; i < k; i++)
            {
                int id = codeIds[i];
                if (id < 0 || id >= n) { s.Error = "code_token_outside_row"; return s; }
                s.Offered[i] = row[id];
                s.LogProbs[i] = logProb != null ? logProb(id) : double.NaN;
            }
            double top = s.Offered[0];
            for (int i = 1; i < k; i++) if (s.Offered[i] > top) top = s.Offered[i];
            double z = 0;
            for (int i = 0; i < k; i++) z += Math.Exp(s.Offered[i] - top);
            for (int i = 0; i < k; i++) s.Shares[i] = Math.Exp(s.Offered[i] - top) / z;
            var tied = new List<string>();
            for (int i = 0; i < k; i++) if (s.Offered[i] == top) tied.Add(codes[i]);
            if (tied.Count > 1) { s.Choice = askCode; s.Reason = "exact_score_tie"; s.Tied = tied.ToArray(); }
            else { s.Choice = tied[0]; s.Reason = "max_offered_logit"; }
            return s;
        }
    }

    /// <summary>On-device self-checks with synthetic rows and records, run before the model loads.</summary>
    public static class ReplaySelfChecks
    {
        public sealed class Result { public string Name; public bool Passed; public string Detail; }

        private static ReplayManifest Constants()
        {
            var m = new ReplayManifest
            {
                context_limit_tokens = 10, continuation_tokens = 1, vocab_size = 1000,
                object_codes = new[] { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J" }, ask_code = "K", ask_target = "ASK",
                choice_codes = new[] { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K" },
                choice_token_ids = new[] { 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42 }
            };
            return m;
        }

        /// <summary>The hand record of the Python tests: "hand prompt", tokens 5 6 7, choices B A K.</summary>
        public static ReplayRecord HandRecord()
        {
            byte[] prompt = Encoding.ASCII.GetBytes("hand prompt");
            var r = new ReplayRecord
            {
                request_id = "hand", kind = "dataset_command", prompt_b64 = Convert.ToBase64String(prompt), prompt_bytes = prompt.Length,
                prompt_sha256 = ReplayHash.Sha256Hex(prompt), token_ids = new[] { 5, 6, 7 }, input_tokens = 3,
                codes = new[] { "B", "A", "K" }, targets = new[] { "o2", "o1", "ASK" }, code_token_ids = new[] { 33, 32, 42 }
            };
            r.token_ids_sha256 = ReplayHash.TokenIds(r.token_ids);
            r.mapping_sha256 = ReplayHash.Mapping(r.codes, r.targets, r.code_token_ids);
            return r;
        }

        public static List<Result> Run()
        {
            var results = new List<Result>();
            Action<string, bool, string> add = (name, ok, detail) => results.Add(new Result { Name = name, Passed = ok, Detail = detail });
            string[] codes = { "B", "A", "K" };
            int[] ids = { 1, 0, 9 };
            float[] row = { 1.5f, 2.5f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -1f };
            PathScore s = ReplayScoring.Score(row, row.Length, codes, ids, "K", null);
            add("the larger offered logit wins (B, 2.5)", s.Error == null && s.Choice == "B" && s.Reason == "max_offered_logit", s.Choice + "/" + s.Reason);
            double sum = 0;
            if (s.Shares != null) foreach (double x in s.Shares) sum += x;
            add("shares over every offered code, K included, sum to 1", Math.Abs(sum - 1.0) < 1e-12, sum.ToString("R", CultureInfo.InvariantCulture));
            float[] tie = { 2.5f, 2.5f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -1f };
            s = ReplayScoring.Score(tie, tie.Length, codes, ids, "K", null);
            add("an exact tie at the top goes to K with both codes listed", s.Choice == "K" && s.Reason == "exact_score_tie"
                && s.Tied.Length == 2 && s.Tied[0] == "B" && s.Tied[1] == "A", s.Choice + "/" + string.Join(",", s.Tied));
            float[] tieK = { 3f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 3f };
            s = ReplayScoring.Score(tieK, tieK.Length, codes, ids, "K", null);
            add("a tie between an object and K also goes to K", s.Choice == "K" && s.Reason == "exact_score_tie" && s.Tied.Length == 2, s.Choice);
            float[] nan = { 1f, 2f, float.NaN, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
            s = ReplayScoring.Score(nan, nan.Length, codes, ids, "K", null);
            add("NaN anywhere in the row is an error, not a choice", s.Error == "non_finite_output" && s.FirstNonFinite == 2 && s.Choice == null, s.Error);
            float[] inf = { 1f, float.PositiveInfinity, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
            s = ReplayScoring.Score(inf, inf.Length, codes, ids, "K", null);
            add("an infinite offered logit is an error, not a choice", s.Error == "non_finite_output" && s.FirstNonFinite == 1, s.Error);
            ReplayManifest m = Constants();
            Func<byte[], int[]> same = b => new[] { 5, 6, 7 };
            InputCheck c = ReplayChecks.Run(HandRecord(), m, same);
            add("a valid record passes every input check", c.Outcome == "passed", c.Outcome + " " + c.Reason);
            ReplayRecord r = HandRecord();
            r.prompt_b64 = Convert.ToBase64String(Encoding.ASCII.GetBytes("Hand prompt"));
            r.token_ids_sha256 = new string('0', 64);
            c = ReplayChecks.Run(r, m, same);
            add("two defects: the prompt check comes first", c.Check == "prompt_bytes" && c.Reason == "prompt_hash_mismatch", c.Check + "/" + c.Reason);
            r = HandRecord();
            r.codes = new[] { "K", "B", "A" }; r.targets = new[] { "ASK", "o2", "o1" }; r.code_token_ids = new[] { 42, 33, 32 };
            r.mapping_sha256 = ReplayHash.Mapping(r.codes, r.targets, r.code_token_ids);
            c = ReplayChecks.Run(r, m, same);
            add("K first is an invalid mapping", c.Outcome == "invalid_input" && c.Reason == "mapping_invalid", c.Reason);
            c = ReplayChecks.Run(HandRecord(), m, b => new[] { 5, 6, 8 });
            add("a runtime tokenization that differs stops at tokenization, position 2",
                c.Outcome == "tokenization_mismatch" && c.FirstTokenDifference == 2, c.Outcome + " " + c.FirstTokenDifference);
            r = HandRecord();
            r.token_ids = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; r.input_tokens = 10; r.token_ids_sha256 = ReplayHash.TokenIds(r.token_ids);
            c = ReplayChecks.Run(r, m, b => new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            add("10 tokens + 1 > 10: context_budget_exceeded", c.Outcome == "context_budget_exceeded", c.Outcome);
            return results;
        }
    }

    /// <summary>A small JSON writer: commas handled, strings escaped, floats written so they read back exactly (G9 for
    /// float, G17 for double), non-finite numbers as null.</summary>
    public sealed class JsonWriter
    {
        private readonly StringBuilder sb = new StringBuilder();
        private readonly Stack<bool> filled = new Stack<bool>();   // per open container: has it an element yet
        private readonly Stack<char> closers = new Stack<char>();
        private bool afterKey;

        private void BeforeValue()
        {
            if (afterKey) { afterKey = false; return; }
            if (filled.Count == 0) return;
            if (filled.Pop()) sb.Append(',');
            filled.Push(true);
        }

        public JsonWriter BeginObj() { BeforeValue(); sb.Append('{'); filled.Push(false); closers.Push('}'); return this; }
        public JsonWriter BeginArr() { BeforeValue(); sb.Append('['); filled.Push(false); closers.Push(']'); return this; }
        public JsonWriter End() { filled.Pop(); sb.Append(closers.Pop()); return this; }

        public JsonWriter Key(string k)
        {
            if (filled.Pop()) sb.Append(',');
            filled.Push(true);
            Str(k);
            sb.Append(':');
            afterKey = true;
            return this;
        }

        private void Str(string s)
        {
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }

        public JsonWriter S(string v) { BeforeValue(); if (v == null) sb.Append("null"); else Str(v); return this; }
        public JsonWriter I(long v) { BeforeValue(); sb.Append(v.ToString(CultureInfo.InvariantCulture)); return this; }
        public JsonWriter B(bool v) { BeforeValue(); sb.Append(v ? "true" : "false"); return this; }
        public JsonWriter Null() { BeforeValue(); sb.Append("null"); return this; }

        public JsonWriter F(float v)
        {
            BeforeValue();
            if (float.IsNaN(v) || float.IsInfinity(v)) sb.Append("null");
            else sb.Append(v.ToString("G9", CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter D(double v)
        {
            BeforeValue();
            if (double.IsNaN(v) || double.IsInfinity(v)) sb.Append("null");
            else sb.Append(v.ToString("G17", CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter Strs(string[] vs) { BeginArr(); if (vs != null) foreach (string v in vs) S(v); return End(); }
        public JsonWriter Ints(int[] vs) { BeginArr(); if (vs != null) foreach (int v in vs) I(v); return End(); }
        public override string ToString() { return sb.ToString(); }
    }
}
