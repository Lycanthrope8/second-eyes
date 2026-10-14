using System;
using System.Collections.Generic;
using SecondEyes.Grounding.Parsing;

namespace SecondEyes.Grounding
{
    /// <summary>
    /// A2.7 (D115): the parser pipeline's native calls on one loaded model, through LlamaNative. It owns the model's
    /// grammar (the v1 GBNF, ending with the configured end token). Like LlamaNative, call it only from the model's
    /// worker thread (LlamaRuntime.WithModel); the desktop test project calls it directly with the host build.
    /// </summary>
    public sealed class LlamaParser : IParserNative, IParserMasks, IDisposable
    {
        private readonly IntPtr model;
        private IntPtr grammar;
        private readonly Dictionary<int, byte[]> pieces = new Dictionary<int, byte[]>();

        public LlamaParser(IntPtr model, string gbnf, int endToken)
        {
            this.model = model;
            grammar = LlamaNative.se_grammar_new(model, LlamaNative.Utf8(gbnf), LlamaNative.Utf8("root"), endToken);
            if (grammar == IntPtr.Zero) throw new InvalidOperationException("the grammar was refused: " + LastError());
        }

        public int[] Tokenize(byte[] utf8)
        {
            var z = new byte[utf8.Length + 1];
            Buffer.BlockCopy(utf8, 0, z, 0, utf8.Length);
            int cap = Math.Max(64, utf8.Length + 8);
            var buf = new int[cap];
            int n = LlamaNative.se_tokenize(model, z, buf, cap);
            if (n < 0) return null;
            var ids = new int[n];
            Array.Copy(buf, ids, n);
            return ids;
        }

        public int Decode(int[] prompt, int maxNew, int[] output, out int status)
        {
            return LlamaNative.se_decode_constrained(model, grammar, prompt, prompt.Length, maxNew, output, out status);
        }

        public byte[] TokenBytes(int token)
        {
            byte[] b;
            if (pieces.TryGetValue(token, out b)) return b;
            var buf = new byte[256];
            int n = LlamaNative.se_piece(model, token, buf, buf.Length);
            if (n < 0) return null;
            b = new byte[n];
            Array.Copy(buf, b, n);
            pieces[token] = b;
            return b;
        }

        /// <summary>The tokens the grammar allows after this prefix, as their count and SHA-256 of the sorted ID list
        /// (decimal, comma-separated), for the headset's mask check; one full-vocabulary application.</summary>
        public int AllowedAfter(int[] prefix, out string idsSha256)
        {
            idsSha256 = null;
            LlamaNative.se_grammar_reset(grammar);
            foreach (int t in prefix) if (LlamaNative.se_grammar_accept(grammar, t) != 0) return -1;
            int nVocab = LlamaNative.se_n_vocab(model);
            var mask = new byte[nVocab];
            int count = LlamaNative.se_grammar_allowed(grammar, mask, nVocab);
            if (count < 0) return -1;
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < nVocab; i++) if (mask[i] != 0) { if (sb.Length > 0) sb.Append(','); sb.Append(i); }
            idsSha256 = Replay.ReplayHash.Sha256Hex(System.Text.Encoding.ASCII.GetBytes(sb.ToString()));
            return count;
        }

        public string LastError() { return LlamaNative.Str(LlamaNative.se_last_error()); }

        public void Dispose()
        {
            if (grammar != IntPtr.Zero) { LlamaNative.se_grammar_free(grammar); grammar = IntPtr.Zero; }
        }
    }
}
