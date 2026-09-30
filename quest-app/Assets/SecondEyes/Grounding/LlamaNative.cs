using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SecondEyes.Grounding
{
    /// <summary>
    /// P/Invoke declarations for libse_llama (native/se_llama.h, A1.8c, D57). The library ships in
    /// Assets/Plugins/Android/libs/arm64-v8a/ (tools/build_llama.py android), so it exists on the headset only; on other
    /// platforms the first call throws DllNotFoundException. Strings cross as UTF-8 byte arrays, not through marshalling
    /// attributes, so IL2CPP and Mono treat them alike. Call these only from LlamaRuntime's worker thread.
    /// </summary>
    internal static class LlamaNative
    {
        private const string Lib = "se_llama";

        public const int NoRepack = 1, FlashAttnOff = 2, FlashAttnOn = 4, NoMmap = 8;   // se_load_ex's options

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr se_system_info();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr se_llama_version();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr se_last_error();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr se_load_ex(byte[] path, int nCtx, int nThreads, int nSeq, int flags);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern void se_free(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern void se_set_threads(IntPtr s, int n);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int se_n_ctx(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int se_n_seq(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int se_tokenize(IntPtr s, byte[] text, [Out] int[] tokens, int max);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int se_piece(IntPtr s, int token, [Out] byte[] buffer, int max);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int se_is_eog(IntPtr s, int token);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int se_n_cached(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int se_eval(IntPtr s, int[] tokens, int n, int keep);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int se_argmax(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int se_score_many(IntPtr s, int[] tokens, int[] lengths, int n, [Out] double[] logprobs);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern long se_memory_kb(int peak);

        /// <summary>A NUL-terminated UTF-8 copy of a string, as the C side expects.</summary>
        public static byte[] Utf8(string text)
        {
            byte[] body = Encoding.UTF8.GetBytes(text ?? "");
            var z = new byte[body.Length + 1];
            Buffer.BlockCopy(body, 0, z, 0, body.Length);
            return z;
        }

        /// <summary>A NUL-terminated UTF-8 string from the C side.</summary>
        public static string Str(IntPtr p)
        {
            if (p == IntPtr.Zero) return "";
            var bytes = new List<byte>();
            for (int i = 0; ; i++)
            {
                byte b = Marshal.ReadByte(p, i);
                if (b == 0) break;
                bytes.Add(b);
            }
            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        public static string LastError() { return Str(se_last_error()); }
    }
}
