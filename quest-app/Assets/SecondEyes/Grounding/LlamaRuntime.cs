using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SecondEyes.Grounding
{
    /// <summary>
    /// Runs llama.cpp for the panel (A1.8c, D57, D58). One dedicated worker thread makes every native call, so Unity's
    /// frame loop never waits on the model: callers await Tasks, and answer tokens are posted to the given
    /// SynchronizationContext (Unity's main thread). The scene, the prompt up to its last object line, stays in the
    /// cache: PrepareAsync evaluates it only when it changes, so a command evaluates only its own tokens. Scores are
    /// log-probabilities of each candidate ID and the closing text after the answer's start, all candidates in one batch
    /// (se_score_many). Pure C#, no Unity types, so it runs in tests outside Unity too.
    /// </summary>
    public sealed class LlamaRuntime : IDisposable
    {
        public sealed class Prepared
        {
            public int[] Ids;          // the whole prompt's tokens
            public int SceneTokens;    // how many of them the cache holds as the scene (0: none)
            public bool SceneReused;   // the scene came from the cache
            public double SceneMs;     // time to evaluate the scene, when it wasn't reused
        }

        public sealed class Answer
        {
            public string Text = "";
            public int[] TokenIds = new int[0];
            public string Stop = "max_new_tokens";   // or "eos", or "stopped"
            public double FirstMs, TotalMs;          // on the caller's clock
        }

        public sealed class Scores
        {
            public Dictionary<string, double> LogProbs = new Dictionary<string, double>();
            public string Best;
            public double Ms;
        }

        private const int Sequences = 16;   // se_score_many scores up to Sequences - 1 candidates per batch
        private readonly SynchronizationContext main;
        private readonly BlockingCollection<Action> work = new BlockingCollection<Action>();
        private readonly Thread worker;
        private IntPtr model = IntPtr.Zero;
        private int[] scene = new int[0];
        private bool disposed;

        public string Version { get; private set; }
        public string SystemInfo { get; private set; }
        public bool IsLoaded { get { return model != IntPtr.Zero; } }
        public int WorkerThreadId { get; private set; }

        public LlamaRuntime(SynchronizationContext main)
        {
            this.main = main;
            worker = new Thread(Loop) { IsBackground = true, Name = "SecondEyes llama.cpp" };
            worker.Start();
        }

        private void Loop()
        {
            WorkerThreadId = Thread.CurrentThread.ManagedThreadId;
            foreach (Action job in work.GetConsumingEnumerable())
            {
                job();
            }
        }

        private Task<T> Run<T>(Func<T> job)
        {
            var done = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
            try
            {
                work.Add(() =>
                {
                    try { done.SetResult(job()); }
                    catch (Exception e) { done.SetException(e); }
                });
            }
            catch (InvalidOperationException)
            {
                done.SetException(new ObjectDisposedException("LlamaRuntime"));
            }
            return done.Task;
        }

        /// <summary>Loads a GGUF model; returns the time it took, in ms. flags: LlamaNative.NoRepack and the others.</summary>
        public Task<double> LoadAsync(string path, int nCtx, int threads, int flags)
        {
            return Run(() =>
            {
                if (model != IntPtr.Zero) throw new InvalidOperationException("The model is already loaded.");
                var watch = Stopwatch.StartNew();
                model = LlamaNative.se_load_ex(LlamaNative.Utf8(path), nCtx, threads, Sequences, flags);
                if (model == IntPtr.Zero) throw new InvalidOperationException(LlamaNative.LastError());
                Version = LlamaNative.Str(LlamaNative.se_llama_version());
                SystemInfo = LlamaNative.Str(LlamaNative.se_system_info()).Trim();
                return watch.Elapsed.TotalMilliseconds;
            });
        }

        public Task SetThreadsAsync(int threads)
        {
            return Run(() => { LlamaNative.se_set_threads(Loaded(), threads); return 0; });
        }

        /// <summary>This process's memory in KB: peak or current resident size.</summary>
        public static long MemoryKb(bool peak) { return LlamaNative.se_memory_kb(peak ? 1 : 0); }

        /// <summary>Tokenizes the formatted prompt, and evaluates the scene (formatted[0:sceneChars]) unless the cache
        /// already holds it. With sceneChars 0, or a scene that isn't a token prefix of the prompt, nothing is cached.</summary>
        public Task<Prepared> PrepareAsync(string formatted, int sceneChars)
        {
            return Run(() =>
            {
                IntPtr s = Loaded();
                int[] ids = Tokenize(s, formatted);
                int[] want = sceneChars > 0 && sceneChars < formatted.Length ? Tokenize(s, formatted.Substring(0, sceneChars))
                                                                           : new int[0];
                if (!StartsWith(ids, want)) want = new int[0];
                var p = new Prepared { Ids = ids, SceneTokens = want.Length };
                if (want.Length > 0 && Same(scene, want) && LlamaNative.se_n_cached(s) >= want.Length)
                {
                    p.SceneReused = true;
                }
                else if (want.Length > 0)
                {
                    var watch = Stopwatch.StartNew();
                    Eval(s, want, 0);
                    p.SceneMs = watch.Elapsed.TotalMilliseconds;
                    scene = want;
                }
                else
                {
                    scene = new int[0];
                }
                return p;
            });
        }

        /// <summary>The greedy answer after the prepared prompt, as Meta's runner and the PC references generate it.
        /// onToken receives each piece and its time on `clock`, on the main thread. Stops between tokens when asked.</summary>
        public Task<Answer> AnswerAsync(Prepared p, int maxNew, Stopwatch clock, Action<string, double> onToken,
                                        CancellationToken stop)
        {
            return Run(() =>
            {
                IntPtr s = Loaded();
                var rest = new int[p.Ids.Length - p.SceneTokens];
                Array.Copy(p.Ids, p.SceneTokens, rest, 0, rest.Length);
                Eval(s, rest, p.SceneTokens);
                var answer = new Answer();
                var ids = new List<int>();
                var text = new StringBuilder();
                Decoder utf8 = Encoding.UTF8.GetDecoder();
                for (int i = 0; i < maxNew; i++)
                {
                    if (stop.IsCancellationRequested)
                    {
                        answer.Stop = "stopped";
                        break;
                    }
                    int token = LlamaNative.se_argmax(s);
                    if (LlamaNative.se_is_eog(s, token) != 0)
                    {
                        answer.Stop = "eos";
                        break;
                    }
                    ids.Add(token);
                    string piece = Piece(s, token, utf8);
                    text.Append(piece);
                    double ms = clock.Elapsed.TotalMilliseconds;
                    if (i == 0) answer.FirstMs = ms;
                    Post(onToken, piece, ms);
                    if (i + 1 < maxNew) Eval(s, new[] { token }, LlamaNative.se_n_cached(s));
                }
                answer.Text = text.ToString();
                answer.TokenIds = ids.ToArray();
                answer.TotalMs = clock.Elapsed.TotalMilliseconds;
                return answer;
            });
        }

        /// <summary>Each candidate's log-probability after the prompt and answerPrefix, the candidate followed by
        /// suffix; any answer generated meanwhile is dropped first.</summary>
        public Task<Scores> ScoreAsync(Prepared p, string answerPrefix, IList<string> candidates, string suffix)
        {
            return Run(() =>
            {
                IntPtr s = Loaded();
                var watch = Stopwatch.StartNew();
                Eval(s, Tokenize(s, answerPrefix), p.Ids.Length);
                var result = new Scores();
                for (int start = 0; start < candidates.Count; start += Sequences - 1)
                {
                    int n = Math.Min(Sequences - 1, candidates.Count - start);
                    var flat = new List<int>();
                    var lengths = new int[n];
                    for (int k = 0; k < n; k++)
                    {
                        int[] t = Tokenize(s, candidates[start + k] + suffix);
                        flat.AddRange(t);
                        lengths[k] = t.Length;
                    }
                    var lp = new double[n];
                    if (LlamaNative.se_score_many(s, flat.ToArray(), lengths, n, lp) != 0)
                    {
                        throw new InvalidOperationException(LlamaNative.LastError());
                    }
                    for (int k = 0; k < n; k++) result.LogProbs[candidates[start + k]] = lp[k];
                }
                foreach (KeyValuePair<string, double> kv in result.LogProbs)
                {
                    if (result.Best == null || kv.Value > result.LogProbs[result.Best]) result.Best = kv.Key;
                }
                result.Ms = watch.Elapsed.TotalMilliseconds;
                return result;
            });
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            try
            {
                work.Add(() =>
                {
                    if (model != IntPtr.Zero) LlamaNative.se_free(model);
                    model = IntPtr.Zero;
                });
            }
            catch (InvalidOperationException) { }
            work.CompleteAdding();
            worker.Join(5000);
        }

        private IntPtr Loaded()
        {
            if (model == IntPtr.Zero) throw new InvalidOperationException("The llama.cpp model isn't loaded.");
            return model;
        }

        private void Post(Action<string, double> onToken, string piece, double ms)
        {
            if (onToken == null) return;
            if (main != null) main.Post(_ => onToken(piece, ms), null);
            else onToken(piece, ms);
        }

        private static void Eval(IntPtr s, int[] tokens, int keep)
        {
            if (LlamaNative.se_eval(s, tokens, tokens.Length, keep) != 0)
            {
                throw new InvalidOperationException(LlamaNative.LastError());
            }
        }

        private static int[] Tokenize(IntPtr s, string text)
        {
            byte[] utf8 = LlamaNative.Utf8(text);
            var buffer = new int[utf8.Length + 16];
            int n = LlamaNative.se_tokenize(s, utf8, buffer, buffer.Length);
            if (n < 0)
            {
                buffer = new int[-n];
                n = LlamaNative.se_tokenize(s, utf8, buffer, buffer.Length);
            }
            var ids = new int[Math.Max(0, n)];
            Array.Copy(buffer, ids, ids.Length);
            return ids;
        }

        private static string Piece(IntPtr s, int token, Decoder utf8)
        {
            var bytes = new byte[256];
            int n = LlamaNative.se_piece(s, token, bytes, bytes.Length);
            if (n <= 0) return "";
            var chars = new char[utf8.GetCharCount(bytes, 0, n)];
            utf8.GetChars(bytes, 0, n, chars, 0);   // a token can end mid-character; the decoder carries the rest over
            return new string(chars);
        }

        private static bool StartsWith(int[] ids, int[] prefix)
        {
            if (prefix.Length > ids.Length) return false;
            for (int i = 0; i < prefix.Length; i++) if (ids[i] != prefix[i]) return false;
            return true;
        }

        private static bool Same(int[] a, int[] b)
        {
            return a.Length == b.Length && StartsWith(a, b);
        }
    }
}
