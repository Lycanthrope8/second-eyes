using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SecondEyes.Grounding.Replay;
using UnityEngine;

namespace SecondEyes.Grounding.Parsing
{
    /// <summary>
    /// A2.7 (D115): the headset acceptance check of the parser path, started from the panel like A2.5's replay. It
    /// verifies the pushed parser folder (files/parser, manifest last), checks the model file against the exported
    /// candidate's SHA-256, loads it with the accepted settings (2 threads, default flags) and the folder's context, runs
    /// the parser's startup check and then ParserAcceptance on the model's thread, and writes everything under
    /// files/parser-results/STAMP: decodes, masks, resolutions and readers (JSONL, a row as it completes), run.json (the
    /// identity: folder manifest, model hash, llama.cpp version and system info, memory) and done.json last. The PC
    /// judges: python -m grounding.a27 headset-judge. Like Replay, it loads its own model and frees it after.
    /// </summary>
    public sealed class ParserCheckRunner : IDisposable
    {
        public static string FolderDir { get { return Path.Combine(Application.persistentDataPath, "parser"); } }
        public static string ResultsRoot { get { return Path.Combine(Application.persistentDataPath, "parser-results"); } }
        public static bool FolderPresent { get { return ParserFolder.Present(FolderDir); } }

        private readonly SynchronizationContext main;
        private readonly Action<string> status;
        private LlamaRuntime runtime;

        public ParserCheckRunner(SynchronizationContext main, Action<string> status)
        {
            this.main = main;
            this.status = status;
        }

        private void Say(string text) { main.Post(_ => status(text), null); }

        private static string FileSha256(string path)
        {
            using (var sha = SHA256.Create())
            using (var fs = File.OpenRead(path))
            {
                byte[] h = sha.ComputeHash(fs);
                var sb = new StringBuilder();
                foreach (byte b in h) sb.Append(b.ToString("x2", CultureInfo.InvariantCulture));
                return sb.ToString();
            }
        }

        public async Task<string> RunAsync()
        {
            string stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            string dir = Path.Combine(ResultsRoot, stamp);
            Directory.CreateDirectory(dir);
            Say("Parser check: verifying the parser folder...");
            ParserFolder f = await Task.Run(() => ParserFolder.Load(FolderDir));
            string model = Path.Combine(Application.persistentDataPath, f.GgufFile);
            if (!File.Exists(model)) throw new FileNotFoundException("the parser model " + f.GgufFile + " isn't on the headset");
            Say("Parser check: hashing the model...");
            string ggufSha = await Task.Run(() => FileSha256(model));
            if (ggufSha != f.GgufSha256) throw new InvalidDataException("the model on the headset is not the exported candidate");
            long rssBefore = LlamaRuntime.MemoryKb(false), peakBefore = LlamaRuntime.MemoryKb(true);
            runtime = new LlamaRuntime(main);
            string captureError = null;
            double loadMs = await runtime.LoadCapturedAsync(model, f.Assets.ContextTokens, ReplayRunner.Threads, ReplayRunner.Flags,
                                                            Path.Combine(dir, "startup-log.txt"),
                                                            e => captureError = captureError == null ? e : captureError + "; " + e);
            long rssAfter = LlamaRuntime.MemoryKb(false), peakAfter = LlamaRuntime.MemoryKb(true);
            LlamaParser parser = await runtime.WithModel(s => new LlamaParser(s, f.Gbnf, f.Assets.EndToken));
            ParserAcceptance.Summary sum;
            string startup, system;
            try
            {
                startup = await runtime.WithModel(s => f.StartupCheck(parser));
                system = await runtime.WithModel(s => LlamaNative.Str(LlamaNative.se_system_info()));
                Say("Parser check: running " + f.Manifest["counts"]["decodes"].Text + " decodes, " + f.Manifest["counts"]["masks"].Text + " masks, "
                    + f.Manifest["counts"]["resolutions"].Text + " resolutions...");
                sum = await runtime.WithModel(s => ParserAcceptance.Run(f, parser, parser, dir, Say, null));
            }
            finally
            {
                await runtime.WithModel(s => { parser.Dispose(); return 0; });   // the grammar goes before its model
            }
            long peakEnd = LlamaRuntime.MemoryKb(true);
            var utf8 = new UTF8Encoding(false);
            File.WriteAllText(Path.Combine(dir, "run.json"), new JsonWriter().BeginObj().Key("record_type").S("a27_parser_check_run")
                .Key("results").S(stamp).Key("manifest_sha256").S(f.ManifestSha256).Key("candidate").S(f.Candidate)
                .Key("gguf_file").S(f.GgufFile).Key("gguf_sha256").S(ggufSha).Key("llama_cpp").S(runtime.Version).Key("system_info").S(system)
                .Key("app_version").S(Application.version).Key("context_tokens").I(f.Assets.ContextTokens).Key("threads").I(ReplayRunner.Threads)
                .Key("flags").I(ReplayRunner.Flags).Key("load_ms").D(loadMs).Key("startup_check").S(startup ?? "passed")
                .Key("capture_error").S(captureError).Key("memory_kb").BeginObj().Key("before_load_rss").I(rssBefore).Key("before_load_peak").I(peakBefore)
                .Key("after_load_rss").I(rssAfter).Key("after_load_peak").I(peakAfter).Key("end_peak").I(peakEnd).End()
                .Key("decodes").I(sum.Decodes).Key("masks").I(sum.Masks).Key("resolutions").I(sum.Resolutions).Key("readers").I(sum.Readers)
                .Key("technical").I(sum.Technical).Key("seconds").D(sum.Seconds).End().ToString() + "\n", utf8);
            File.WriteAllText(Path.Combine(dir, "done.json"), new JsonWriter().BeginObj().Key("record_type").S("a27_parser_check_done")
                .Key("results").S(stamp).End().ToString() + "\n", utf8);
            return "Parser check " + stamp + ": " + sum.Decodes + " decodes, " + sum.Masks + " masks, " + sum.Resolutions + " resolutions, "
                   + sum.Readers + " readers, " + sum.Technical + " technical failures; startup check " + (startup ?? "passed")
                   + ". Pull files/parser-results/" + stamp + ".";
        }

        public void Dispose()
        {
            if (runtime != null) runtime.Dispose();
            runtime = null;
        }
    }
}
