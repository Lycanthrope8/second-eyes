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
    /// <summary>A2.5 delivery 4 on the headset: one interactive session, fed by the ADB inbox and the panel's presets
    /// through one path (D99(1)), operational, so uncached path U only.
    ///   Scenes: the three annotated golden snapshots (3, 6 and 10 objects), read through the golden check's
    ///     hash-checked reader. The panel switches between them; each switch is a new epoch.
    ///   Inbox: files/interactive/inbox/(request ID).json, written under a temporary name and renamed when complete
    ///     (D99(2)). Each request ID is processed once. Its acknowledgement (outbox/(ID).ack.json) and result
    ///     (outbox/(ID).result.json) are written under the same ID. Processed files move to inbox/processed/.
    ///   Times: every stage on one headset stopwatch (D99(3)).
    ///   Log: sessions/(UTC time)/outcomes.jsonl, one line per answered request, presets included.</summary>
    public sealed class InteractiveService : IDisposable
    {
        public sealed class Scene { public string SnapshotId, SceneId, SceneRevision, Sha256, Text; public int Objects; public JNode Record; }
        public sealed class Preset { public string Label, CommandText; }

        public static string Root { get { return Path.Combine(Application.persistentDataPath, "interactive"); } }
        public static string InboxDir { get { return Path.Combine(Root, "inbox"); } }
        public static string OutboxDir { get { return Path.Combine(Root, "outbox"); } }

        public readonly List<Scene> Scenes = new List<Scene>();
        public readonly Dictionary<string, List<Preset>> Presets = new Dictionary<string, List<Preset>>();
        public SceneBinding Current { get; private set; }
        public int Answered { get; private set; }
        public int Waiting { get { return core.Waiting; } }

        private readonly SynchronizationContext main;
        private readonly Action<Ticket, InteractiveOutcome> onOutcome;
        private readonly Action<string> status;
        private readonly SessionCore core = new SessionCore();
        private readonly PrefixCache cache = new PrefixCache(CacheMode.Off);
        private readonly Stopwatch clock = Stopwatch.StartNew();
        private readonly object logGate = new object();
        private CancellationTokenSource stop, inboxStop;
        private Task inboxTask, workerTask;   // kept, so that shutdown can await them (ChatGPT's A2.5 review)

        public string SessionStamp { get { return sessionStamp; } }
        public PromptAsset Asset { get { return goldens != null ? goldens.Asset : null; } }
        private LlamaRuntime runtime;
        private GoldenRunner.Goldens goldens;
        private StreamWriter log;
        private string sessionStamp;
        private int presetCount, epoch;

        public InteractiveService(SynchronizationContext main, Action<string> status, Action<Ticket, InteractiveOutcome> onOutcome)
        {
            this.main = main;
            this.status = status;
            this.onOutcome = onOutcome;
        }

        private double Now { get { return clock.Elapsed.TotalMilliseconds; } }
        private void Say(string text) { main.Post(_ => status(text), null); }
        private void Event(string ev, string json) { main.Post(_ => EventLog.Write(ev, json), null); }

        private static void WriteAtomic(string path, string json)
        {
            string tmp = path + ".tmp";
            File.WriteAllText(tmp, json + "\n", new UTF8Encoding(false));
            if (File.Exists(path)) File.Delete(path);
            File.Move(tmp, path);
        }

        public async Task<string> StartAsync(string modelPath)
        {
            string goldenDir = GoldenRunner.GoldenDir;
            sessionStamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            string dir = Path.Combine(Root, "sessions", sessionStamp);
            Directory.CreateDirectory(dir);
            Directory.CreateDirectory(Path.Combine(InboxDir, "processed"));
            Directory.CreateDirectory(OutboxDir);
            goldens = await Task.Run(() => GoldenRunner.Read(goldenDir));
            foreach (JNode row in goldens.Rows)
            {
                string sid = row["snapshot_id"].Text, text = goldens.Texts[row["scene_file"].Text];
                if (!Presets.ContainsKey(sid))
                {
                    JNode rec = JParse.Parse(text);
                    Scenes.Add(new Scene { SnapshotId = sid, SceneId = rec["scene_id"].Text, SceneRevision = rec["scene_revision"].Text,
                                           Sha256 = ReplayHash.Sha256Hex(Encoding.UTF8.GetBytes(text)), Text = text,
                                           Objects = rec["objects"].Items.Count, Record = rec });
                    Presets[sid] = new List<Preset>();
                }
                if (row["kind"].Text == "dataset_command")   // written fixtures are implementation checks only (D99(5))
                {
                    string cmd = goldens.Texts[row["command_file"].Text];
                    Presets[sid].Add(new Preset { Label = JParse.Parse(cmd)["text"].Text, CommandText = cmd });
                }
            }
            Scenes.Sort((a, b) => a.Objects.CompareTo(b.Objects));
            SetScene(0);
            Say("Session: loading the model...");
            long rssBefore = LlamaRuntime.MemoryKb(false), peakBefore = LlamaRuntime.MemoryKb(true);
            string captureError = null, capturePath = Path.Combine(dir, "startup-log.txt");
            runtime = new LlamaRuntime(main);
            double loadMs = await runtime.LoadCapturedAsync(modelPath, Pipeline.ContextTokens, ReplayRunner.Threads, ReplayRunner.Flags,
                                                            capturePath, e => captureError = captureError == null ? e : captureError + "; " + e);
            long rssAfter = LlamaRuntime.MemoryKb(false), peakAfter = LlamaRuntime.MemoryKb(true);
            StartupGuard.Result guard = await runtime.WithModel(s => StartupGuard.Run(goldens.Asset, goldens.Rows, goldens.Texts,
                                                                                     b => new LlamaAdapter(s).Tokenize(b)));
            Event("interactive.startup_check", guard.WriteTo(new JsonWriter().BeginObj().Key("session").S(sessionStamp)).End().ToString());
            if (!guard.Ok)   // D104: refuse to serve requests (the panel's failure path ends the session)
                throw new InvalidOperationException("the startup check failed (" + guard.FirstFailure + "); no request will be served");
            log = new StreamWriter(Path.Combine(dir, "outcomes.jsonl"), false, new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(dir, "session.json"), new JsonWriter().BeginObj().Key("record_type").S("a25_interactive_session")
                .Key("session").S(sessionStamp).Key("purpose").S("operational").Key("cache_mode").S("Off").Key("execution_path").S("U")
                .Key("golden_manifest_sha256").S(goldens.ManifestSha256).Key("prompt_asset_sha256").S(goldens.AssetSha256)
                .Key("model").S(modelPath).Key("context_tokens").I(Pipeline.ContextTokens).Key("flags").I(ReplayRunner.Flags)
                .Key("llama_cpp").S(runtime.Version).Key("load_ms").D(loadMs).Key("app_version").S(Application.version)
                .Key("memory_kb").BeginObj().Key("source").S("the wrapper's se_memory_kb: resident (VmRSS) and peak (VmHWM)")
                .Key("before_load_rss").I(rssBefore).Key("before_load_peak").I(peakBefore)
                .Key("after_load_rss").I(rssAfter).Key("after_load_peak").I(peakAfter).End()
                .Key("startup_log").BeginObj().Key("file").S("startup-log.txt").Key("verbose").S("on during the load only")
                .Key("capture_error").S(captureError).End()
                .Key("event_log").S(EventLog.FilePath).Key("event_session").S(EventLog.SessionId).Key("startup_check_ok").B(guard.Ok)
                .Key("goldens_equal").I(guard.GoldensEqual).Key("prompt_self_checks_passed").I(guard.SelfChecksPassed).End().ToString() + "\n",
                new UTF8Encoding(false));
            stop = new CancellationTokenSource();
            inboxStop = new CancellationTokenSource();
            CancellationToken ct = stop.Token, inboxCt = CancellationTokenSource.CreateLinkedTokenSource(ct, inboxStop.Token).Token;
            inboxTask = Task.Run(() => InboxLoop(inboxCt));     // both loops catch and log their own errors
            workerTask = Task.Run(() => WorkerLoop(ct));
            Event("interactive.session.start", new JsonWriter().BeginObj().Key("session").S(sessionStamp).Key("scenes").I(Scenes.Count)
                .Key("load_ms").D(loadMs).End().ToString());
            return "Session " + sessionStamp + " ready: " + Scenes.Count + " scenes, path U (uncached). Inbox: files/interactive/inbox.";
        }

        /// <summary>Binds the session to a scene. Every switch is a new epoch, so a result still running for the
        /// previous binding is answered as stale.</summary>
        public Scene SetScene(int index)
        {
            Scene s = Scenes[((index % Scenes.Count) + Scenes.Count) % Scenes.Count];
            epoch++;
            Current = new SceneBinding { SnapshotId = s.SnapshotId, SceneId = s.SceneId, SceneRevision = s.SceneRevision,
                                         SnapshotSha256 = s.Sha256, SceneText = s.Text, Epoch = epoch };
            Event("interactive.scene", new JsonWriter().BeginObj().Key("snapshot_id").S(s.SnapshotId).Key("objects").I(s.Objects)
                .Key("epoch").I(epoch).End().ToString());
            return s;
        }

        public string SubmitPreset(Preset p)
        {
            double pressed = Now;   // timing for presets starts at the button event (D99(3))
            SceneBinding b = Current;
            presetCount++;
            var t = new Ticket { RequestId = "preset-" + sessionStamp + "-" + presetCount.ToString("D3", CultureInfo.InvariantCulture),
                                 Source = "preset", SceneEpoch = b.Epoch, Session = sessionStamp };
            t.Times.Intake = "preset";
            t.Times.Detected = pressed;
            t.Request = new InteractiveRequest { RequestId = t.RequestId, Source = "preset", SnapshotId = b.SnapshotId,
                                                 SnapshotSha256 = b.SnapshotSha256, SceneText = b.SceneText, CommandText = p.CommandText };
            core.Claim(t.RequestId);
            t.Times.Validated = Now;
            if (!core.Admit(t))
            {
                t.Times.Outcome = Now;
                Publish(t, SessionCore.Answer(t.RequestId, t.Source, "cancelled", "session_ended", "the session was ending; not admitted"));
                return t.RequestId;
            }
            t.Times.Queued = Now;
            return t.RequestId;
        }

        /// <summary>Cancels the queued requests at once and flags the running one. Its evaluation cannot be
        /// interrupted, so its result will be answered as cancelled.</summary>
        public int Cancel() { return Cancel("cancelled_while_queued", "cancelled before evaluation"); }

        private int Cancel(string reason, string detail)
        {
            List<Ticket> gone = core.CancelAll();
            foreach (Ticket t in gone)
            {
                t.Times.Outcome = Now;
                Publish(t, SessionCore.Answer(t.RequestId, t.Source, "cancelled", reason, detail));
            }
            return gone.Count;
        }

        /// <summary>Ends the session so that every accepted request is answered and recorded before anything is closed
        /// (ChatGPT's A2.5 review), in this order:
        ///   1. admission closes; the queued requests are answered as cancelled (session_ended), and the running one is
        ///      flagged;
        ///   2. the inbox loop stops and is awaited: a handler already running finds admission closed and answers its own
        ///      request as session_ended;
        ///   3. the worker is awaited: it publishes the running request (answered as cancelled) and exits;
        ///   4. only then are the log closed and the model unloaded.</summary>
        public async Task<int> EndAsync()
        {
            List<Ticket> gone = core.Close();
            foreach (Ticket t in gone)
            {
                t.Times.Outcome = Now;
                Publish(t, SessionCore.Answer(t.RequestId, t.Source, "cancelled", "session_ended", "the session ended before this request was evaluated"));
            }
            if (inboxStop != null) inboxStop.Cancel();
            if (inboxTask != null) await Task.WhenAny(inboxTask, Task.Delay(10000));
            if (workerTask != null) await Task.WhenAny(workerTask, Task.Delay(120000));
            Dispose();
            return gone.Count;
        }

        private void Ack(string id, string state, string reason, double detected)
        {
            WriteAtomic(Path.Combine(OutboxDir, id + ".ack.json"), new JsonWriter().BeginObj().Key("record_type").S("a25_interactive_ack")
                .Key("request_id").S(id).Key("session").S(sessionStamp).Key("status").S(state).Key("reason").S(reason)
                .Key("detected_ms").D(detected).Key("acknowledged_ms").D(Now).End().ToString());
        }

        private async Task InboxLoop(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    string[] files = Directory.GetFiles(InboxDir, "*.json");
                    Array.Sort(files, string.CompareOrdinal);
                    foreach (string f in files) Handle(f);
                }
                catch (Exception e)
                {
                    Event("interactive.inbox.error", new JsonWriter().BeginObj().Key("error").S(e.GetType().Name + ": " + e.Message).End().ToString());
                }
                await Task.Delay(200);
            }
        }

        private void Handle(string file)
        {
            double detected = Now;
            string text = File.ReadAllText(file, Encoding.UTF8);
            string name = Path.GetFileName(file), kept = Path.Combine(InboxDir, "processed", name);
            for (int k = 2; File.Exists(kept); k++) kept = Path.Combine(InboxDir, "processed", Path.GetFileNameWithoutExtension(name) + "." + k + ".json");
            File.Move(file, kept);
            JNode req;
            string id, why = RequestFile.Validate(text, out req, out id);
            var t = new Ticket { RequestId = id, Source = "adb_inbox", Session = sessionStamp };
            t.Times.Intake = "adb_inbox";
            t.Times.Detected = detected;
            if (id == null || !System.Text.RegularExpressions.Regex.IsMatch(id, "^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"))
            {
                Event("interactive.inbox.unanswerable", new JsonWriter().BeginObj().Key("file").S(name).Key("reason").S(why).End().ToString());
                return;   // no usable request ID: nothing can be answered under it; the file is kept in processed/
            }
            if (!core.Claim(id))
            {
                WriteAtomic(Path.Combine(OutboxDir, id + ".duplicate-" + DateTime.UtcNow.ToString("HHmmssfff", CultureInfo.InvariantCulture) + ".json"),
                    new JsonWriter().BeginObj().Key("record_type").S("a25_interactive_duplicate").Key("request_id").S(id)
                    .Key("detail").S("this request ID was processed before in this session; the first acknowledgement and result stand").End().ToString());
                return;
            }
            t.Times.Validated = Now;
            SceneBinding b = Current;
            string field = why == null && b != null ? b.Mismatch(req["expected_scene"]) : null;
            if (why != null || b == null || field != null)
            {
                string reason = why != null ? "invalid_request" : b == null ? "no_scene_bound" : "stale_scene_binding";
                string detail = why ?? (field != null ? "expected_scene." + field + " differs from the current scene" : "");
                Ack(id, "rejected", reason, detected);
                t.Times.Outcome = Now;
                Publish(t, SessionCore.Answer(id, "adb_inbox", "refused", reason, detail));
                return;
            }
            t.SceneEpoch = b.Epoch;
            t.Request = new InteractiveRequest { RequestId = id, Source = "adb_inbox", SnapshotId = b.SnapshotId, SnapshotSha256 = b.SnapshotSha256,
                                                 SceneText = b.SceneText, CommandText = Canon.Enc(req["command"]) };
            if (!core.Admit(t))   // the session is ending: answered here, never queued behind the final drain
            {
                Ack(id, "rejected", "session_ended", detected);
                t.Times.Outcome = Now;
                Publish(t, SessionCore.Answer(id, "adb_inbox", "cancelled", "session_ended", "the session was ending; not admitted"));
                return;
            }
            t.Times.Queued = Now;
            Ack(id, "accepted", null, detected);
        }

        private async Task WorkerLoop(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                Ticket t = core.Next();
                if (t == null)
                {
                    if (core.Closed) return;   // admission closed and nothing left: the session is ending
                    await Task.Delay(50);
                    continue;
                }
                try
                {
                    await Serve(t);
                }
                finally
                {
                    core.Release(t);   // only after its result is published
                }
            }
        }

        private async Task Serve(Ticket t)
        {
            {
                InteractiveOutcome early = SessionCore.StaleAtStart(t, Current);
                if (early != null)
                {
                    core.Finish(t, early, Current);
                    t.Times.Outcome = Now;
                    Publish(t, early);
                    return;
                }
                t.Times.InferenceStart = Now;
                InteractiveOutcome o;
                try
                {
                    o = await runtime.WithModel(s => Pipeline.Run(t.Request, goldens.Asset, new LlamaAdapter(s), cache, ExecutionPurpose.Operational,
                                                                  new float[LlamaNative.se_n_vocab(s)], () => t.CancelRequested));
                }
                catch (Exception e)
                {
                    o = SessionCore.Answer(t.RequestId, t.Source, "failed", "runtime_error", e.GetType().Name + ": " + e.Message);
                }
                t.Times.InferenceEnd = Now;
                try
                {
                    o = core.Finish(t, o, Current);
                    t.Times.Outcome = Now;
                    Publish(t, o);
                }
                catch (Exception e)   // the worker never dies silently: the error is logged and the next ticket runs
                {
                    Event("interactive.worker.error", new JsonWriter().BeginObj().Key("request_id").S(t.RequestId)
                        .Key("error").S(e.GetType().Name + ": " + e.Message).End().ToString());
                }
            }
        }

        private void Publish(Ticket t, InteractiveOutcome o)
        {
            string line = "{\"record_type\":\"a25_interactive_result\",\"request_id\":" + Canon.Enc(JNode.Str(t.RequestId))
                          + ",\"session\":\"" + sessionStamp + "\",\"scene_epoch\":" + t.SceneEpoch
                          + ",\"snapshot_id\":" + (t.Request != null ? Canon.Enc(JNode.Str(t.Request.SnapshotId)) : "null")
                          + ",\"outcome\":" + o.ToJson() + ",\"times\":" + t.Times.ToJson()
                          + ",\"memory_kb\":{\"rss\":" + LlamaRuntime.MemoryKb(false) + ",\"peak\":" + LlamaRuntime.MemoryKb(true) + "}}";
            if (t.Source == "adb_inbox") WriteAtomic(Path.Combine(OutboxDir, t.RequestId + ".result.json"), line);
            lock (logGate)
            {
                if (log != null) { log.WriteLine(line); log.Flush(); }
                Answered++;
            }
            Event("interactive.result", new JsonWriter().BeginObj().Key("request_id").S(t.RequestId).Key("source").S(t.Source)
                .Key("status").S(o.Status).Key("reason").S(o.Reason).Key("execution_path").S(o.ExecutionPath)
                .Key("app_observed_ms").D(t.Times.AppObserved).End().ToString());
            main.Post(_ => onOutcome(t, o), null);
        }

        public void Dispose()
        {
            if (stop != null) stop.Cancel();
            if (runtime != null) runtime.Dispose();
            runtime = null;
            lock (logGate)
            {
                if (log != null) log.Dispose();
                log = null;
            }
        }
    }
}
