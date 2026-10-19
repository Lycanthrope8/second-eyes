using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using SecondEyes.Grounding.Interactive;
using SecondEyes.Grounding.Prompting;
using UnityEngine;
using UnityEngine.UI;

namespace SecondEyes.Grounding
{
    // A2.5 delivery 4: the interactive session's row, on the panel's main page since A4 Delivery 1 (ChatPanel.Layout.cs).
    // Start session starts it and End session ends it (its own model load, path U). Scene cycles through the 3-, 6- and
    // 10-object snapshots. Preset sends the current scene's next dataset command. Cancel cancels queued requests and flags
    // the running one. ADB inbox requests arrive while the session runs, and Send submits a typed command
    // (ChatPanel.Prompt.cs). The result box shows the scene's objects with their letters, the target marked with an
    // arrow, and the last outcome.
    public partial class ChatPanel
    {
        private Perception.PerceptionFolder perceptionFolder;   // A4 (stage 3)
        private static string PerceptionDir => Path.Combine(Application.persistentDataPath, "perception");

        private Button sessionButton, sceneButton, sendPresetButton, cancelButton;
        private Text sessionLabel, sceneLabel, sendPresetLabel, cancelLabel;
        private InteractiveService interactiveSession;
        private bool sessionStarting, loadWasBeforeSession;
        private int sceneIndex, presetIndex;

        private void BuildSessionRow()
        {
            sessionButton = RowButton("Session", 20f, 112f, out sessionLabel, ToggleSession, SessionRowY, mainPage);
            sceneButton = RowButton("Scene", 142f, 112f, out sceneLabel, NextScene, SessionRowY, mainPage);
            sendPresetButton = RowButton("SendPreset", 264f, 112f, out sendPresetLabel, SendPreset, SessionRowY, mainPage);
            BuildScanButton(386f, 112f);   // A4 (stage 3): ChatPanel.Scan.cs
            cancelButton = RowButton("CancelRequests", 508f, 112f, out cancelLabel, CancelRequests, SessionRowY, mainPage);
            sceneLabel.text = "Scene";
            sendPresetLabel.text = "Preset";
            cancelLabel.text = "Cancel";
            UpdateSessionButtons();
        }

        private void UpdateSessionButtons()
        {
            bool on = interactiveSession != null && !sessionStarting;
            if (sessionButton != null)
                sessionButton.interactable = !sessionStarting && (on || (!replaying && !panelModelUsed && GoldenRunner.GoldensPresent));
            if (sceneButton != null) sceneButton.interactable = on;
            if (sendPresetButton != null) sendPresetButton.interactable = on;
            if (cancelButton != null) cancelButton.interactable = on;
            if (sendTypedButton != null) sendTypedButton.interactable = on;   // the typed command (ChatPanel.Prompt.cs)
            if (sessionLabel != null) sessionLabel.text = on ? "End session" : "Start session";
        }

        private async void ToggleSession()
        {
            if (sessionStarting) return;   // O28: a start is already under way
            if (interactiveSession != null)
            {
                EndSession();
                return;
            }
            // O28: what needs Unity is read here, on the main thread; the start-up state is set and the controls disabled
            // before anything is awaited; the folders' integrity checks, unchanged, then run off the main thread.
            string dataDir = Application.persistentDataPath;
            string path = Path.Combine(dataDir, ggufFile);
            string perceptionDir = PerceptionDir;
            bool perceptionPresent = Perception.PerceptionFolder.Present(perceptionDir);   // A4 (stage 3): scanned rooms join the session
            bool parserPresent = Parsing.ParserCheckRunner.FolderPresent;   // A2.7 (D115): with a parser folder, the session grounds through the parser
            string parserDir = Parsing.ParserCheckRunner.FolderDir;
            perceptionFolder = null;
            replaying = true;
            sessionStarting = true;
            loadWasBeforeSession = loadButton.interactable;
            loadButton.interactable = false;
            UpdateReplayButton();
            UpdateSessionButtons();
            SetStatus("Checking the folders before the session starts...");
            Parsing.ParserFolder parserFolder = null;
            Perception.PerceptionFolder loaded = null;
            string failed = await System.Threading.Tasks.Task.Run(() =>
            {
                if (perceptionPresent)
                {
                    try { loaded = Perception.PerceptionFolder.Load(perceptionDir); }
                    catch (Exception e) { return "The perception folder failed its check (" + e.Message + "); no session started. Push it again."; }
                }
                if (parserPresent)
                {
                    try
                    {
                        parserFolder = loaded == null ? Parsing.ParserFolder.Load(parserDir)
                            : Parsing.ParserFolder.Load(parserDir, loaded.SchemaText, loaded.Maps);
                    }
                    catch (Exception e) { return "The parser folder failed its check (" + e.Message + "); no session started. Push it again."; }
                }
                return null;
            });
            perceptionFolder = loaded;
            if (parserFolder != null) path = Path.Combine(dataDir, parserFolder.GgufFile);
            if (failed == null && !File.Exists(path))
                failed = parserFolder != null ? "The parser model " + parserFolder.GgufFile + " isn't on the headset. Push it."
                                              : "llama.cpp's model isn't on the headset. Push it: python grounding/llama_headset.py push-model";
            if (failed != null)
            {
                SetStatus(failed);
                replaying = false;
                sessionStarting = false;
                loadButton.interactable = loadWasBeforeSession;
                UpdateReplayButton();
                UpdateSessionButtons();
                return;
            }
            interactiveSession = new InteractiveService(System.Threading.SynchronizationContext.Current, SetStatus, ShowOutcome);
            UpdateReplayButton();
            UpdateSessionButtons();
            try
            {
                string ready = await interactiveSession.StartAsync(path, parserFolder);
                sessionStarting = false;
                ShowScene(null, null);
                SetStatus(ready);
            }
            catch (Exception e)
            {
                sessionStarting = false;
                Fail("interactiveSession", e.GetType().Name + ": " + e.Message);
                EndSession();
            }
            UpdateSessionButtons();
        }

        private async void EndSession()
        {
            SecondEyes.Perception.Scan.ScanRegistry.Holds = 0; SecondEyes.Perception.Scan.ScanRegistry.HoldUntil = 0;   // no command is waiting any more
            InteractiveService s = interactiveSession;
            ShutdownReport r = null;
            if (s != null)
            {
                sessionStarting = true;   // keeps the row disabled until every task has completed
                UpdateSessionButtons();
                SetStatus("Ending the session: answering waiting requests, letting the running one finish...");
                try
                {
                    r = await s.EndAsync(msg => SetStatus("Ending the session: " + msg + "..."));
                }
                catch (Exception e)   // EndAsync is built never to throw; if it does, the state is unknown: stay locked
                {
                    Fail("session", e.GetType().Name + ": " + e.Message);
                    SetStatus("Ending the session failed unexpectedly; the session row stays locked. Restart the app.");
                    return;   // no Dispose here, no reset: another model load must not start while work may be active
                }
            }
            interactiveSession = null;
            sessionStarting = false;
            replaying = false;
            loadButton.interactable = loadWasBeforeSession;
            resultText.text = NoSessionHint;
            UpdateReplayButton();
            UpdateSessionButtons();
            if (r == null || r.Clean)
                SetStatus("Session ended" + (r != null && r.CancelledQueued > 0 ? "; " + r.CancelledQueued + " waiting request(s) answered as cancelled." : "."));
            else
                SetStatus("Session ended with a fault: " + string.Join("; ", r.Faults.ToArray()));
        }

        private void NextScene()
        {
            if (interactiveSession == null) return;
            sceneIndex++;
            presetIndex = 0;
            InteractiveService.Scene s = interactiveSession.SetScene(sceneIndex);
            sendPresetLabel.text = "Preset";
            ShowScene(null, null);
            SetStatus("Scene: " + s.Objects + " objects (" + Short(s.SnapshotId) + "); " + (interactiveSession.Presets.ContainsKey(s.SnapshotId)
                      ? interactiveSession.Presets[s.SnapshotId].Count + " presets" : "a scanned room: type a command"));
        }

        private void SendPreset()
        {
            if (interactiveSession != null && interactiveSession.Current != null && !interactiveSession.Presets.ContainsKey(interactiveSession.Current.SnapshotId))
            {
                SetStatus("A scanned room has no presets: type a command about its objects");   // A4 (stage 3)
                return;
            }
            if (interactiveSession == null || interactiveSession.Current == null) return;
            List<InteractiveService.Preset> list = interactiveSession.Presets[interactiveSession.Current.SnapshotId];
            if (list.Count == 0)
            {
                SetStatus("This scene has no dataset presets.");
                return;
            }
            InteractiveService.Preset p = list[presetIndex % list.Count];
            presetIndex++;
            string id = interactiveSession.SubmitPreset(p);
            sendPresetLabel.text = "Preset " + (presetIndex % list.Count + 1) + "/" + list.Count;
            SetStatus("Sent " + id + ": \"" + p.Label + "\" (" + interactiveSession.Waiting + " waiting)");
        }

        private void CancelRequests()
        {
            if (interactiveSession == null) return;
            int n = interactiveSession.Cancel();
            SetStatus("Cancel: " + n + " queued request(s) cancelled; a running evaluation finishes and is answered as cancelled.");
        }

        private void ShowOutcome(Ticket t, InteractiveOutcome o)
        {
            AfterCommand();
            InteractiveService s = interactiveSession;
            Presentability p = s == null || s.Current == null ? Presentability.OtherSession
                             : Presentation.Decide(s.SessionStamp, s.Current.Epoch, t);
            if (p == Presentability.OtherSession) return;   // an ended session's late result: logged, never shown
            if (p == Presentability.StaleScene)
            {
                ShowScene(null, null);   // the current scene, with no target marked
                SetStatus(t.RequestId + ": answered for an earlier scene binding; not presented (" + o.Status + ", logged)");
                return;
            }
            string what = o.Path == "parser"   // A2.7 (D115): the parser's kinds stay distinct on the panel too
                        ? (o.Status == "completed" ? "target " + o.TargetObjectId
                           : o.Status == "ask" ? "ASK (" + o.Reason + ")"
                           : o.Status + " (" + (o.ParserKind ?? "?") + "): " + o.Reason)
                        : o.Status == "completed" ? "target " + o.TargetObjectId + " (" + o.ChoiceCode + ")"
                        : o.Status == "ask" ? (o.AskBasis == "exact_tie" ? "ASK (exact tie)" : "ASK (model-selected)")
                        : o.Status + ": " + o.Reason;
            SetStatus(t.RequestId + ": " + what + "; " + Seconds(t.Times.AppObserved) + " app-observed, path " + o.ExecutionPath);
            ShowScene(t, o);
        }

        private static string Seconds(double ms)
        {
            return double.IsNaN(ms) ? "?" : (ms / 1000.0).ToString("0.0", CultureInfo.InvariantCulture) + " s";
        }

        private static string Short(string id) { return id.Length > 8 ? "..." + id.Substring(id.Length - 8) : id; }

        private static string Category(JNode scene, string objectId)
        {
            foreach (JNode obj in scene["objects"].Items)
                if (obj["object_id"].Text == objectId && obj.Has("category") && obj["category"].Has("value"))
                    return "  " + obj["category"]["value"].Text;
            return "";
        }

        private void ShowScene(Ticket t, InteractiveOutcome o)
        {
            InteractiveService svc = interactiveSession;
            if (svc == null || svc.Current == null || svc.Asset == null) return;
            InteractiveService.Scene s = svc.Scenes.Find(x => x.SnapshotId == svc.Current.SnapshotId);
            bool present = t != null && o != null && Presentation.Decide(svc.SessionStamp, svc.Current.Epoch, t) == Presentability.Present;
            var sb = new StringBuilder();
            sb.Append("Scene: ").Append(s.Objects).Append(" objects (").Append(Short(s.SnapshotId)).Append(") rev ").Append(s.SceneRevision).Append('\n');
            IEnumerable<string> lines = Perceived(s.Record) ? ScanLines(s.Record, o, present)   // A4: a scanned room (ChatPanel.Scan.cs)
                : Presentation.SceneLines(s.Record, svc.Asset, o, present, id => Category(s.Record, id));
            foreach (string line in lines)
                sb.Append(line).Append('\n');   // always the current scene's own mapping (D104)
            if (present)
            {
                sb.Append("Last: ").Append(t.RequestId).Append(", ").Append(o.Status).Append(o.Reason != null ? " (" + o.Reason + ")" : "")
                  .Append(", ").Append(Seconds(t.Times.AppObserved));
                string words = CommandWords(t);   // A4: the whole command, and how the parser read it
                if (words != null) sb.Append("\nYou: ").Append(words);
                if (!string.IsNullOrEmpty(o.DecodedText)) sb.Append("\nRead as: ").Append(o.DecodedText);
            }
            resultText.text = sb.ToString();
        }
    }
}
