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
    // A2.5 delivery 4: the interactive interactiveSession's row on the panel. Session starts and ends it (its own model load,
    // path U). Scene cycles through the 3-, 6- and 10-object snapshots. Preset sends the current scene's next dataset
    // command. Cancel cancels queued requests and flags the running one. ADB inbox requests arrive while the interactiveSession
    // runs. The text box shows the scene's objects with their letters, the target marked with an arrow, and the last
    // outcome.
    public partial class ChatPanel
    {
        private Button sessionButton, sceneButton, sendPresetButton, cancelButton;
        private Text sessionLabel, sceneLabel, sendPresetLabel, cancelLabel;
        private InteractiveService interactiveSession;
        private bool sessionStarting, loadWasBeforeSession;
        private int sceneIndex, presetIndex, boxFontBefore;

        private void BuildSessionRow()
        {
            sessionButton = RowButton("Session", 20f, 140f, out sessionLabel, ToggleSession, 712f);
            sceneButton = RowButton("Scene", 173f, 140f, out sceneLabel, NextScene, 712f);
            sendPresetButton = RowButton("SendPreset", 326f, 140f, out sendPresetLabel, SendPreset, 712f);
            cancelButton = RowButton("CancelRequests", 479f, 141f, out cancelLabel, CancelRequests, 712f);
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
            if (sessionLabel != null) sessionLabel.text = on ? "End interactiveSession" : "Session (A2.5 d4)";
        }

        private async void ToggleSession()
        {
            if (interactiveSession != null)
            {
                if (!sessionStarting) EndSession();
                return;
            }
            string path = Path.Combine(Application.persistentDataPath, ggufFile);
            if (!File.Exists(path))
            {
                SetStatus("llama.cpp's model isn't on the headset. Push it: python grounding/llama_headset.py push-model");
                return;
            }
            replaying = true;
            sessionStarting = true;
            loadWasBeforeSession = loadButton.interactable;
            loadButton.interactable = false;
            boxFontBefore = boxText.fontSize;
            interactiveSession = new InteractiveService(System.Threading.SynchronizationContext.Current, SetStatus, ShowOutcome);
            UpdateReplayButton();
            UpdateSessionButtons();
            try
            {
                string ready = await interactiveSession.StartAsync(path);
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
            InteractiveService s = interactiveSession;
            int queued = 0;
            if (s != null)
            {
                sessionStarting = true;   // keeps the row disabled while the running request finishes
                UpdateSessionButtons();
                SetStatus("Ending the session: answering waiting requests, letting the running one finish...");
                try { queued = await s.EndAsync(); }
                catch (Exception e) { Fail("session", e.GetType().Name + ": " + e.Message); s.Dispose(); }
            }
            interactiveSession = null;
            sessionStarting = false;
            replaying = false;
            loadButton.interactable = loadWasBeforeSession;
            boxText.fontSize = boxFontBefore > 0 ? boxFontBefore : boxText.fontSize;
            boxText.text = text;   // the chat prompt the box showed before the session
            UpdateReplayButton();
            UpdateSessionButtons();
            SetStatus("Session ended" + (queued > 0 ? "; " + queued + " waiting request(s) answered as cancelled." : "."));
        }

        private void NextScene()
        {
            if (interactiveSession == null) return;
            sceneIndex++;
            presetIndex = 0;
            InteractiveService.Scene s = interactiveSession.SetScene(sceneIndex);
            sendPresetLabel.text = "Preset";
            ShowScene(null, null);
            SetStatus("Scene: " + s.Objects + " objects (" + Short(s.SnapshotId) + "); " + interactiveSession.Presets[s.SnapshotId].Count + " presets");
        }

        private void SendPreset()
        {
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
            string what = o.Status == "completed" ? "target " + o.TargetObjectId + " (" + o.ChoiceCode + ")"
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
            foreach (string line in Presentation.SceneLines(s.Record, svc.Asset, o, present, id => Category(s.Record, id)))
                sb.Append(line).Append('\n');   // always the current scene's own mapping (D104)
            if (present)
                sb.Append("Last: ").Append(t.RequestId).Append(", ").Append(o.Status).Append(o.Reason != null ? " (" + o.Reason + ")" : "")
                  .Append(", ").Append(Seconds(t.Times.AppObserved));
            boxText.fontSize = 13;
            boxText.text = sb.ToString();
        }
    }
}
