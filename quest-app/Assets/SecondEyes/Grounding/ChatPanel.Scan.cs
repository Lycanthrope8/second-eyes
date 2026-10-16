using System.Collections.Generic;
using System.Linq;
using SecondEyes.Grounding.Interactive;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Perception.Scan;
using UnityEngine.UI;

namespace SecondEyes.Grounding
{
    /// <summary>
    /// The panel's scan (A4 Delivery 1, stage 3; docs/quest-panel.md, docs/scan.md). The Scan button starts the automatic scan
    /// (Glue/RoomScanner.cs, reached through ScanRegistry) and becomes Finish while it runs; the status line shows the
    /// scan's live hint. A finished scan's room joins the session as its current scene, and the result box lists its
    /// objects with how each was placed.
    /// </summary>
    public partial class ChatPanel
    {
        private Button scanButton;
        private Text scanLabel;
        private string scanShown;

        private void BuildScanButton(float x, float width)
        {
            scanButton = RowButton("Scan", x, width, out scanLabel, ToggleScan, SessionRowY, mainPage);
            scanLabel.text = "Scan";
        }

        private void ToggleScan()
        {
            IRoomScanner sc = ScanRegistry.Scanner;
            if (interactiveSession == null) { SetStatus("Start a session first: a scanned room joins the session"); return; }
            if (sc == null) { SetStatus("This build has no scanner"); return; }
            if (sc.Phase == "scanning") { sc.Finish(); return; }
            if (sc.Phase != "idle") return;
            if (perceptionFolder == null)
            {
                SetStatus("No perception folder: push it from the PC, then start the session again");
                return;
            }
            string why;
            if (!sc.TryStart(perceptionFolder, out why)) SetStatus("The scan can't start: " + why);
        }

        /// <summary>Each frame, from LateUpdate: the button's state, the scan's hint, and a finished scan's room.</summary>
        private void UpdateScan()
        {
            IRoomScanner sc = ScanRegistry.Scanner;
            if (scanButton != null) scanButton.interactable = interactiveSession != null && sc != null && (sc.Phase == "idle" || sc.Phase == "scanning");
            if (scanLabel != null) scanLabel.text = sc != null && sc.Phase == "scanning" ? "Finish" : sc == null || sc.Phase == "idle" ? "Scan" : "...";
            if (sc == null) return;
            if (sc.Phase != "idle" && sc.StatusLine != scanShown)
            {
                scanShown = sc.StatusLine;
                SetStatus(scanShown);
            }
            ScanResult r = sc.TakeResult();
            if (r != null) BindScan(r, sc.LastCaptureId);
        }

        private void BindScan(ScanResult r, string captureId)
        {
            scanShown = null;
            if (!r.Ok) { SetStatus("The scan didn't make a room: " + Why(r) + " (files/scans/" + captureId + ")"); return; }
            if (interactiveSession == null) { SetStatus("The scan finished after the session ended; its files are in files/scans/" + captureId); return; }
            sceneIndex = interactiveSession.AddScan(captureId, PyJson.Canonical(r.Scene), r.Scene);
            ShowScene(null, null);
            SetStatus("Scanned room: " + r.Scene["objects"].Items.Count + " objects. Type a command about them, such as the chair closest to the bed");
        }

        private static string Why(ScanResult r)
        {
            if (r.Error != null) return r.Error;
            if (r.CaptureIssues.Count > 0) return "the capture failed its check (" + r.CaptureIssues[0].Code + " " + r.CaptureIssues[0].Path + ")";
            var e = r.SceneIssues.FirstOrDefault(i => i.IsError);
            return e != null ? "the scene failed the contract (" + e.Code + " " + e.Path + ")" : "no scene";
        }

        private static bool Perceived(JNode scene) { return scene != null && scene["evidence_profile"].Text == "perceived"; }

        /// <summary>A scanned room's objects: each one's category and how it was placed, the target marked.</summary>
        private static IEnumerable<string> ScanLines(JNode scene, InteractiveOutcome o, bool present)
        {
            foreach (JNode obj in scene["objects"].Items)
            {
                string id = obj["object_id"].Text;
                JNode c = obj["geometry"]["center_m"], a = obj["observation"]["association"];
                string how = a["status"].Text == "possible_duplicate" ? "maybe the same as " + string.Join(", ", a["may_duplicate"].Items.Select(x => x.Text))
                           : c["state"].Text != "known" ? "position unknown"
                           : a["status"].Text == "multi_position" ? "placed from " + obj["observation"]["views"]["positions"].Text + " spots"
                           : "placed from its size";
                bool target = present && o != null && o.TargetObjectId == id;
                yield return id + "  " + obj["category"]["value"]["standard"].Text + "  " + how + (target ? "   <- target" : "");
            }
        }

        /// <summary>The command's own words, from its record; null when the record doesn't read.</summary>
        private static string CommandWords(Ticket t)
        {
            try { return t == null || t.Request == null ? null : JParse.Parse(t.Request.CommandText)["text"].Text; }
            catch (System.Exception) { return null; }
        }
    }
}
