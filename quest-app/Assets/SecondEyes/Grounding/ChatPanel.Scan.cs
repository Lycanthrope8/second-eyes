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
        private object scanBoundTo;   // the session the scan's first snapshot was bound to
        private const double CommandHoldS = 60;   // the scan waits at most this long for a command's outcome

        private void BuildScanButton(float x, float width)
        {
            scanButton = RowButton("Scan", x, width, out scanLabel, ToggleScan, SessionRowY, mainPage);
            scanLabel.text = "Scan";
        }

        private void ToggleScan()
        {
            IRoomScanner sc = ScanRegistry.Scanner;
            if (sc == null) { SetStatus("This build has no scanner"); return; }
            if (sc.Phase == "scanning") { sc.Finish(); return; }   // the automatic scan: pause
            if (sc.Phase == "paused" || sc.Phase == "failed")
            {
                string w;
                if (!sc.TryStart(perceptionFolder, out w)) SetStatus("The scan can't go on: " + w);
                return;
            }
            if (sc.Phase != "idle") return;
            if (interactiveSession == null) { SetStatus("Start a session first: a scanned room joins the session"); return; }
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
            if (scanButton != null) scanButton.interactable = sc != null && (sc.Phase == "idle" || sc.Phase == "scanning" || sc.Phase == "paused" || sc.Phase == "failed");
            if (scanLabel != null) scanLabel.text = sc == null || sc.Phase == "idle" || sc.Phase == "failed" ? "Scan" : sc.Phase == "scanning" ? "Pause"
                                                  : sc.Phase == "paused" ? "Resume" : "...";
            if (sc == null) return;
            if (sc.Phase != "idle" && sc.StatusLine != scanShown)
            {
                scanShown = sc.StatusLine;
                SetStatus(scanShown);
            }
            // A snapshot joins the session only with a session, and not while a command is being processed.
            if (interactiveSession == null || UnityEngine.Time.realtimeSinceStartupAsDouble < ScanRegistry.HoldUntil) return;
            ScanResult r = sc.TakeResult();
            if (r != null) BindScan(r, sc.LastCaptureId);
        }

        /// <summary>A4 (the automatic scan): before a command is sent, the newest snapshot joins the session, so the
        /// command sees everything found so far; then the scan holds until the command's outcome is shown.</summary>
        private void BeforeCommand()
        {
            IRoomScanner sc = ScanRegistry.Scanner;
            if (sc != null && interactiveSession != null)
            {
                ScanResult r = sc.TakeResult();
                if (r != null) BindScan(r, sc.LastCaptureId);
            }
            // A4: once a room is scanned, a typed command is about it: back to the room if a dataset scene is showing
            // (the Scene button and presets are for the dataset scenes).
            if (interactiveSession != null && interactiveSession.Current != null)
            {
                int room = interactiveSession.Scenes.FindIndex(x => Perceived(x.Record));
                InteractiveService.Scene current = interactiveSession.Scenes.Find(x => x.SnapshotId == interactiveSession.Current.SnapshotId);
                if (room >= 0 && (current == null || !Perceived(current.Record)))
                {
                    sceneIndex = room;
                    interactiveSession.SetScene(room);
                    ShowScene(null, null);
                    SetStatus("Typed commands go to your scanned room");
                }
            }
            ScanRegistry.HoldUntil = UnityEngine.Time.realtimeSinceStartupAsDouble + CommandHoldS;
        }

        /// <summary>The command's outcome is shown: the scan may go on.</summary>
        private static void AfterCommand() { ScanRegistry.HoldUntil = 0; }

        private void BindScan(ScanResult r, string captureId)
        {
            scanShown = null;
            string folderName = r.Scene != null && r.Scene.Has("scene_id") ? r.Scene["scene_id"].Text : captureId;
            if (!r.Ok) { SetStatus("The scan didn't make a room: " + Why(r) + " (files/scans/" + folderName + ")"); return; }
            if (interactiveSession == null) { SetStatus("The scan finished after the session ended; its files are in files/scans/" + folderName); return; }
            bool first = !ReferenceEquals(scanBoundTo, interactiveSession);
            sceneIndex = interactiveSession.AddScan(captureId, PyJson.Canonical(r.Scene), r.Scene);
            scanBoundTo = interactiveSession;
            if (!first) return;   // later snapshots replace it quietly, so a shown outcome stays on screen
            ShowScene(null, null);
            SetStatus("Scanned room: " + r.Scene["objects"].Items.Count + " objects so far. Type a command about them, such as the chair closest to the bed");
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
                JNode c = obj["geometry"]["center_m"], a = obj["observation"]["association"], v = obj["observation"]["views"];
                bool depth = c["state"].Text == "known" && c["evidence"]["assumptions"].Items.Any(x => x.Text == "depth_surface");
                JNode size = obj["geometry"]["size_m"];
                bool boxed = c["state"].Text == "known" && size["state"].Text == "known"
                             && c["evidence"]["assumptions"].Items.Any(x => x.Text == "box_centre" || x.Text == "meta_volume");
                string how = a["status"].Text == "possible_duplicate" ? "maybe the same as " + string.Join(", ", a["may_duplicate"].Items.Select(x => x.Text))
                           : c["state"].Text != "known" ? "position unknown"
                           : boxed ? (c["evidence"]["assumptions"].Items.Any(x => x.Text == "meta_volume") ? "Meta's box " : "box ")
                                     + string.Join(" x ", size["value"].Items.Select(x => x.Num.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture))) + " m, seen "
                                     + v["keyframes"].Text + (v["keyframes"].Num == 1 ? " time" : " times")
                           : depth ? "placed by depth, seen " + v["keyframes"].Text + " times" + (v["positions"].Num >= 2 ? " from " + v["positions"].Text + " places" : "")
                           : a["status"].Text == "multi_position" ? "placed from " + obj["observation"]["views"]["positions"].Text + " spots"
                           : "placed from its size";
                bool target = present && o != null && o.TargetObjectId == id;
                string name = obj["category"]["state"].Text == "known" ? obj["category"]["value"]["standard"].Text : "?";   // an unnamed box
                yield return id + "  " + name + "  " + how + (target ? "   <- target" : "");
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
