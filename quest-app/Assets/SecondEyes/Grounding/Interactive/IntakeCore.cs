using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Replay;

namespace SecondEyes.Grounding.Interactive
{
    // A2.5 delivery 4 (D98, D99(1)-(3)): the intake side, free of Unity so it is tested on a PC. ADB inbox files and
    // on-panel presets become Tickets in one session, which feeds delivery 3's pipeline (operational, so path U only).
    // Each request ID is processed once, and answered with an acknowledgement and then a result. A request bound to
    // another scene than the current one is rejected. A result that finishes after the scene changed is rejected as
    // stale. Cancel stops queued requests at once. A native evaluation in progress cannot be interrupted without a
    // native change, so its result is marked cancelled and its target discarded.

    /// <summary>The scene the session currently answers for: a registered snapshot and an epoch that changes on every
    /// switch, so results computed for an earlier binding can be recognized.</summary>
    public sealed class SceneBinding
    {
        public string SnapshotId, SceneId, SceneRevision, SnapshotSha256, SceneText;
        public int Epoch;

        /// <summary>Null when the request's expected scene is this one; otherwise the first field that differs.</summary>
        public string Mismatch(JNode expected)
        {
            if (expected["snapshot_id"].Text != SnapshotId) return "snapshot_id";
            if (expected["scene_id"].Text != SceneId) return "scene_id";
            if (expected["scene_revision"].Text != SceneRevision) return "scene_revision";
            if (expected["snapshot_sha256"].Text != SnapshotSha256) return "snapshot_sha256";
            return null;
        }
    }

    public static class RequestFile
    {
        public const string RecordType = "a25_interactive_request";
        private static readonly Regex Id = new Regex("^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$");

        /// <summary>Null when the request is well formed (schema a25_interactive_request, format 1); otherwise why not.
        /// The request ID is returned whenever it can be read, so even a rejection is answered under it.</summary>
        public static string Validate(string text, out JNode request, out string requestId)
        {
            request = null;
            requestId = null;
            try
            {
                request = JParse.Parse(text);
                if (request.Kind != JKind.Object) return "not a JSON object";
                if (request.Has("request_id") && request["request_id"].Kind == JKind.String) requestId = request["request_id"].Text;
                if (!request.Has("record_type") || request["record_type"].Text != RecordType) return "record_type is not " + RecordType;
                if (!request.Has("format_version") || request["format_version"].Text != "1") return "format_version is not 1";
                if (requestId == null || !Id.IsMatch(requestId)) return "request_id must match " + Id;
                if (!request.Has("expected_scene") || request["expected_scene"].Kind != JKind.Object) return "expected_scene is missing";
                JNode e = request["expected_scene"];
                foreach (string k in new[] { "snapshot_id", "scene_id", "snapshot_sha256" })
                    if (!e.Has(k) || e[k].Kind != JKind.String) return "expected_scene." + k + " must be a string";
                if (!e.Has("scene_revision") || e["scene_revision"].Kind != JKind.Number) return "expected_scene.scene_revision must be a number";
                if (!request.Has("command") || request["command"].Kind != JKind.Object) return "command is missing";
                JNode c = request["command"];
                if (!c.Has("text") || c["text"].Kind != JKind.String || c["text"].Text.Trim().Length == 0) return "command.text must be non-empty text";
                if (!c.Has("scene_id") || c["scene_id"].Text != e["scene_id"].Text || !c.Has("scene_revision")
                    || c["scene_revision"].Text != e["scene_revision"].Text)
                    return "command's scene identity differs from expected_scene";
                return null;
            }
            catch (FormatException ex)
            {
                return "not valid JSON: " + ex.Message;
            }
        }
    }

    /// <summary>The stages D99(3) logs, in milliseconds on one headset clock (the session's stopwatch). The intake
    /// method says where the clock starts: the file's detection (adb_inbox) or the button event (preset). ADB delivery
    /// is never inferred by subtracting the PC's clock.</summary>
    public sealed class StageTimes
    {
        public string Intake;
        public double Detected = double.NaN, Validated = double.NaN, Queued = double.NaN, InferenceStart = double.NaN,
                      InferenceEnd = double.NaN, Outcome = double.NaN;

        public double AppObserved { get { return Outcome - Detected; } }

        public string ToJson()
        {
            return new JsonWriter().BeginObj().Key("intake").S(Intake).Key("clock").S("headset session stopwatch, ms")
                .Key("detected").D(Detected).Key("validated").D(Validated).Key("queued").D(Queued)
                .Key("inference_start").D(InferenceStart).Key("inference_end").D(InferenceEnd).Key("outcome").D(Outcome)
                .Key("app_observed_ms").D(AppObserved).End().ToString();
        }
    }

    public sealed class Ticket
    {
        public string RequestId, Source;
        public InteractiveRequest Request;
        public int SceneEpoch;
        public StageTimes Times = new StageTimes();
        public volatile bool CancelRequested;
        public string State = "queued";   // queued, running, done
    }

    /// <summary>The session's request bookkeeping: the processed-once ledger, the queue, cancellation and the
    /// stale-result rule. One worker takes tickets in order.</summary>
    public sealed class SessionCore
    {
        private readonly HashSet<string> ledger = new HashSet<string>();
        private readonly LinkedList<Ticket> queue = new LinkedList<Ticket>();
        private readonly object gate = new object();
        public Ticket Running { get; private set; }

        public int Processed { get { lock (gate) return ledger.Count; } }
        public int Waiting { get { lock (gate) return queue.Count; } }

        /// <summary>Claims the request ID for good. False when it was seen before in this session (a duplicate is
        /// never processed again).</summary>
        public bool Claim(string requestId) { lock (gate) return ledger.Add(requestId); }

        public void Enqueue(Ticket t) { lock (gate) queue.AddLast(t); }

        public Ticket Next()
        {
            lock (gate)
            {
                if (queue.Count == 0) return null;
                Ticket t = queue.First.Value;
                queue.RemoveFirst();
                t.State = "running";
                Running = t;
                return t;
            }
        }

        /// <summary>Cancels every queued ticket (returned, to be answered as cancelled) and flags the running one.</summary>
        public List<Ticket> CancelAll()
        {
            lock (gate)
            {
                var cancelled = new List<Ticket>(queue);
                queue.Clear();
                foreach (Ticket t in cancelled) { t.CancelRequested = true; t.State = "done"; }
                if (Running != null) Running.CancelRequested = true;
                return cancelled;
            }
        }

        /// <summary>The outcome as it may be shown. A cancel requested during evaluation, or a scene switched while
        /// the request ran, discards the target and says why. The scores are kept as evidence.</summary>
        public InteractiveOutcome Finish(Ticket t, InteractiveOutcome o, SceneBinding current)
        {
            lock (gate)
            {
                if (Running == t) Running = null;
                t.State = "done";
            }
            if (o.Status != "completed" && o.Status != "ask") return o;   // failures and refusals keep their own reasons
            if (t.CancelRequested)
                Discard(o, "cancelled", "cancelled_during_evaluation", "a cancel was requested while the model evaluated; the result is not used");
            else if (current == null || current.Epoch != t.SceneEpoch)
                Discard(o, "refused", "stale_result_scene_changed", "the current scene changed while the request ran; the result is not used");
            return o;
        }

        /// <summary>A ticket whose scene binding changed while it waited is answered at once, without evaluation (r018
        /// showed the cost of evaluating it first). Null when it is still current.</summary>
        public static InteractiveOutcome StaleAtStart(Ticket t, SceneBinding current)
        {
            if (current != null && current.Epoch == t.SceneEpoch) return null;
            return Answer(t.RequestId, t.Source, "refused", "scene_changed_before_evaluation",
                          "the current scene changed while the request waited; it was not evaluated");
        }

        private static void Discard(InteractiveOutcome o, string status, string reason, string detail)
        {
            o.Detail = detail + " (it was " + o.Status + (o.ChoiceCode != null ? ", choice " + o.ChoiceCode : "") + ")";
            o.Status = status;
            o.Reason = reason;
            o.TargetObjectId = null;
            o.AskBasis = null;
        }

        /// <summary>An outcome for a request answered without evaluation (invalid, duplicate, stale binding, cancelled
        /// while queued).</summary>
        public static InteractiveOutcome Answer(string requestId, string source, string status, string reason, string detail)
        {
            return new InteractiveOutcome { RequestId = requestId, Source = source, Status = status, Reason = reason,
                                            Detail = detail ?? "", CacheMode = CacheMode.Off.ToString(), Purpose = "operational" };
        }
    }
}
