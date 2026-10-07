using System.Text;
using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.Grounding
{
    /// <summary>
    /// A1.10d's fixed command schedule (D88, O21): the panel's prompt choices sent in a fixed order at fixed times, one
    /// every periodS seconds from the start. A command that comes due while the model is busy, or before it has loaded,
    /// is skipped and logged, never queued, so every condition sends the same commands at the same offsets. Started
    /// and stopped by the right controller's grip, or by the detector's sequential mode after each release.
    /// </summary>
    public class CommandSchedule : MonoBehaviour
    {
        [SerializeField] private ChatPanel panel;
        [Tooltip("Prompt choices to send, in order, then again from the first (0 is the fixed prompt, then the presets).")]
        [SerializeField] private int[] order = { 0, 1, 2, 3 };
        [SerializeField] private float periodS = 12f;

        private double next;
        private int seq;

        public bool Running { get; private set; }

        /// <summary>Whether a panel and an order are assigned, so a start can actually send (r046 ran without).</summary>
        public bool IsWired => panel != null && order != null && order.Length > 0;

        public void Toggle(string source)
        {
            if (Running) StopSchedule(source); else StartSchedule(source);
        }

        public void StartSchedule(string source)
        {
            if (Running) return;
            if (!IsWired)
            {
                EventLog.Error("commands", "the command schedule has no ChatPanel or no order assigned; nothing will be sent");
                return;
            }
            Running = true;
            seq = 0;
            next = Time.realtimeSinceStartupAsDouble;
            var sb = new StringBuilder("{\"state\":\"start\",\"source\":");
            Json.AppendString(sb, source);
            sb.Append(",\"period_s\":");
            Json.AppendNumber(sb, periodS);
            sb.Append(",\"order\":[");
            for (int k = 0; k < order.Length; k++) sb.Append(k > 0 ? "," : "").Append(order[k]);
            EventLog.Write("command.schedule", sb.Append("]}").ToString());
        }

        public void StopSchedule(string source)
        {
            if (!Running) return;
            Running = false;
            var sb = new StringBuilder("{\"state\":\"stop\",\"source\":");
            Json.AppendString(sb, source);
            sb.Append(",\"period_s\":");
            Json.AppendNumber(sb, periodS);
            sb.Append(",\"order\":[");
            for (int k = 0; k < order.Length; k++) sb.Append(k > 0 ? "," : "").Append(order[k]);
            EventLog.Write("command.schedule", sb.Append("]}").ToString());
        }

        private void Update()
        {
            if (!Running) return;
            double now = Time.realtimeSinceStartupAsDouble;
            if (now < next) return;
            int choice = order[seq % order.Length];
            double lateMs = (now - next) * 1000.0;
            string why = panel.TryDispatch(choice);
            var sb = new StringBuilder("{\"seq\":").Append(seq + 1).Append(",\"choice\":").Append(choice).Append(",\"prompt_id\":");
            string id = panel.ChoiceId(choice);
            if (id == null) sb.Append("null"); else Json.AppendString(sb, id);
            sb.Append(",\"dispatched\":").Append(why == null ? "true" : "false").Append(",\"reason\":");
            if (why == null) sb.Append("null"); else Json.AppendString(sb, why);
            sb.Append(",\"late_ms\":");
            Json.AppendNumber(sb, (float)lateMs);
            EventLog.Write("command.dispatch", sb.Append('}').ToString());
            seq++;
            next += periodS;
        }
    }
}
