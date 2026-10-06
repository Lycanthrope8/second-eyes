using System.Collections.Generic;
using System.Text;
using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// The app's own frame intervals, summarized once a second as a frame.second event (A1.10, D88): frames, how many
    /// took longer than 1.5 refresh intervals, the longest interval, the longest run of such frames, and when in the
    /// second each one ended. These are app-side frame times, measured from one Update to the next; they are not the
    /// compositor's stale frames, which OVR Metrics counts. The over-budget times also calibrate the clock between
    /// the two (analysis/detector_phases.py).
    /// </summary>
    public class FrameTimes : MonoBehaviour
    {
        [Tooltip("The display rate whose interval is the frame budget.")]
        [SerializeField] private float refreshHz = 72f;
        [Tooltip("A frame is over budget when its interval exceeds this many refresh intervals.")]
        [SerializeField] private float overFactor = 1.5f;

        private readonly List<int> overAt = new List<int>();
        private double last = -1, windowStart;
        private int frames, over, run, longestRun;
        private double maxMs;

        private void Start()
        {
            windowStart = Time.realtimeSinceStartupAsDouble;
        }

        private void Update()
        {
            double now = Time.realtimeSinceStartupAsDouble;
            double budgetMs = 1000.0 / refreshHz;
            if (last >= 0)
            {
                double ms = (now - last) * 1000.0;
                frames++;
                if (ms > maxMs) maxMs = ms;
                if (ms > overFactor * budgetMs)
                {
                    over++;
                    run++;
                    if (run > longestRun) longestRun = run;
                    overAt.Add((int)((now - windowStart) * 1000.0));
                }
                else
                {
                    run = 0;
                }
            }
            last = now;
            if (now - windowStart < 1.0) return;
            var sb = new StringBuilder("{\"window_ms\":").Append((int)((now - windowStart) * 1000.0))
                .Append(",\"frames\":").Append(frames).Append(",\"budget_ms\":");
            Json.AppendNumber(sb, (float)budgetMs);
            sb.Append(",\"over_budget\":").Append(over).Append(",\"max_ms\":");
            Json.AppendNumber(sb, (float)maxMs);
            sb.Append(",\"longest_over_run\":").Append(longestRun).Append(",\"over_at_ms\":[");
            for (int k = 0; k < overAt.Count; k++) sb.Append(k > 0 ? "," : "").Append(overAt[k]);
            EventLog.Write("frame.second", sb.Append("]}").ToString());
            windowStart = now;
            frames = over = longestRun = 0;
            maxMs = 0;
            overAt.Clear();
        }
    }
}
