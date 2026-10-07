using System.Collections.Generic;
using System.Text;
using SecondEyes.Logging;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace SecondEyes.App
{
    /// <summary>
    /// Per-frame GPU times from Unity's FrameTimingManager (D89), for profiling the detector's scheduling steps and for
    /// placing stale frames against inference stages. Every frame it captures timings and records the latest completed
    /// frame's GPU and CPU time with that frame's start timestamp, beside this frame's number and a Stopwatch timestamp;
    /// a gpu.second event carries one second of these. FrameTimingManager reports a frame a few frames late, so the
    /// analysis matches each timing to its frame by timestamp. Needs Player Settings > Frame Timing Stats; gpu.timing
    /// records whether it is enabled and the timer frequencies. GPU times of 0 mean the platform does not report them.
    /// </summary>
    public class GpuFrameTimes : MonoBehaviour
    {
        private readonly FrameTiming[] latest = new FrameTiming[1];
        private readonly List<int> frames = new List<int>();
        private readonly List<long> updateTicks = new List<long>();
        private readonly List<ulong> timingStart = new List<ulong>();
        private readonly List<double> gpuMs = new List<double>(), cpuMs = new List<double>();
        private double windowStart;

        private void Start()
        {
            windowStart = Time.realtimeSinceStartupAsDouble;
            var sb = new StringBuilder("{\"feature_enabled\":").Append(FrameTimingManager.IsFeatureEnabled() ? "true" : "false")
                .Append(",\"cpu_timer_frequency\":").Append(FrameTimingManager.GetCpuTimerFrequency())
                .Append(",\"gpu_timer_frequency\":").Append(FrameTimingManager.GetGpuTimerFrequency())
                .Append(",\"stopwatch_frequency\":").Append(Stopwatch.Frequency).Append('}');
            EventLog.Write("gpu.timing", sb.ToString());
        }

        private void Update()
        {
            FrameTimingManager.CaptureFrameTimings();
            uint n = FrameTimingManager.GetLatestTimings(1, latest);
            frames.Add(Time.frameCount);
            updateTicks.Add(Stopwatch.GetTimestamp());
            if (n > 0)
            {
                timingStart.Add(latest[0].frameStartTimestamp);
                gpuMs.Add(latest[0].gpuFrameTime);
                cpuMs.Add(latest[0].cpuFrameTime);
            }
            else
            {
                timingStart.Add(0);
                gpuMs.Add(-1);
                cpuMs.Add(-1);
            }
            double now = Time.realtimeSinceStartupAsDouble;
            if (now - windowStart < 1.0) return;
            var sb = new StringBuilder("{\"frames\":[");
            for (int k = 0; k < frames.Count; k++) sb.Append(k > 0 ? "," : "").Append(frames[k]);
            sb.Append("],\"update_ticks\":[");
            for (int k = 0; k < updateTicks.Count; k++) sb.Append(k > 0 ? "," : "").Append(updateTicks[k]);
            sb.Append("],\"timing_start\":[");
            for (int k = 0; k < timingStart.Count; k++) sb.Append(k > 0 ? "," : "").Append(timingStart[k]);
            sb.Append("],\"gpu_ms\":[");
            for (int k = 0; k < gpuMs.Count; k++)
            {
                if (k > 0) sb.Append(',');
                Json.AppendNumber(sb, (float)gpuMs[k]);
            }
            sb.Append("],\"cpu_ms\":[");
            for (int k = 0; k < cpuMs.Count; k++)
            {
                if (k > 0) sb.Append(',');
                Json.AppendNumber(sb, (float)cpuMs[k]);
            }
            EventLog.Write("gpu.second", sb.Append("]}").ToString());
            windowStart = now;
            frames.Clear();
            updateTicks.Clear();
            timingStart.Clear();
            gpuMs.Clear();
            cpuMs.Clear();
        }
    }
}
