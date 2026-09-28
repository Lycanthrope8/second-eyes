using System;
using System.Text;
using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// Requests the target display refresh rate once at startup (D13: 72 Hz), so a change in the
    /// headset's default can't silently change our measurements. The request and the rates the
    /// headset offers go into the event log as display.rate; DebugOverlay shows the rate it runs at.
    /// </summary>
    public class DisplayRate : MonoBehaviour
    {
        [Tooltip("Refresh rate to request, in Hz (D13).")]
        [SerializeField] private float targetHz = 72f;

        private void Start()
        {
            float[] available = OVRPlugin.systemDisplayFrequenciesAvailable;
            bool known = available != null && available.Length > 0;
            string list = known ? string.Join(", ", available) : "unknown";

            if (known && Array.IndexOf(available, targetHz) < 0)
            {
                string message = $"{targetHz} Hz is not offered by this headset (available: {list})";
                Debug.LogWarning("[SecondEyes] DisplayRate: " + message);
                EventLog.Error("DisplayRate", message);
                return;
            }

            OVRPlugin.systemDisplayFrequency = targetHz;
            Debug.Log($"[SecondEyes] DisplayRate: requested {targetHz} Hz (available: {list}).");

            var data = new StringBuilder("{\"requested_hz\":");
            Json.AppendNumber(data, targetHz);
            data.Append(",\"available_hz\":[");
            if (known)
            {
                for (int i = 0; i < available.Length; i++)
                {
                    if (i > 0)
                    {
                        data.Append(',');
                    }
                    Json.AppendNumber(data, available[i]);
                }
            }
            data.Append("]}");
            EventLog.Write("display.rate", data.ToString());
        }
    }
}
