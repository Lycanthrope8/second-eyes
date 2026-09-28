using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// Requests the target display refresh rate once at startup (D13: 72 Hz), so a change
    /// in the headset's default can't silently change our measurements.
    /// DebugOverlay shows the rate the headset actually runs at.
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

            if (known && System.Array.IndexOf(available, targetHz) < 0)
            {
                Debug.LogWarning($"[SecondEyes] DisplayRate: {targetHz} Hz is not offered by this headset (available: {list}).");
                return;
            }

            OVRPlugin.systemDisplayFrequency = targetHz;
            Debug.Log($"[SecondEyes] DisplayRate: requested {targetHz} Hz (available: {list}).");
        }
    }
}
