using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// Watches whether each hand is tracked. The first state and every change after it are logged
    /// as hands.state, and DebugOverlay shows the current state. No hand models are drawn: you see
    /// your real hands through passthrough (D21).
    /// Hand tracking must be allowed in the app (OVRManager: Hand Tracking Support) and on the headset.
    /// </summary>
    public class HandState : MonoBehaviour
    {
        /// <summary>True while a HandState is active in the scene.</summary>
        public static bool Present { get; private set; }

        public static bool LeftTracked { get; private set; }
        public static bool RightTracked { get; private set; }

        // Reused every frame, so reading the hands allocates nothing after the first call.
        private OVRPlugin.HandState left = new OVRPlugin.HandState();
        private OVRPlugin.HandState right = new OVRPlugin.HandState();
        private bool logged;

        private void OnEnable()
        {
            Present = true;
        }

        private void OnDisable()
        {
            Present = false;
        }

        private void Update()
        {
            bool leftNow = IsTracked(OVRPlugin.Hand.HandLeft, ref left);
            bool rightNow = IsTracked(OVRPlugin.Hand.HandRight, ref right);
            if (logged && leftNow == LeftTracked && rightNow == RightTracked)
            {
                return;
            }

            LeftTracked = leftNow;
            RightTracked = rightNow;
            logged = true;
            EventLog.Write("hands.state",
                "{\"left_tracked\":" + (leftNow ? "true" : "false") +
                ",\"right_tracked\":" + (rightNow ? "true" : "false") + "}");
        }

        private static bool IsTracked(OVRPlugin.Hand hand, ref OVRPlugin.HandState state)
        {
            return OVRPlugin.GetHandState(OVRPlugin.Step.Render, hand, ref state)
                && (state.Status & OVRPlugin.HandStatus.HandTracked) != 0;
        }
    }
}
