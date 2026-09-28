using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// The stop button: B on the right controller logs control.stop (D21).
    /// This path goes straight from the button to its action and must stay independent of speech
    /// recognition and the language model (proposal section 6.7). Once the drone is connected, it
    /// will also send the land command here.
    /// </summary>
    public class StopButton : MonoBehaviour
    {
        private void Update()
        {
            // Read from the right controller only, so no hand gesture can be taken for B.
            if (OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch))
            {
                EventLog.Write("control.stop", "{\"source\":\"button_b\"}");
            }
        }
    }
}
