using SecondEyes.Logging;
using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// Adds a mark to the event log each time you press A on the right controller.
    /// </summary>
    public class MarkButton : MonoBehaviour
    {
        private int count;

        private void Update()
        {
            // Read from the right controller only, so a hand pinch can't be taken for A.
            if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch))
            {
                count++;
                EventLog.Mark($"button A #{count}");
            }
        }
    }
}
