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
            if (OVRInput.GetDown(OVRInput.Button.One))
            {
                count++;
                EventLog.Mark($"button A #{count}");
            }
        }
    }
}
