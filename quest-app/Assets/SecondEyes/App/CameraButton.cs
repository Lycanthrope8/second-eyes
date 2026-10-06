using SecondEyes.Perception;
using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// X on the left controller turns the passthrough camera on or off (A1.10a). The right controller keeps its
    /// A1 bindings: A marks the log, B is the stop button (D21) and the trigger drives the UI pointer.
    /// </summary>
    public class CameraButton : MonoBehaviour
    {
        [SerializeField] private PassthroughFrameSource frameSource;

        private void Update()
        {
            if (frameSource != null && OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.LTouch))
            {
                frameSource.Toggle("button_x");
            }
        }
    }
}
