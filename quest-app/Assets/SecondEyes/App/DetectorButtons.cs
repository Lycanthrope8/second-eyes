using SecondEyes.Perception;
using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// The detector's controls on the left controller (A1.10c): Y turns it on and off; pressing the left thumbstick
    /// switches between the GPU and CPU backends while it is off; the left index trigger saves the next camera frame's
    /// letterboxed input as a snapshot while it is on.
    /// </summary>
    public class DetectorButtons : MonoBehaviour
    {
        [SerializeField] private DetectorRunner detector;

        private void Update()
        {
            if (detector == null) return;
            if (OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.LTouch)) detector.Toggle("button_y");
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick, OVRInput.Controller.LTouch)) detector.SwitchBackend("left_thumbstick");
            if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.LTouch)) detector.RequestSnapshot("left_trigger");
        }
    }
}
