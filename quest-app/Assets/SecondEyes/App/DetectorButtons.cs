using SecondEyes.Perception;
using UnityEngine;

namespace SecondEyes.App
{
    /// <summary>
    /// The detector's controls on the left controller (A1.10c, D88). Y turns it on and off. While it is off: pressing
    /// the left thumbstick switches between the GPU and CPU backends, pushing it up or down cycles the mode (scan,
    /// burst, continuous) and pushing it left or right cycles the scheduling steps per frame. While it is on, the left
    /// index trigger saves the next camera frame's letterboxed input as a snapshot. The overlay shows the settings.
    /// </summary>
    public class DetectorButtons : MonoBehaviour
    {
        [SerializeField] private DetectorRunner detector;

        private void Update()
        {
            if (detector == null) return;
            const OVRInput.Controller left = OVRInput.Controller.LTouch;
            if (OVRInput.GetDown(OVRInput.Button.Two, left)) detector.Toggle("button_y");
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick, left)) detector.SwitchBackend("left_thumbstick");
            if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, left)) detector.RequestSnapshot("left_trigger");
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickUp, left)) detector.CycleMode(1, "left_thumbstick_up");
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickDown, left)) detector.CycleMode(-1, "left_thumbstick_down");
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickRight, left)) detector.CycleSteps(1, "left_thumbstick_right");
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickLeft, left)) detector.CycleSteps(-1, "left_thumbstick_left");
        }
    }
}
