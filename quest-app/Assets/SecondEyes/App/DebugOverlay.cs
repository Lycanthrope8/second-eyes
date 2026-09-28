using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

namespace SecondEyes.App
{
    /// <summary>
    /// A small text that follows the view, low in the field of view. It shows the app version,
    /// the frames per second the app renders, and the refresh rate the headset runs at.
    /// It updates once per second, so it costs almost nothing.
    /// </summary>
    public class DebugOverlay : MonoBehaviour
    {
        [Tooltip("Where the text sits relative to the head, in meters: right, up, forward.")]
        [SerializeField] private Vector3 offsetM = new Vector3(0f, -0.25f, 1.0f);

        [Tooltip("Height of the whole text block, in meters.")]
        [SerializeField] private float heightM = 0.06f;

        [Tooltip("How often the text updates, in seconds.")]
        [SerializeField] private float updateIntervalS = 1f;

        private readonly List<XRDisplaySubsystem> displays = new List<XRDisplaySubsystem>();
        private TextMesh label;
        private MeshRenderer labelRenderer;
        private int frames;
        private float windowStart;
        private bool sized;

        private void Start()
        {
            Camera head = Camera.main != null ? Camera.main : FindAnyObjectByType<Camera>();
            if (head == null)
            {
                Debug.LogWarning("[SecondEyes] DebugOverlay: no camera found, overlay disabled.");
                enabled = false;
                return;
            }

            var textObject = new GameObject("DebugOverlayText");
            textObject.transform.SetParent(head.transform, false);
            textObject.transform.localPosition = offsetM;

            label = textObject.AddComponent<TextMesh>();
            label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            label.fontSize = 64;
            label.characterSize = 1f;
            label.anchor = TextAnchor.MiddleCenter;
            label.alignment = TextAlignment.Center;
            label.color = Color.white;
            label.text = "Second Eyes\nstarting";

            labelRenderer = textObject.GetComponent<MeshRenderer>();
            labelRenderer.sharedMaterial = label.font.material;

            windowStart = Time.unscaledTime;
        }

        private void Update()
        {
            frames++;
            float now = Time.unscaledTime;
            float elapsed = now - windowStart;
            if (elapsed < updateIntervalS)
            {
                return;
            }

            float fps = frames / elapsed;
            frames = 0;
            windowStart = now;
            label.text = $"Second Eyes {Application.version}\n{fps:0.0} fps | {RefreshRateText()}";

            if (!sized)
            {
                FitHeight();
            }
        }

        private string RefreshRateText()
        {
            SubsystemManager.GetSubsystems(displays);
            foreach (XRDisplaySubsystem display in displays)
            {
                if (display.running && display.TryGetDisplayRefreshRate(out float hz))
                {
                    return $"{hz:0} Hz";
                }
            }
            return "? Hz";
        }

        // TextMesh sizes aren't in meters, so scale the text once until the block is heightM tall.
        private void FitHeight()
        {
            float height = labelRenderer.localBounds.size.y;
            if (height <= 0f)
            {
                return; // the text mesh isn't built yet; try again at the next update
            }
            label.transform.localScale = Vector3.one * (heightM / height);
            sized = true;
        }
    }
}
