using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace SecondEyes.App
{
    /// <summary>
    /// A small pointer for world-space panels (A1.7c-2, D41). It draws a ray from the right controller, or from the right
    /// hand while the hand is tracked and no controller is. The controller's trigger, or a thumb-index pinch, clicks
    /// whatever the ray points at: a button runs its onClick, and a text box is selected, which on the headset opens the
    /// Quest system keyboard (D42). It works on every world-space Canvas in the scene and sends Unity UI's pointer events
    /// itself, so no input module is needed. A (mark) and B (stop) keep their meanings, and no hand model is drawn (D21).
    /// </summary>
    public class UiPointer : MonoBehaviour
    {
        [Tooltip("How far the ray reaches, in meters.")]
        [SerializeField] private float maxDistanceM = 3f;
        [SerializeField] private Color idleColor = new Color(1f, 1f, 1f, 0.35f);
        [SerializeField] private Color hoverColor = new Color(0.35f, 0.8f, 1f, 0.95f);

        private OVRCameraRig rig;
        private LineRenderer line;
        private Transform dot;
        private Material dotMaterial;
        private Selectable hovered;
        private PointerEventData eventData;
        private OVRPlugin.HandState hand = new OVRPlugin.HandState();
        private bool pinchedBefore;
        private Canvas[] canvases = new Canvas[0];
        private float nextCanvasScan;

        private void Start()
        {
            if (EventSystem.current == null)
            {
                new GameObject("EventSystem", typeof(EventSystem));   // text boxes need one to be selected
            }
            Shader shader = Shader.Find("Sprites/Default") ?? Shader.Find("UI/Default");
            if (shader == null)
            {
                Debug.LogError("[UiPointer] No unlit shader found; the ray is not drawn, but clicking still works.");
                return;
            }
            line = new GameObject("UiPointerRay").AddComponent<LineRenderer>();
            line.material = new Material(shader);
            line.useWorldSpace = true;
            line.positionCount = 2;
            line.widthMultiplier = 0.003f;
            line.enabled = false;

            GameObject dotObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Destroy(dotObject.GetComponent<Collider>());
            dotObject.name = "UiPointerDot";
            dot = dotObject.transform;
            dot.localScale = Vector3.one * 0.012f;
            dotMaterial = new Material(shader);
            dotObject.GetComponent<Renderer>().material = dotMaterial;
            dotObject.SetActive(false);
        }

        private void Update()
        {
            if (Time.unscaledTime >= nextCanvasScan)
            {
                canvases = FindObjectsByType<Canvas>(FindObjectsSortMode.None);
                nextCanvasScan = Time.unscaledTime + 1f;
            }

            Ray ray;
            bool pressed;
            if (!TryGetRay(out ray, out pressed))
            {
                Show(false, Vector3.zero, Vector3.zero, false);
                SetHovered(null);
                return;
            }

            Selectable target = null;
            float best = maxDistanceM;
            Vector3 end = ray.GetPoint(maxDistanceM);
            foreach (Canvas canvas in canvases)
            {
                if (canvas == null || !canvas.isActiveAndEnabled || canvas.renderMode != RenderMode.WorldSpace)
                {
                    continue;
                }
                var plane = new Plane(canvas.transform.forward, canvas.transform.position);
                float distance;
                if (!plane.Raycast(ray, out distance) || distance > best)
                {
                    continue;
                }
                Vector3 point = ray.GetPoint(distance);
                foreach (Selectable selectable in canvas.GetComponentsInChildren<Selectable>())
                {
                    if (!selectable.IsInteractable())
                    {
                        continue;
                    }
                    var rect = (RectTransform)selectable.transform;
                    Vector3 local = rect.InverseTransformPoint(point);
                    if (rect.rect.Contains(new Vector2(local.x, local.y)))
                    {
                        target = selectable;
                        best = distance;
                        end = point;
                    }
                }
            }

            SetHovered(target);
            Show(true, ray.origin, end, target != null);
            if (pressed)
            {
                if (target != null)
                {
                    ExecuteEvents.Execute(target.gameObject, EventData(), ExecuteEvents.pointerClickHandler);
                }
                else if (EventSystem.current != null)
                {
                    EventSystem.current.SetSelectedGameObject(null);   // clicking empty space leaves the text box
                }
            }
        }

        /// <summary>The ray and whether a click started this frame: the right controller first, else the right hand.</summary>
        private bool TryGetRay(out Ray ray, out bool pressed)
        {
            ray = default(Ray);
            pressed = false;
            if (rig == null)
            {
                rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig == null)
                {
                    return false;
                }
            }
            if (OVRInput.IsControllerConnected(OVRInput.Controller.RTouch)
                && OVRInput.GetControllerPositionTracked(OVRInput.Controller.RTouch))
            {
                Transform anchor = rig.rightControllerAnchor;
                ray = new Ray(anchor.position, anchor.forward);
                pressed = OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
                pinchedBefore = false;
                return true;
            }
            if (OVRPlugin.GetHandState(OVRPlugin.Step.Render, OVRPlugin.Hand.HandRight, ref hand)
                && (hand.Status & OVRPlugin.HandStatus.HandTracked) != 0
                && (hand.Status & OVRPlugin.HandStatus.InputStateValid) != 0)
            {
                OVRPose pose = hand.PointerPose.ToOVRPose();   // tracking space, like the hand ray of the Quest's own menus
                Transform space = rig.trackingSpace;
                ray = new Ray(space.TransformPoint(pose.position), space.rotation * (pose.orientation * Vector3.forward));
                bool pinched = (hand.Pinches & OVRPlugin.HandFingerPinch.Index) != 0;
                pressed = pinched && !pinchedBefore;
                pinchedBefore = pinched;
                return true;
            }
            pinchedBefore = false;
            return false;
        }

        private void SetHovered(Selectable target)
        {
            if (target == hovered)
            {
                return;
            }
            if (hovered != null)
            {
                ExecuteEvents.Execute(hovered.gameObject, EventData(), ExecuteEvents.pointerExitHandler);
            }
            hovered = target;
            if (hovered != null)
            {
                ExecuteEvents.Execute(hovered.gameObject, EventData(), ExecuteEvents.pointerEnterHandler);
            }
        }

        private PointerEventData EventData()
        {
            if (eventData == null && EventSystem.current != null)
            {
                eventData = new PointerEventData(EventSystem.current) { button = PointerEventData.InputButton.Left };
            }
            return eventData;
        }

        private void Show(bool visible, Vector3 from, Vector3 to, bool onTarget)
        {
            if (line == null)
            {
                return;
            }
            line.enabled = visible;
            dot.gameObject.SetActive(visible && onTarget);
            if (!visible)
            {
                return;
            }
            Color color = onTarget ? hoverColor : idleColor;
            line.startColor = color;
            line.endColor = new Color(color.r, color.g, color.b, color.a * 0.3f);
            line.SetPosition(0, from);
            line.SetPosition(1, to);
            dot.position = to;
            dotMaterial.color = color;
        }
    }
}
