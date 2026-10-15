using UnityEngine;
using UnityEngine.UI;

namespace SecondEyes.Grounding
{
    // The panel's layout (A4 Delivery 1), built in code like DebugOverlay: 640 units wide at 1 mm each, in two pages.
    // The top is always shown: the title, More (Back on the More page), the status line and the text box, which shows the
    // chosen prompt or the session's outcomes. Below it, the main page holds the parser session's row
    // (ChatPanel.Session.cs), and the More page holds every earlier control, unchanged and at its old place: the
    // A1.7-A1.8 model test (prompt presets, Load model, Repeat, Steps, Send, Runtime, Backend, Weights, the answer) and
    // the A2.5-A2.7 checks (Replay, Golden check, Interactive check, Parser check). A hidden page's buttons are inactive,
    // so the UI pointer can't reach them. Switching pages writes no log event. docs/quest-panel.md has the layout.
    public partial class ChatPanel
    {
        private const float PanelWidth = 640f, PanelHeight = 776f;   // one canvas size, so switching pages never moves the top
        private const float MainBottom = 376f, MoreBottom = 736f;    // where the background ends on each page
        private const float SessionRowY = 304f;                       // the session row, on the main page

        private RectTransform background, mainPage, morePage;
        private Text moreLabel;
        private bool showingMore;

        private void BuildPanel()
        {
            var canvasObject = new GameObject("ChatPanelCanvas", typeof(Canvas), typeof(CanvasScaler));
            canvasObject.transform.SetParent(transform, false);
            canvasObject.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            canvasObject.GetComponent<CanvasScaler>().dynamicPixelsPerUnit = 4f;   // sharper text in world space
            panel = (RectTransform)canvasObject.transform;
            panel.sizeDelta = new Vector2(PanelWidth, PanelHeight);
            panel.localScale = Vector3.one * 0.001f;
            panel.position = new Vector3(0f, -100f, 0f);   // out of sight until Place() runs

            background = Box(panel, "Background", 0f, 0f, PanelWidth, MainBottom);
            background.gameObject.AddComponent<Image>().color = new Color(0.08f, 0.09f, 0.11f, 0.92f);
            Label(panel, "Title", 20f, 16f, 460f, 36f, 26, FontStyle.Bold).text = "Second Eyes";
            RowButton("More", 500f, 120f, out moreLabel, ToggleMore, 12f, panel, 40f);
            status = Label(panel, "Status", 20f, 56f, 600f, 30f, 18, FontStyle.Normal);
            status.color = new Color(0.75f, 0.8f, 0.85f);

            // The text box only shows text: the chosen prompt (the preset buttons choose it, D49), or the session's outcomes.
            RectTransform box = Box(panel, "TextBox", 20f, 96f, 600f, 196f);
            Image boxImage = box.gameObject.AddComponent<Image>();
            boxImage.color = new Color(0.16f, 0.18f, 0.21f);
            boxText = Label(box, "Text", 12f, 8f, 576f, 180f, 16, FontStyle.Normal);
            boxText.supportRichText = false;

            // Both pages cover the whole canvas, so their children keep the panel's coordinates.
            mainPage = Box(panel, "MainPage", 0f, 0f, PanelWidth, PanelHeight);
            morePage = Box(panel, "MorePage", 0f, 0f, PanelWidth, PanelHeight);
            BuildMorePage();
            BuildSessionRow();
            ShowPage(false);
            UpdateReplayButton();
            UpdateLabels();
        }

        /// <summary>The More page: the A1.7-A1.8 model test and the A2.5-A2.7 checks, as they were.</summary>
        private void BuildMorePage()
        {
            for (int k = 0; k < choices.Count && k < 4; k++)   // up to four prompts fit in a row
            {
                int which = k;
                RectTransform presetBox = Box(morePage, "Preset " + choices[k].id, 20f + k * 153f, 300f, 140f, 50f);
                Image presetImage = presetBox.gameObject.AddComponent<Image>();
                presetImage.color = PresetColor;
                Text presetText = Label(presetBox, "Label", 6f, 3f, 128f, 44f, 14, FontStyle.Normal);
                presetText.text = Command(choices[k]);
                presetText.alignment = TextAnchor.MiddleCenter;
                Button preset = presetBox.gameObject.AddComponent<Button>();
                preset.targetGraphic = presetImage;
                preset.onClick.AddListener(() => Choose(which));
                presetButtons.Add(preset);
            }

            loadButton = RowButton("Load", 20f, 140f, out loadLabel, LoadModel, 362f, morePage);
            loadLabel.text = "Load model";
            loadButton.interactable = false;   // until the settings are read
            repeatButton = RowButton("Repeat", 173f, 140f, out repeatLabel, ToggleRepeat, 362f, morePage);
            repeatButton.interactable = false;   // until the model is loaded
            stepsButton = RowButton("Steps", 326f, 120f, out stepsLabel, NextSteps, 362f, morePage);

            RectTransform buttonBox = Box(morePage, "Send", 460f, 362f, 160f, 52f);
            Image buttonImage = buttonBox.gameObject.AddComponent<Image>();
            buttonImage.color = new Color(0.2f, 0.5f, 0.95f);
            buttonLabel = Label(buttonBox, "Label", 0f, 0f, 160f, 56f, 22, FontStyle.Bold);
            buttonLabel.text = "Send";
            buttonLabel.alignment = TextAnchor.MiddleCenter;
            sendButton = buttonBox.gameObject.AddComponent<Button>();
            sendButton.targetGraphic = buttonImage;
            sendButton.interactable = false;
            sendButton.onClick.AddListener(OnButton);

            // Backend and Weights apply at Load (A1.8a, D53)
            runtimeButton = RowButton("Runtime", 20f, 190f, out runtimeLabel, ToggleRuntime, 424f, morePage);
            backendButton = RowButton("Backend", 223f, 190f, out backendLabel, ToggleBackend, 424f, morePage);
            weightsButton = RowButton("Weights", 426f, 194f, out weightsLabel, ToggleWeights, 424f, morePage);
            runtimeButton.interactable = backendButton.interactable = weightsButton.interactable = false;   // until the settings are read

            Label(morePage, "AnswerTitle", 20f, 488f, 600f, 28f, 18, FontStyle.Bold).text = "Answer";
            answer = Label(morePage, "Answer", 20f, 518f, 600f, 130f, 18, FontStyle.Normal);
            replayButton = RowButton("Replay", 20f, 140f, out replayLabel, StartReplay, 664f, morePage);
            replayLabel.text = "Replay (A2.5 d1)";
            goldenButton = RowButton("Golden", 173f, 140f, out goldenLabel, StartGolden, 664f, morePage);
            goldenLabel.text = "Golden check (A2.5 d2)";
            interactiveButton = RowButton("Interactive", 326f, 140f, out interactiveLabel, StartInteractive, 664f, morePage);
            interactiveLabel.text = "Interactive check (A2.5 d3)";
            parserButton = RowButton("ParserCheck", 479f, 141f, out parserLabel, StartParserCheck, 664f, morePage);
            parserLabel.text = "Parser check (A2.7)";
        }

        private void ToggleMore()
        {
            ShowPage(!showingMore);
        }

        /// <summary>Shows the main page or the More page; the top stays where it is.</summary>
        private void ShowPage(bool more)
        {
            showingMore = more;
            mainPage.gameObject.SetActive(!more);
            morePage.gameObject.SetActive(more);
            moreLabel.text = more ? "Back" : "More";
            background.sizeDelta = new Vector2(PanelWidth, more ? MoreBottom : MainBottom);
        }

        /// <summary>A button with a centred label, on the given page (the panel itself by default).</summary>
        private Button RowButton(string name, float x, float width, out Text label, UnityEngine.Events.UnityAction onClick,
                                 float y = 362f, RectTransform parent = null, float height = 52f)
        {
            RectTransform rect = Box(parent != null ? parent : panel, name, x, y, width, height);
            Image image = rect.gameObject.AddComponent<Image>();
            image.color = PresetColor;
            label = Label(rect, "Label", 4f, 0f, width - 8f, height, 17, FontStyle.Normal);
            label.alignment = TextAnchor.MiddleCenter;
            Button button = rect.gameObject.AddComponent<Button>();
            button.targetGraphic = image;
            button.onClick.AddListener(onClick);
            return button;
        }

        /// <summary>A child rectangle placed from the parent's top-left corner, in panel units.</summary>
        private static RectTransform Box(Transform parent, string name, float x, float y, float width, float height)
        {
            var box = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
            box.SetParent(parent, false);
            box.anchorMin = box.anchorMax = box.pivot = new Vector2(0f, 1f);
            box.anchoredPosition = new Vector2(x, -y);
            box.sizeDelta = new Vector2(width, height);
            return box;
        }

        private static Text Label(Transform parent, string name, float x, float y, float width, float height, int size, FontStyle style)
        {
            Text label = Box(parent, name, x, y, width, height).gameObject.AddComponent<Text>();
            label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            label.fontSize = size;
            label.fontStyle = style;
            label.color = Color.white;
            label.horizontalOverflow = HorizontalWrapMode.Wrap;
            label.verticalOverflow = VerticalWrapMode.Truncate;
            return label;
        }
    }
}
