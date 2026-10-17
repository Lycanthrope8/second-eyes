using UnityEngine;
using UnityEngine.UI;

namespace SecondEyes.Grounding
{
    // The panel's layout (A4 Delivery 1), built in code like DebugOverlay: 640 units wide at 1 mm each, in two pages
    // under a shared top (the title, More or Back, and the status line).
    // - The main page: the result box (the session's scene and last outcome), the prompt with Send, the session's row
    //   (Start session, Scene, Preset, Cancel), and the keyboard, shown while the prompt is open (ChatPanel.Prompt.cs).
    // - The More page: every earlier control at its old place: the model test (its prompt box, the prompt presets, Load
    //   model, Repeat, Steps, Send to model, Runtime, Backend, Weights, the model's answer) and the checks (Replay check,
    //   Prompt check, Pipeline check, Parser check).
    // A hidden page's buttons are inactive, so the UI pointer can't reach them. Switching pages writes no log event.
    // docs/quest-panel.md has the layout.
    public partial class ChatPanel
    {
        private const float PanelWidth = 640f, PanelHeight = 776f;   // one canvas size, so switching pages never moves the top
        private const float BoxY = 96f, BoxHeight = 196f;             // the result box (main) and the model test's prompt box (More)
        private const float PromptRowY = 304f, SessionRowY = 368f, KeyboardY = 432f;
        private const float MainBottom = 440f, MainKeyboardBottom = 662f, MoreBottom = 736f;   // where the background ends
        private const float KeyWidth = 56f, KeyHeight = 48f, KeyGap = 6f, WideKey = 110f, SpaceKey = 300f;
        private static readonly string[] KeyRows = { "qwertyuiop", "asdfghjkl", "zxcvbnm" };
        private const float InViewM = 3f, InViewDeg = 35f;   // the panel counts as in view this close, this near the gaze

        /// <summary>A4: the right thumbstick hides the panel when it is in view, and otherwise brings it in front of you,
        /// shown again if it was hidden.</summary>
        private void ToggleOrPlace()
        {
            if (panel == null) return;
            Camera head = Camera.main;
            if (panel.gameObject.activeSelf && head != null)
            {
                Vector3 to = panel.position - head.transform.position;
                if (to.magnitude < InViewM && Vector3.Angle(head.transform.forward, to) < InViewDeg)
                {
                    panel.gameObject.SetActive(false);
                    return;
                }
            }
            panel.gameObject.SetActive(true);
            Place();
        }
        private const string NoSessionHint = "No session yet. Press Start session, then press the prompt below, type a command and press Send.";

        private RectTransform background, mainPage, morePage, keyboard;
        private Text moreLabel, resultText, promptText;
        private Button sendTypedButton;
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

            // Both pages cover the whole canvas, so their children keep the panel's coordinates.
            mainPage = Box(panel, "MainPage", 0f, 0f, PanelWidth, PanelHeight);
            morePage = Box(panel, "MorePage", 0f, 0f, PanelWidth, PanelHeight);
            BuildMorePage();
            BuildMainPage();
            BuildSessionRow();
            ShowPage(false);
            ShowKeyboard(false);
            UpdateReplayButton();
            UpdateLabels();
        }

        /// <summary>The main page: the result box, the prompt with Send, and the keyboard (hidden until the prompt is
        /// pressed). The session's row is added by BuildSessionRow.</summary>
        private void BuildMainPage()
        {
            RectTransform box = Box(mainPage, "ResultBox", 20f, BoxY, 600f, BoxHeight);
            box.gameObject.AddComponent<Image>().color = new Color(0.16f, 0.18f, 0.21f);
            resultText = Label(box, "Text", 12f, 8f, 576f, 180f, 13, FontStyle.Normal);
            resultText.supportRichText = false;
            resultText.text = NoSessionHint;

            RowButton("Prompt", 20f, 450f, out promptText, ToggleKeyboard, PromptRowY, mainPage);
            promptText.alignment = TextAnchor.MiddleLeft;
            promptText.fontSize = 16;
            Text sendLabel;
            sendTypedButton = RowButton("Send", 482f, 138f, out sendLabel, SendTyped, PromptRowY, mainPage);
            sendLabel.text = "Send";

            keyboard = Box(mainPage, "Keyboard", 0f, KeyboardY, PanelWidth, 4f * KeyHeight + 3f * KeyGap);
            for (int r = 0; r < KeyRows.Length; r++)
            {
                string row = KeyRows[r];
                bool withDelete = r == KeyRows.Length - 1;
                float width = row.Length * KeyWidth + (row.Length - 1) * KeyGap + (withDelete ? KeyGap + WideKey : 0f);
                float x = (PanelWidth - width) / 2f, y = r * (KeyHeight + KeyGap);
                for (int k = 0; k < row.Length; k++)
                {
                    char c = row[k];
                    Key("Key " + c, c.ToString(), x + k * (KeyWidth + KeyGap), y, KeyWidth, () => TypeKey(c));
                }
                if (withDelete) Key("Delete", "Delete", x + row.Length * (KeyWidth + KeyGap), y, WideKey, DeleteLast);
            }
            float lastY = KeyRows.Length * (KeyHeight + KeyGap);
            float lastX = (PanelWidth - (WideKey + SpaceKey + WideKey + 2f * KeyGap)) / 2f;
            Key("Clear", "Clear", lastX, lastY, WideKey, ClearTyped);
            Key("Space", "Space", lastX + WideKey + KeyGap, lastY, SpaceKey, () => TypeKey(' '));
            Key("Done", "Done", lastX + WideKey + SpaceKey + 2f * KeyGap, lastY, WideKey, () => ShowKeyboard(false));
        }

        private void Key(string name, string text, float x, float y, float width, UnityEngine.Events.UnityAction onClick)
        {
            Text label;
            RowButton(name, x, width, out label, onClick, y, keyboard, KeyHeight);
            label.text = text;
        }

        /// <summary>The More page: the model test and the checks, at their old places.</summary>
        private void BuildMorePage()
        {
            // The model test's text box only shows the chosen prompt; the preset buttons below choose it (D49).
            RectTransform box = Box(morePage, "TextBox", 20f, BoxY, 600f, BoxHeight);
            Image boxImage = box.gameObject.AddComponent<Image>();
            boxImage.color = new Color(0.16f, 0.18f, 0.21f);
            boxText = Label(box, "Text", 12f, 8f, 576f, 180f, 16, FontStyle.Normal);
            boxText.supportRichText = false;
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

            RectTransform buttonBox = Box(morePage, "SendToModel", 460f, 362f, 160f, 52f);
            Image buttonImage = buttonBox.gameObject.AddComponent<Image>();
            buttonImage.color = new Color(0.2f, 0.5f, 0.95f);
            buttonLabel = Label(buttonBox, "Label", 0f, 0f, 160f, 56f, 22, FontStyle.Bold);
            buttonLabel.text = "Send to model";
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

            Label(morePage, "AnswerTitle", 20f, 488f, 600f, 28f, 18, FontStyle.Bold).text = "Model answer";
            answer = Label(morePage, "Answer", 20f, 518f, 600f, 130f, 18, FontStyle.Normal);
            // The checks keep their code names (Replay, Golden, Interactive, ParserCheck); docs/quest-panel.md maps the labels.
            replayButton = RowButton("Replay", 20f, 140f, out replayLabel, StartReplay, 664f, morePage);
            replayLabel.text = "Replay check";
            goldenButton = RowButton("Golden", 173f, 140f, out goldenLabel, StartGolden, 664f, morePage);
            goldenLabel.text = "Prompt check";
            interactiveButton = RowButton("Interactive", 326f, 140f, out interactiveLabel, StartInteractive, 664f, morePage);
            interactiveLabel.text = "Pipeline check";
            parserButton = RowButton("ParserCheck", 479f, 141f, out parserLabel, StartParserCheck, 664f, morePage);
            parserLabel.text = "Parser check";
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
            FitBackground();
        }

        /// <summary>The background ends below the shown page, and below the keyboard while it is open.</summary>
        private void FitBackground()
        {
            float bottom = showingMore ? MoreBottom : keyboardOpen ? MainKeyboardBottom : MainBottom;
            background.sizeDelta = new Vector2(PanelWidth, bottom);
        }

        /// <summary>A button with a centred label, on the given parent (the panel itself by default).</summary>
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
