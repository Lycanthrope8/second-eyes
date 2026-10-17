using UnityEngine;

namespace SecondEyes.Grounding
{
    // The typed command (A4 Delivery 1). Pressing the prompt on the main page opens the panel's own keyboard (and closes it
    // again); Done closes it too. The Quest system keyboard isn't used: it didn't appear in this OpenXR setup (D49, r015),
    // while the panel's keys need no manifest change or package and work with the UI pointer like every other button.
    // Send hands the text to the running session as a request with source "typed" (InteractiveService.SubmitTyped); its
    // outcome shows in the result box like a preset's. Only text that is non-empty after trimming is sent, the inbox's rule.
    public partial class ChatPanel
    {
        private const int PromptLimit = 200;   // characters; a command is a short phrase
        private const int PromptShown = 80;    // the prompt shows the last 80 characters of a longer command
        private static readonly Color HintColor = new Color(0.6f, 0.65f, 0.7f);

        private string typed = "";
        private bool keyboardOpen;

        private void ToggleKeyboard()
        {
            ShowKeyboard(!keyboardOpen);
        }

        private void ShowKeyboard(bool open)
        {
            keyboardOpen = open;
            keyboard.gameObject.SetActive(open);
            FitBackground();
            ShowTyped();
        }

        private void TypeKey(char c)
        {
            if (typed.Length >= PromptLimit)
            {
                SetStatus("A command can have up to " + PromptLimit + " characters.");
                return;
            }
            typed += c;
            ShowTyped();
        }

        private void DeleteLast()
        {
            if (typed.Length > 0) typed = typed.Substring(0, typed.Length - 1);
            ShowTyped();
        }

        private void ClearTyped()
        {
            typed = "";
            ShowTyped();
        }

        /// <summary>The prompt shows the command with a caret while the keyboard is open, or a hint when it is empty.</summary>
        private void ShowTyped()
        {
            if (typed.Length == 0 && !keyboardOpen)
            {
                promptText.text = "Press here to type a command";
                promptText.color = HintColor;
                return;
            }
            string shown = typed.Length > PromptShown ? "..." + typed.Substring(typed.Length - PromptShown) : typed;
            promptText.text = shown + (keyboardOpen ? "|" : "");
            promptText.color = Color.white;
        }

        private void SendTyped()
        {
            if (interactiveSession == null || interactiveSession.Current == null)
            {
                SetStatus("Start a session first; the command stays in the prompt.");
                return;
            }
            string command = typed.Trim();
            if (command.Length == 0)
            {
                SetStatus("Type a command first: press the prompt to open the keyboard.");
                return;
            }
            BeforeCommand();   // A4: the newest scan snapshot first, then the scan waits for the outcome
            string id = interactiveSession.SubmitTyped(command);
            SetStatus("Sent " + id + ": \"" + command + "\" (" + interactiveSession.Waiting + " waiting)");
            typed = "";
            ShowKeyboard(false);
        }
    }
}
