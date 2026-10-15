# The headset panel

The app's one on-screen panel, about a metre ahead (the right thumbstick's click puts it in front again). It is built in
code: `ChatPanel.Layout.cs` lays it out, `ChatPanel.Prompt.cs` handles typing, and `ChatPanel.cs` and
`ChatPanel.Session.cs` hold what the other buttons do. Since A4 Delivery 1 it has two pages under a shared top: the
title, **More** (**Back** on the More page) and the status line. Nothing was removed.

## The main page

- **The result box:** while a session runs, the current scene's objects with their letters, the target marked with an
  arrow, and the last outcome; before a session, a one-line hint.
- **The prompt and Send.** Pressing the prompt opens the panel's own keyboard (letters, Delete, Space, Clear, Done);
  pressing it again, or Done, closes it. **Send** submits the typed command to the running session as a request with
  source and intake `typed`, handled exactly as a preset is (`quest-interactive.md`). Commands are lowercase letters and
  spaces, up to 200 characters; only text that is non-empty after trimming is sent. The Quest system keyboard isn't used:
  it didn't appear in this OpenXR setup (D49, run r015).
- **The session's row:** **Start session** (**End session** while one runs), **Scene**, **Preset** and **Cancel**
  (`quest-interactive.md`). Send, Scene, Preset and Cancel work only while a session runs. Scene and Preset stay until
  A4's scan replaces them; until then, typed commands are grounded against the registered dataset scenes that Scene
  cycles through.

## The More page

Press **More**; **Back** returns. Every control here is at its old place; the labels lost their phase tags.

| Row | Controls | Label before A4 Delivery 1 | Described in |
|---|---|---|---|
| Prompt box | the model test's chosen prompt | (the shared text box) | `ChatPanel.cs`'s summary |
| Prompt presets | up to four prompt buttons; the chosen one fills the box | same | `ChatPanel.cs`'s summary |
| 1 | Load model, Repeat, Steps (Threads with llama.cpp), **Send to model** | Send | `profiling.md` |
| 2 | Runtime, Backend, Weights | same | `profiling.md`, `setup/quest-model.md` |
| Model answer | the model's reply | Answer | `ChatPanel.cs`'s summary |
| 3 | **Replay check** | Replay (A2.5 d1) | `quest-replay.md` |
| 3 | **Prompt check** | Golden check (A2.5 d2) | `quest-prompting.md` |
| 3 | **Pipeline check** | Interactive check (A2.5 d3) | `quest-interactive.md` |
| 3 | **Parser check** | Parser check (A2.7) | `a27-headset.md` |

## Notes

A hidden page's buttons, and the keyboard's while it is closed, are inactive, so the pointer can't reach them. Switching
pages and typing write no log event; a sent command is logged by the session like any request. The controller's
buttons (X, Y, A, B, grip, thumbstick, triggers) aren't on the panel and keep their meanings.

The panel's files are type-checked outside Unity: `dotnet build tools/csharp-tests/PanelPartial`
(`tools/csharp-tests/README.md`).
