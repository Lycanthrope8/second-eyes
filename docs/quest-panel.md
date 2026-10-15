# The headset panel

The app's one on-screen panel, about a metre ahead (the right thumbstick's click puts it in front again). It is built in
code: `ChatPanel.Layout.cs` lays it out, and `ChatPanel.cs` and `ChatPanel.Session.cs` hold what the buttons do.
Since A4 Delivery 1 it has two pages under a shared top. Nothing was removed: every control works as before.

**The top, always shown:** the title, **More** (**Back** on the More page), the status line and the text box. The text
box shows the chosen prompt, or, while a session runs, the session's outcomes.

**The main page:** the parser session's row (`quest-interactive.md`): **Start session** (**End session** while one runs),
**Scene**, **Preset** and **Cancel**. Scene and Preset stay until A4's scan replaces them.

**The More page** (press **More**; **Back** returns), with every control at its old place:

| Row | Controls | Described in |
|---|---|---|
| Prompt presets | up to four prompt buttons; the chosen one fills the text box | `ChatPanel.cs`'s summary (A1.7) |
| 1 | Load model, Repeat, Steps (Threads with llama.cpp), Send | `profiling.md` (A1.7, A1.8) |
| 2 | Runtime, Backend, Weights | `profiling.md`, `setup/quest-model.md` |
| Answer | the model's reply | `ChatPanel.cs`'s summary |
| 3 | Replay (A2.5 d1), Golden check (A2.5 d2), Interactive check (A2.5 d3), Parser check (A2.7) | `quest-replay.md`, `quest-prompting.md`, `quest-interactive.md`, `a27-headset.md` |

A hidden page's buttons are inactive, so the pointer can't reach them. Switching pages changes only what is shown and
writes no log event. The controller's buttons (X, Y, A, B, grip, thumbstick, triggers) aren't on the panel and keep
their meanings.

The layout is type-checked outside Unity with the panel's other files: `dotnet build tools/csharp-tests/PanelPartial`
(`tools/csharp-tests/README.md`).
