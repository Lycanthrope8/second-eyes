# Decisions

Open questions first, then decisions taken, newest first. A decision records what was chosen, the alternatives, the reason or evidence, and who decided. Suggestions become decisions only when approved.

## Open

| ID | Question | Needed by | Options so far |
|---|---|---|---|
| O3 | Coordinate frames (and preferred units) for poses in logs | the first logged pose | to be proposed; Unity is left-handed with Y up, while the memory example in proposal §4.3 has Z up |
| O4 | How Vicon data are recorded for clock alignment | first use of Vicon (D10) | exported files · live stream (Vicon DataStream SDK) |
| O5 | What to disable on the Quest 3 | A1.6 (D11) | decided with our app running, from its profile |
| O6 | Where each component's headset (C#) code lives | the first component with C# code (D14) | inside `quest-app/` · in its own component folder as a Unity local package |
| O7 | Where Track B (SpatialLM) work lives | B1 | to be proposed |
| O8 | Git tag names for gates | Gate A | to be proposed |
| O9 | Raw-data storage (S0 item 5) | after Gate A | deferred, see D1 |
| O10 | Experiment tracking and figure scripts (S0 item 6) | after Gate A | deferred, see D1 |
| O12 | How a headset log is tied to its run | A1, with the log-pull command (D7) | tie at pull time (the pull copies new session files into `runs/<run ID>/raw/`) · give the headset the run ID before each session |

## Decided

| ID | Date | Decision | Alternatives considered | Reason / evidence | By |
|---|---|---|---|---|---|
| D14 | 2026-09-27 | A1.3 as proposed: URP; package name `com.secondeyes.quest`; app-level code in `quest-app/Assets/SecondEyes/`; Unity and SDK versions recorded in `quest-app/README.md`; O6 waits until the first component has C# code | none recorded | A1.3 proposal approved as proposed | project lead |
| D13 | 2026-09-27 | Target frame rate 72 Hz | 90 Hz | As recommended in the A1.3 proposal: the most headroom for the model, speech recognition and the detector | project lead |
| D12 | 2026-09-27 | Meta XR SDK for the Quest app | Unity OpenXR + AR Foundation | As recommended in the A1.3 proposal: the Meta samples the research proposal cites (on-device inference, object detection, passthrough camera API) build on it | project lead |
| D11 | 2026-09-27 | Headset left unchanged; what to disable (O5) is decided in A1.6 with our app running | Uninstall some of the 8 user-installed apps; disable the unneeded system apps that were running | Baseline `20260927_A1_r001`: about 2% CPU in use and 4.6 GiB free at idle; the unneeded apps that were running held about 0.23 GiB, mostly cached. As recommended after the baseline | project lead |
| D10 | 2026-09-27 | Time sync (S0 item 4) and every Vicon question, including O4, wait until Vicon is actually used | The S0.4 proposal: motion sync, shaking a marker-fitted controller at the start and end of each session | Project lead's instruction: no need to think about Vicon until using it | project lead |
| D9 | 2026-09-27 | Event log format v1 as proposed: JSON Lines, one file per session; every line has `seq`, `mono_us`, `utc_us`, `ev`, `data`; v1 defines `session.start`, `session.end`, `mark` and `error`; pipeline events get their fields in the phase that first emits them and sync events in S0.4; a field with a unit carries it in its name; O3 moves to the first logged pose. Details in `docs/logging.md` | Clocks in nanoseconds (Unix time would not stay exact in JSON readers) | S0.3 proposal approved as proposed | project lead |
| D8 | 2026-09-27 | Until O9 is decided, a run's logs and recordings live in `runs/<run ID>/raw/`, which Git ignores | A folder outside the repository, with its path in the config | As proposed in S0.3; each run's data sits next to its config | project lead |
| D7 | 2026-09-27 | The C# logger and the log-pull command are built in A1, not S0.3; how a log is tied to its run moves with them (O12) | Build them in S0.3, untested | As recommended in the S0.3 proposal: they depend on the app's package name, its file location on the headset and O6, and can't be tested without the app | project lead |
| D6 | 2026-09-27 | `git_dirty` counts every uncommitted change, including `runs/`, `notes/` and `docs/` (was O11) | Ignore those three folders, as first built in S0.2 | Project lead's choice. Consequence: commit before creating each run, or the run is marked dirty | project lead |
| D5 | 2026-09-27 | Run registry v1 as proposed: IDs `YYYYMMDD_<phase>_r<NNN>` with the local date and a phase from the proposal's list; purpose and operator required; the optional fields of proposal S0 item 2; "schema version" means the model's output/scene schema (§6.2); `new` refuses outside a Git repository; `--checkpoint` and `--headset` extras. Details in `runs/README.md` | The same without the two extras | S0.2 proposal approved as proposed | project lead |
| D4 | 2026-09-27 | The run tool lives in a new top-level `tools/` folder (was O2) | Inside `analysis/` | As recommended in the S0.2 proposal: creating runs isn't analysis, and later operator helpers such as pulling logs off the headset fit there too | project lead |
| D3 | 2026-09-27 | Run numbers count per phase and never reset (was O1) | Restart each day | As recommended in the S0.2 proposal: "A1 r007" is unique on its own and shows how many runs a phase took | project lead |
| D2 | 2026-09-27 | Top level split by component, as listed in proposal S0 (Quest app, perception, grounding, drone bridge, analysis, notes), plus `docs/`, `runs/` and `schemas/` | Split by where code runs (`headset/`, `workstation/`) | Project lead's choice; matches the proposal's S0 wording. Known cost: Unity code normally lives inside `quest-app/`, so a component's C# code may sit apart from its Python prototype (see O6) | project lead |
| D1 | 2026-09-27 | S0 items 5 (raw-data storage) and 6 (experiment tracking) deferred until headset feasibility is known (Gate A) | Set them up in S0, as planned | Project lead's instruction: set them up once headset feasibility is confirmed | project lead |
