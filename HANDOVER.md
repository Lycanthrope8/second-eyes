# Second Eyes · Handover

Written 2026-10-01, at the end of A1.8, to start A2 in a new chat. Update it at every handover.

**For the assistant in a new chat:** read this file first, then check every fact you use in the files it points to. This
file summarizes; the repository decides. Don't rely on memory, yours or a summary's: when this file and a record
disagree, the record wins, and the disagreement is worth reporting.

## The project lead's standing constraints

These apply to every chat, every phase, until the project lead changes them. The quotes are the project lead's own
words.

1. **Approval first** (2026-09-27): *"Throughout this chat you must not assume anything and do it on your own. You must
   ask me for approval."* Propose before building or changing anything, in the repository or on the headset, including
   small fixes noticed along the way. Where there's a choice, give the options with a recommendation. If something is
   unclear or inconsistent, raise it instead of fixing it quietly; if an interpretation is needed, state it and ask.
2. **Few steps per response** (2026-09-27): *"Do not try to finish a lot of steps in one response."* One step or
   sub-step per response, delivered and tested before the next.
3. **No far-future planning** (2026-09-27): *"I don't want you to overthink about too far future."* The Go/No-go gates
   may end or change the plan, so plan up to the next gate.
4. **Easy to understand later** (2026-09-27): *"Do it in such a way that in future it takes less time for me to
   understand everything"*, with a repository that is well structured and easy to maintain.
5. **Every computation on the Quest counts** (2026-09-27): *"I don't want to waste any computational power."*
6. **Later, not now:** Vicon until it is actually used (2026-09-27, D10); participants and Vicon until they are needed
   (2026-10-01); S0 items 5 and 6, storage and experiment tracking, until headset feasibility is known (D1).
7. **Earlier chats** (2026-10-01): *"If you must look into any chats, just refer to the chat named "A1 Complete (Except
   Object Detector and Speech)." Don't look into any other chats because I don't want you to be confused."* Only that
   chat, and only when needed; the repository still decides.

**A slip to avoid:** in A1.8c's results round the assistant added two tool changes nobody had asked for
(`runs.py set`, the load-jump report). Useful, but they should have been proposed first, under constraint 1.

**Direction, 2026-10-01:** finish the whole pipeline first: build every stage of the system before participants and
Vicon. This doesn't mean closing Gate A first. The order is A2, then A1.10, then A1.11, with on-device speech (A1.9)
last, before the pilot study; commands are typed text until then (D62, D63). The project lead may split the project
into one repository per paper after meeting the advisor next week; don't restructure the repository before the
project lead decides.

## 1. Sources

| What | Where |
|---|---|
| The plan | `Second_Eyes_Research_Proposal_Revision_3.pdf`, in the Claude project's files. Phase IDs (S0, A0–A10, B1–B3) follow its Section 10 and its Phases part |
| The plan's A2 revision | `Second_Eyes___Revision_3_1_supplement_A2_and_the_sections_it_touches.pdf`, in the Claude project's files; text in `docs/plan/revision-3.1-supplement.md`. Replaces Revision 3's A2 card and every passage A2 touches (D63) |
| Decisions and open items | `notes/decisions.md`: D1–D65, and the open items O3–O21 at its top |
| Failures, including the assistant's own errors | `notes/failures.md` |
| Limitations for the papers | `notes/limitations.md` |
| Phase and sub-step notes | `notes/phases/`: S0, A1.7d, A1.8b, A1.8c, and `_template.md` |
| Facts about tools, runtimes and procedures | `docs/` (see section 5) |
| Status | the table in `README.md` |
| Earlier chats | only the chat the project lead names (constraint 7), and only when needed; the repository wins when they disagree |

## 2. The project in brief

A micro-UAV pre-scans a workspace the user can't see; a Meta Quest 3 turns the drone's observations into a semantic
memory of objects and positions; the user tasks the drone in natural language about things only the drone has seen
("inspect the box behind the table"). A sub-billion-parameter language model on the headset decides what the user
means, once per command; deterministic geometry decides where the drone goes. All AI inference runs on the headset at
deployment. Track A (Paper 1) is the critical path; Track B (SpatialLM on the RTX workstation) runs alongside and
never blocks it.

## 3. Where things stand

S0 closed on 2026-09-27 (`notes/phases/S0_infrastructure.md`). A1, headset feasibility, follows an 11-step roadmap
agreed at its start:

| Step | What | Status | Evidence |
|---|---|---|---|
| A1.1 | Headset baseline | done | `20260927_A1_r001`, `docs/setup/quest3.md` |
| A1.2 | Headset cleanup | closed, no changes: the busy processes are the XR system itself | D11, D26 |
| A1.3 | App with passthrough, overlay, 72 Hz | done | `r002`, `r003`; D12–D15 |
| A1.4 | Event log on the headset, log pull | done | `r004`, `r005`; D16–D19, `docs/logging.md` |
| A1.5 | Hand tracking and stop button | done | `r006`; D20, D21 |
| A1.6 | Cost of the empty app | tools done (`r007`); the measurement folded into A1.7d | D24, D25 |
| A1.7 | Language model on the headset with Meta's runner (path A) | done: correct, but too slow and drops frames | `r008`–`r022`, `notes/phases/A1.7d_model_cost.md` |
| A1.8 | Token probabilities and caching; path B, llama.cpp | done (A1.8a–d) | `r023`–`r036`, A1.8b and A1.8c notes; D53–D60 |
| A1.9 | Offline speech recognition, push-to-talk | **open**: last, before A9 (D63) | |
| A1.10 | Object detector on passthrough, alone and beside the model | **open**: after A2 (D63) | `docs/meta-ai.md` has the starting point |
| A1.11 | Everything together, 30-minute soak: Gate A evidence | **open**: after A1.10 (D63); Gate A per O19 | |

**Gate A** is open. For the language model alone: the frame rate holds (2 stale frames in 10 minutes of commands),
memory fits (+1.07 GB, about 2 GB left free), no throttling in 10 minutes, and with the scene cached the panel as
built takes about 1.1 s of the 3 s latency budget: the greedy answer (0.8 s), then the candidates' scores (0.30 s).
The 0.6 s given earlier, here and in the A1 report, was an estimate for a scores-only design (the command's pass,
0.3 s, plus the scores), which no run measured as such (`notes/failures.md`). Speech, the detector and the 30-minute
soak are still to measure.

**Next:** A2, as the Revision 3.1 supplement plans it (D63, section 10), starting with A2.1's specification. Then
A1.10, A1.11, and A1.9 last. How Gate A is judged while speech waits is the advisor's decision, before A1.11 (D65,
O19).

## 4. The language model on the headset (A1's outcome)

| Part | What | Where |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct, pinned to commit `7ae55760…` in its description (D61) | `grounding/models/qwen2.5-0.5b-instruct.json` |
| Headset file | GGUF, 8-bit (Q8_0), 531 MB, SHA-256 starting `dd753cd6` | made by `grounding/export_gguf.py`; pushed by `grounding/llama_headset.py push-model` into `/sdcard/Android/data/com.secondeyes.quest/files/` |
| Runtime | llama.cpp `b11277` (`eae11d22`) | pinned in `native/llama.cpp.pin`; fetched and built by `tools/build_llama.py` |
| Our C interface | load with options, prompt with scene cache, greedy answer, candidate scoring in one batch (`se_score_many`) | `native/se_llama.h`, `native/se_llama.cpp`; tests in `native/tests/` |
| Command-line test program | runs a job on the headset from `adb shell` | `native/se_llama_cli.cpp`; `docs/setup/llama-headset.md` |
| In Unity | P/Invoke, one worker thread for every native call, the panel | `quest-app/Assets/SecondEyes/Grounding/LlamaNative.cs`, `LlamaRuntime.cs`, `ChatPanel.cs` |
| Configuration | flash attention automatic, weights repacked, memory-mapped, 16 sequences, **2 threads** | D58, D59 |
| Prompt format, scoring | answer prefix, candidate suffix, which objects are candidates | `grounding/scene.py`; prompts in `grounding/prompts/` (copies in the Unity project) |
| References | greedy answers: `20260929_A1_r016`; candidate log-probabilities from PyTorch 32-bit: `20260930_A1_r026` | |
| Checks | `grounding/check_headset.py <run> --reference 20260929_A1_r016 20260930_A1_r026` | judged as D56 sets out |

**D56's rule:** scores pass when the best candidate and the order of the plausible ones (share at least 1%) equal
PyTorch's, and the cached path's shares lie within a total variation distance of 0.05.

**Numbers in the app, 2 threads** (`notes/phases/A1.8c_llama_app.md`): load 1.1–1.5 s; first command 1.3 s to the
first token (the scene is evaluated); with the scene cached 0.3 s to the first token, 0.03 s per token, the 15-token
answer at 0.8 s, scoring five objects 0.30 s; 72.5 fps with 0 seconds below 71 while answering; app +1.07 GB.
Meta's runner for comparison, at 50 steps per frame: 24.8 s per answer and 37.1 fps while answering, 43.3 fps over
the run (A1.7d; while answering it gave 13.4, 37.1 and 64.6 fps at 150, 50 and 15 steps).

**Reproduced is not correct.** "Equal to the PC" means the headset reproduces the model, not that the model is right.
Zero-shot, it answers `a17-fixed` ("Inspect the box behind the table.") with `box_1`; the expected `box_2` ranks third
in PyTorch's scores (`r026`). The other three presets are right. Each prompt file records its expected target.

**Why 2 threads:** the app's extra threads get cores 3–5 only; four llama.cpp threads on three cores wait for each
other at every step, and a token took 3.45 s (D59).

## 5. Repository map

| Folder | Holds |
|---|---|
| `tools/` | `runs.py` (run registry: `new`, `check`, `note`, `set`), `logs.py` (pull headset logs into a run), `profile.py` (record a measurement from the PC), `build_llama.py` (fetch and build llama.cpp) |
| `analysis/` | `eventlog.py` (read and check event logs), `profile.py` (summarize, compare and tabulate measurements) |
| `grounding/` | the language model's PC side: exports (ONNX, GGUF), references, headset checks, `scene.py`, prompts, model descriptions. A2's code goes here |
| `native/` | `se_llama`, our C interface over llama.cpp, its test program and tests |
| `quest-app/` | the Unity project; our code in `Assets/SecondEyes/` (`App/`, `Logging/`, `Grounding/`, `Editor/`); versions in `quest-app/README.md` |
| `perception/`, `drone-bridge/` | placeholders for A4 and A6 |
| `schemas/` | `run-config.v1.json`, `log-event.v1.json` |
| `docs/` | `conventions.md`, `logging.md`, `profiling.md`, `meta-ai.md`, `llama-cpp.md`, `setup/` (`quest3.md`, `quest-model.md`, `llama-headset.md`), `plan/` (`revision-3.1-supplement.md`, the plan's text, verbatim) |
| `notes/` | decisions, failures, limitations, phase notes |
| `runs/` | one folder per run: `config.yaml` tracked, `raw/` ignored (D8) |

Not in Git: `runs/*/raw/`, model files, `build/`, `native/out/`, `third_party/`, Unity's generated folders.
Qwen's tokenizer files are tracked (D35), in `quest-app/Assets/SecondEyes/Models/<model>/`.

## 6. How we work

**Steps and approvals** (constraints 1 and 2). Each step is proposed first: what, how, its test, and any choice with
a recommendation. The project lead approves, usually by picking an option, and the approval becomes a D-entry. Nothing
changes on the headset without approval (`docs/setup/quest3.md`). Reading and analysing files the project lead sends,
to answer what was asked, has been done directly; anything that builds or changes something waits for approval.

**Instructions.** Numbered steps labeled **[PC]**, **[Unity]** or **[Headset]**, with commands to copy into Windows
PowerShell. Keep commands and run notes in plain ASCII. The newest run's ID is captured with
`$run = (Get-ChildItem runs -Directory -Filter "*_A1_r*" | Sort-Object Name | Select-Object -Last 1).Name`.

**Deliveries.** Changed files come as one zip with repository-relative paths, unzipped at the repository's root with
`Expand-Archive -Force $HOME\Downloads\<zip> .`, then committed. Git's LF→CRLF warnings are harmless.

**Runs.** Every measurement or result is a run: `python tools/runs.py new <phase> --purpose "..." --operator JH`
(`--headset` records the headset's build, `--checkpoint` hashes a model). IDs are `YYYYMMDD_<phase>_rNNN`, numbered
per phase, never reset (D3, D5). Commit before a run, or it is marked `git_dirty` (D6, D51). Afterwards
`runs.py note` and `runs.py set <run> purpose "..."` record what happened.

**Measurements on the headset** follow `docs/profiling.md`: reboot, about two minutes in Home, OVR Metrics' CSV on,
headset plugged in (D22) and kept on (O14), `tools/profile.py record`, then `tools/logs.py pull` and
`analysis/profile.py summary`. Comparisons follow D24 and D52.

**Records.** Decisions in `notes/decisions.md`; every failure in `notes/failures.md`, the assistant's own errors
marked as such; limitations in `notes/limitations.md`; a phase note from the template when a phase or sub-step ends;
facts in `docs/`. Every number traces to a run ID.

**Testing before delivery.** Python tools are tested on copies of real run files; native code is built and tested on
Linux with a tiny model (`native/tests/`), with portable builds (`-DGGML_NATIVE=OFF`); C# is checked against
stand-ins where practical. Unity, the Android build and the headset are only on the project lead's side, so their
first real test is always the project lead's run.

**The assistant's sandbox** has no GPU, no Unity and no Android NDK. It reaches GitHub and PyPI but not Hugging Face,
so real model weights stay on the project lead's PCs. Uploads with the same file name overwrite each other (four
`sampler.jsonl` became one): ask for one run's files at a time, or one zip per run.

**Communication.** The project lead likes reasons explained, short summaries when asked for, and one question at a
time with options to pick.

## 7. Setup

| | |
|---|---|
| Working PC | Windows; repository at `C:\Users\jubay\second-eyes`, Python environment `.venv`; `adb` on PATH; CMake and Ninja inside `.venv` |
| Unity | 6000.3.25f1; Android NDK 27.2.12479018 inside Unity's Android module (found by `build_llama.py`); Meta XR Core SDK 207.0.0; OpenXR 1.18.0; Inference Engine 2.2.1 |
| Headset | Meta Quest 3, developer mode; OS build in `docs/setup/quest3.md`; app `com.secondeyes.quest`; OVR Metrics Tool installed |
| Lab PC | NVIDIA RTX 6000 Pro, reached by Remote Desktop; operating system, CUDA, Python and free disk to confirm at A2's start |
| Repository | GitHub `Lycanthrope8/second-eyes`, private |
| Operator | JH, also the project lead who approves decisions |

## 8. Open items and loose ends

**Open items** (`notes/decisions.md`): O3 coordinate frames and units in logs · O4 recording Vicon for clock
alignment · O7 where Track B lives · O8 Git tag names for gates · O9 raw-data storage · O10 experiment tracking and
figure scripts · O13 drawing hand and controller models · O14 graphics memory grows about 1.6 GB per headset off-on ·
O16 where three Android permissions come from · O19 how Gate A is judged while speech waits (with the advisor) · O20
Gate B's thresholds (with the advisor) · O21 how typed commands reach the app for A2.5.

**Loose ends from A1:**

- **Latency against prompt length.** A1's checklist asks for this curve, and A2 needs it to turn a scene format's
  length into headset seconds. The data so far: whole prompt 1.39 s for 230 tokens from the command line at 2
  threads, and 1.3 s to the first token in the app, so command-line numbers carry over to the app. Measure a few
  prompt lengths with `llama_headset.py` early in A2.
- **Battery drain.** The proposal's soak measures it, but runs are plugged in (D22). A1.11 needs an unplugged run or
  a recorded decision.
- **Latency percentiles and memory per component.** The proposal's tables want the 90th percentile and PSS per
  component; the tools report medians, ranges and the app's total so far.
- **The panel's new default** (`startWithLlama`, D60) wasn't compiled by the assistant: the C# stand-ins were lost
  with a machine change. The first Unity build checks it; the panel should then show "Runtime: llama.cpp" with
  Backend and Weights greyed out.
- **A1's phase note** is written when A1 closes, from the sub-step notes.
- **A0** (scope freeze, IRB submission, thresholds agreed with the advisor) hasn't been part of these chats. Ask
  about its status when it matters: Gate B's thresholds are proposed defaults until agreed (O20), and A3 needs IRB
  approval (participants wait, constraint 6).
- **Experiment tracking.** D1 defers it until Gate A, but A2's training runs start before Gate A. Ask whether run
  records are enough for A2 or tracking comes earlier.

## 9. Lessons and pitfalls

- Unity misread model data stored past 2 GiB: keep one word table, weights file under 2 GiB (D47).
- Meta's runner: its GPU path allows no buffer over 128 MiB, 32-bit weights get the app killed, and every prompt pass
  blocks the app (D54, A1.7d).
- llama.cpp with flash attention isn't batch-invariant: the cached and uncached paths differ by a few hundredths in
  score, within D56's limits; turning it off made scores overconfident (D58).
- Threads: at most the cores the app gets; 2 in the app (D59).
- The headset coming off and on while the app runs adds about 1.6 GB (O14): runs keep the headset on.
- The summary flags memory jumps; a model load inside the recording is reported as the load instead.
- Windows PowerShell's `>` writes UTF-16; our tools write their own files.
- The assistant's sandbox can change machines between turns: build natively portable.

## 10. A2 · Grounding baselines

**The plan** is the Revision 3.1 supplement's A2 card (D63; text in `docs/plan/revision-3.1-supplement.md`), which
replaces Revision 3's. A2 ends at preliminary Gate B, whose thresholds are still to agree with the advisor (O20), and
claims no Gate A pass (D65). Commands are typed text, published data is the main training source, no custom scene
generator is built, and the natural lab commands are for evaluation only.

| Step | What | Runs on |
|---|---|---|
| A2.1 | A short specification of the scene contract, relations, outputs and outcomes; then the validator, serializer, relation library and shared resolver, checked with independently reviewed fixtures | any PC |
| A2.2 | Published data, starting with an IRef-VLA subset: access, an adapter and scene-disjoint splits; 50–100 natural lab-member text commands collected and held out | a PC with room for the data |
| A2.3 | Rules, zero-shot Qwen2.5-0.5B-Instruct and a larger reference model on equivalent scene evidence; two formats shortlisted | any PC; the larger model on the RTX |
| A2.4 | A bounded, matched LoRA comparison of direct selection and query parsing | RTX |
| A2.5 | The exported grounder checked against the PC and measured in the Quest app, from typed submission to validated goal, grounding-only (O21) | RTX, then the headset |
| A2.6 | Preliminary Gate B on frozen evaluations, with public data and lab commands reported separately | any PC |

**What A1 gives A2:** candidate scoring is already the "direct selection" design, with confidences; any fine-tuned
model becomes a GGUF for the same runtime, and the headset's command line can check its scores against PyTorch's
(D56) in minutes. **What A1 doesn't give:** any evidence of correctness. A1 checked that the headset reproduces the
model; on its four prompts the model is right on three and wrong on "the box behind the table". Measuring and
improving correctness over many commands is A2's job.

**Where A2 runs:** almost all of it on the lab PC through Remote Desktop, so being in the lab isn't needed. Long jobs
keep running when Remote Desktop is disconnected, but not after signing out or a forced restart. The headset is
needed only for the latency curve (section 8) and A2.5, from whichever PC it is plugged into.

**To settle when A2 first needs the lab PC:** its operating system, CUDA and Python versions and free disk; how models
move between the lab PC and the PC the headset is plugged into (Git carries code, not models); which larger reference
model to use; A0's status.

**Needed from the project lead:** access to the lab PC; the data-access requests, which start with A2 (ScanNet where
the chosen files need it); about an hour of lab members' time for the 50–100 commands, which can be collected
remotely.

## 11. Starting a new chat

Make the zip from a clean, pushed working copy (tracked files only, a few MB):

```
git status
git rev-parse --short HEAD
git archive --format=zip -o "$HOME\Downloads\second-eyes-src.zip" HEAD
```

Open the new chat inside the Claude project, so it sees the proposal and its supplement and can read the chat the
prompt names. Upload the zip and paste the prompt below, with the commit's short hash and that chat's name filled in.

> We're continuing the Second Eyes project. Attached is second-eyes-src.zip, the repository at commit `<short hash>`.
> The proposal, Second_Eyes_Research_Proposal_Revision_3.pdf, and its Revision 3.1 supplement are in this project's
> files.
>
> Read HANDOVER.md first, then check what you need in the files it points to. Don't rely on memory or summaries:
> the repository decides.
>
> My standing constraints, from the start of the project:
>
> 1. Don't assume anything and don't do anything on your own. Propose first and wait for my approval, even for small
>    fixes. Where there's a choice, give me the options with your recommendation. If something is unclear or
>    inconsistent, ask; don't fix it quietly.
> 2. One step at a time. Don't try to finish many steps in one response.
> 3. Don't plan too far ahead. The Go/No-go gates may end or change things, so plan up to the next gate.
> 4. Keep everything easy for me to understand later: clear records and a well-structured repository.
> 5. Every computation on the Quest 3 counts. Don't waste its compute or memory.
> 6. Vicon and participants wait until I actually need them. Storage and experiment tracking wait until headset
>    feasibility is settled.
> 7. If you must look into any chats, just refer to the chat named "<chat name>." Don't look into any other chats
>    because I don't want you to be confused.
>
> Start by telling me briefly what you understood. Then propose the next step and wait for my approval.
