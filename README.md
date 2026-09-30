# Second Eyes

A micro-UAV pre-scans a workspace the user has not seen. The Meta Quest 3 turns what the drone saw into a semantic memory, and the user tasks the drone in natural language about things only the drone has seen. All AI inference runs on the headset at deployment.

The plan is the research proposal, Revision 3 (September 2026). Phase IDs (S0, A1, B1, …) and gates (A–F) in this repository refer to it.

## Start here

1. This README: the map and the current status.
2. `docs/conventions.md`: where things go and the rules.
3. `notes/decisions.md`: what is decided and what is still open.
4. The newest note in `notes/phases/`: what happened most recently.

## Map

```text
second-eyes/
├── README.md          start here
├── docs/              how things work now (edited in place)
│   ├── conventions.md   where things go and the rules
│   ├── logging.md       event log format
│   └── setup/quest3.md  headset record: OS build, changes, how to undo
├── notes/             what happened (dated, append-only)
│   ├── phases/          one note per phase
│   ├── decisions.md     open questions and decisions taken
│   ├── limitations.md   running list
│   └── failures.md      running list
├── runs/              run registry, one folder per run (S0.2)
├── schemas/           file formats all components follow (S0.2, S0.3)
├── tools/             command-line helpers: creating and checking runs (S0.2)
├── quest-app/         Unity app for the Quest 3 (A1)
├── perception/        detection, ranging, association, triangulation (A4)
├── grounding/         scene schema, relations, language models (A2)
├── drone-bridge/      drone link, video, scan routine, registration (A6)
├── analysis/          reading logs, aligning clocks, plots (S0.3)
└── .gitignore
```

The phase in parentheses is when a folder starts to fill; until then it holds only its README.

## Current status

| | |
|---|---|
| Phase | A1 · headset feasibility (started 2026-09-27) |
| Done | S0 (closed 2026-09-27, `notes/phases/S0_infrastructure.md`); A1.1 headset baseline (`20260927_A1_r001`); A1.3a empty passthrough app (`20260928_A1_r002`); A1.3b overlay and 72 Hz request (`20260928_A1_r003`); A1.4a headset event log (`20260928_A1_r004`); A1.4b log pull (`20260928_A1_r005`); A1.5 hand tracking and stop button (`20260928_A1_r006`); A1.6a measuring tools (trial `20260928_A1_r007`); O5 closed, no headset changes (D26); A1.7a starting point found in Meta's docs and source (`docs/meta-ai.md`); A1.7b PC reference (`20260928_A1_r008`); A1.7c-1 model in Unity (`qwen2.5-0.5b-instruct-72303ef1-f16.sentis`, 1.26 GB; D34–D38); A1.7c-2 the model answers the fixed prompt exactly like the PC reference, on the PC (`20260929_A1_r014`) and on the headset (`20260929_A1_r015`), after D47 |
| Deferred | A1.6's trial and reference measurement, replaced by A1.7d (D25); S0 item 4 (time sync) until Vicon is used; S0 items 5–6 (storage, experiment tracking) until after Gate A |
| Next | A1.7d: what the model costs on the headset, six runs (D50, `docs/profiling.md`); then A1.8, which settles the runtime A2 needs |
| Gates passed | none yet |

Update this table whenever a step finishes.
