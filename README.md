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
| Phase | S0 closed on 2026-09-27 (`notes/phases/S0_infrastructure.md`); next phase not started |
| Done | S0 items 1–3: repository, run registry, event log format |
| Deferred | S0 item 4 (time sync) until Vicon is used; items 5–6 (storage, experiment tracking) until after Gate A |
| Next | not decided yet: headset prep step 2, A0 or A1 |
| Gates passed | none yet |

Update this table whenever a step finishes.
