# Second Eyes

A micro-UAV pre-scans a workspace the user has not seen. The Meta Quest 3 turns what the drone saw into a semantic memory, and the user tasks the drone in natural language about things only the drone has seen. All AI inference runs on the headset at deployment.

The plan is the research proposal, Revision 3 (September 2026), with its Revision 3.1 supplement (October 2026, `docs/plan/`), which replaces the A2 card and every section A2 touches. Phase IDs (S0, A1, B1, …) and gates (A–F) in this repository refer to them.

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
├── analysis/          reading logs, aligning clocks, plots (S0.3); token audits (A2.1d)
└── .gitignore
```

The phase in parentheses is when a folder starts to fill; until then it holds only its README.

## Current status

| | |
|---|---|
| Phase | A1 · headset feasibility (started 2026-09-27): A1.1–A1.8 done, A1.9–A1.11 open · A2 · grounding baselines (starting) |
| Done | S0 (closed 2026-09-27, `notes/phases/S0_infrastructure.md`); A1.1 headset baseline (`20260927_A1_r001`); A1.3a empty passthrough app (`20260928_A1_r002`); A1.3b overlay and 72 Hz request (`20260928_A1_r003`); A1.4a headset event log (`20260928_A1_r004`); A1.4b log pull (`20260928_A1_r005`); A1.5 hand tracking and stop button (`20260928_A1_r006`); A1.6a measuring tools (trial `20260928_A1_r007`); O5 closed, no headset changes (D26); A1.7a starting point found in Meta's docs and source (`docs/meta-ai.md`); A1.7b PC reference (`20260928_A1_r008`); A1.7c-1 model in Unity (`qwen2.5-0.5b-instruct-72303ef1-f16.sentis`, 1.26 GB; D34–D38); A1.7c-2 the model answers the fixed prompt exactly like the PC reference, on the PC (`20260929_A1_r014`) and on the headset (`20260929_A1_r015`), after D47; A1.7d what the model costs (`notes/phases/A1.7d_model_cost.md`, runs `20260929_A1_r017`–`20260930_A1_r022`); A1.8a only the CPU with 16-bit weights loads (D54); A1.8b llama.cpp on the headset from the command line, passing (`notes/phases/A1.8b_llama_cli.md`, D56); A1.8c llama.cpp inside the app, 72.5 fps while answering (`notes/phases/A1.8c_llama_app.md`, D58, D59); A1.8d llama.cpp is the runtime (D60) |
| Deferred | A1.6's trial and reference measurement, replaced by A1.7d (D25); S0 item 4 (time sync) until Vicon is used; S0 items 5–6 (storage, experiment tracking) until after Gate A |
| Next | A2 · grounding baselines (D63): A2.1a, the scene contract and its offline validator (D66), A2.1b, the relation library without directions (D67), A2.1c, the directional relations (D68), A2.1d, the offline serializer of both candidate formats (D69, with D70's category-map correction), and A2.1e, the offline structured resolver (D71, with D72's integer-validation correction), are in place; the serializer, its D70 correction, the resolver and its D72 correction are accepted on Windows. A2.2a, the IRef-VLA metadata adapter with scene format v2 (D74), is accepted on Windows and committed as `4d847ee`. A2.2b, the text-only rules baseline on the pinned sample (D75), and its validation correction (D76) passed Windows acceptance and are committed as `22ec822`. A2.2c, the category-complete selection audit (D77), is accepted on Windows and committed as `c5b7704`. A2.2d, the materialized model inputs and their token measurement (D78), is accepted on Windows. A2.3a, the bounded zero-shot direct-selection pilot (D81), passed its real-model run (`20261005_A2_r003`, lab RTX PRO 6000); its code is committed as `33478f3` and its run records as `e71f8a5` and `06d4828`. A2.3b, saved-pilot scoring with a matched rules baseline (D82), is accepted on Windows: the model's agreement with the source targets did not exceed always choosing B, so the pilot does not yet demonstrate spatial reasoning; ChatGPT recommends a controlled alias and ordering check (A2.3c), which awaits its brief. A1.10a, passthrough camera access and its cost, is done; A1.10b and A1.10c, the detector package with its PC reference and the detector in the app (D83 to D87), work on the headset and match the PC exactly; run back to back, the GPU detector cost about one stale frame per inference, so D88 slices each inference over several frames. Fixed 8, 16 and 32 steps per frame each failed the joint frame and latency criteria (r043), so D89 tries one profiled, cost-balanced schedule before A1.10d; CPU sequential operation is the fallback. Then A1.10, then A1.11; A1.9 (speech) last, with typed text commands until then. `HANDOVER.md` has the state of the project |
| Gates passed | none yet |

Update this table whenever a step finishes.
