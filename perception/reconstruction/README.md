# quest_recon

Quest-only reconstruction for Second Eyes: single-camera video to a metric, anchored point cloud, computed on the
headset. Separate from Second Eyes Edge.

- `RUNBOOK.md` — one ordered procedure for testing everything together (workstation, laptop, headset)
- `docs/evaluation-spec.md` — the procedure fixed before results (v1.2)
- `docs/decision-record.md` — D131, D132 and open items O30–O40 for `notes/decisions.md`
- `docs/capture-interface.md` — frame interface, calibration, the ChArUco references
- `docs/platform-requirements.md` — what each piece needs
- `configs/references.yaml` — the two boards and the house video's board windows
- `qrecon/` — the desktop harness: sources, keyframes, landmarks, calibration, consistency, fusion, metrics, anchoring
- `questpipe/qrecon_quest.cc` — the same stages for the headset: anchoring, keyframes, depth (ONNX), calibration,
  cross-view filter, bounded fusion
- `tracking/` — the headset tracker (`qtrack.cc`), the stella patch (D132), pins
- `tools/build_tracking.py` — builds both headset programs; `tools/reconstruct_on_quest.py` — runs them on the headset
- `scripts/qrecon.py`, `scripts/video_tools.py`, `scripts/export_dav2_onnx.py` — evaluation, video tools, model export
- `tests/` — 21 tests on synthetic data; 4 run the headset program (built for Linux) against the harness, including
  one anchored reconstruction from tracker units to a metric cloud

Evidence so far: synthetic only.
- **Tested here:** the headset program, built for Linux, matches the harness on generated input, and its board
  anchoring recovers scale within 2%.
- **Not run yet:** the Android build, anything on the headset, and anything on real data.
