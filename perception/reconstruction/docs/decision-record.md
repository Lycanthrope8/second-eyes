# Entries for the Second Eyes decision log (Quest-only path)

For `notes/decisions.md`. The pushed log ends at D130 and O29 (19 October); if you've added entries since, shift the
numbers. Rows follow the log's columns. Decisions go newest first; open items go in the Open table.

## Decisions

| ID | Date | Decision | Alternatives | Reason or evidence | Who decided |
|---|---|---|---|---|---|
| D132 | 2026-11-15 | stella_vslam is built without LGPL code: its two uses of g2o's CSparse solver (global bundle adjustment, loop-closure pose graph) are switched to g2o's Eigen solver, which stella already uses in its other three optimisers (`tracking/patch_stella.py`); g2o is built without CSparse, CHOLMOD, its viewer and other LGPL or GPL parts. Eigen 3.4.0 (MPL-2.0) is approved as a D83 exception; its sparse ordering code carries an MPL-2.0 relicensing note in 3.4.0. | A: approve LGPL CSparse as a separate shared library, unpatched stella · B as chosen | No LGPL in the build, and no cross-build of SuiteSparse for Android. The normal equations and their solution are unchanged; only the factorisation library differs. Patched files compile against stella 0.7.0 and g2o 20230223_git; tracking accuracy is checked by the public-sequence runs. Claude recommended A first, then changed to B after inspecting the build | Claude, on the project lead's delegation ("do what's in the best interest of our research", 15 Nov) |
| D131 | 2026-11-14 | Quest-only reconstruction of drone video, provisional evaluation decision: MASt3R-SLAM stays on the workstation as a comparison, never ground truth, not ported; tracking is stella_vslam on the Quest CPU anchored to the room reference, with flow and range telemetry as cross-checks; dense depth is Depth Anything V2 Small (baseline) against DA3-Small on 4- and 8-keyframe windows with supplied poses (challenger), ZipDepth as the resource fallback; RD-VIO only after camera and motion-sensor timing is shown usable; geometry comparison first, then a Quest resource test for each method with useful geometry, integration last; evaluation per `docs/evaluation-spec.md`, frozen before validation; 3% and 5 cm are screening limits; no flight-planning space is cleared by this work | Port MASt3R-SLAM; RD-VIO first; DA3 only | GPT Astra's recommendations and corrections, 11–14 Nov | project lead, 14 Nov |

## Open

| ID | Question | Needed by | Options so far |
|---|---|---|---|
| O30 | DA3-Small's parameter count: model card, paper and checkpoint disagree | before any size claim | count the loaded model (the harness records it) |
| O31 | `depth-anything.cpp` can't take camera poses as input | DA3's Quest resource test | extend the port · another export that keeps pose conditioning |
| O32 | The house video's references: board layout, dictionary and measured square and marker lengths for ChArUco 01 and 02 (`configs/references.yaml`) | anchoring and calibrating the house video | measured by the project lead; `video_tools.py charuco-inspect` lists the dictionary and IDs |
| O33 | Camera and motion-sensor timing on the drone | before RD-VIO is considered | measure offset and drift on the CM4 setup |
| O34 | A SpatialLM-Lite distilled from SpatialLM 1.1's non-commercial weights needs its own D83 decision | before Lite work starts | to be decided |
| O35 | Headset limits for memory and rendering in the resource tests | the first Quest resource test | reuse Gate A's definitions (O19) · set them for this work |
| O36 | Which Revision 3 phase this work's runs belong to (D5) | the first registered run | A6, drone integration and registration (proposed) · 2A.1 · another |
| O37 | Where the work lives in the repository (D2) | before the first commit | `perception/reconstruction/` (proposed) · elsewhere |
| O38 | How the depth model runs on the headset | the app integration | ONNX Runtime 1.29.0 (MIT) on the CPU, as the command-line pipeline does now · Unity Inference Engine 2.2.1 on the GPU inside the app |
| O39 | Board 02's print scale is not measured; its scale-drift figure includes any print difference from board 01 | reading the house result | measure one printed square on board 02 |
| O40 | Whether the boards hang upright and level | a z-up cloud for SpatialLM | yes: set `mounting: upright_level` · no: take the up direction from the floor plane (not yet written) |
