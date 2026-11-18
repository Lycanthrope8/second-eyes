# Results-sheet entries, drafted 18 November 2026

For the test guide's results sheet. Values come from the session records; where a record has no value, the entry says
so. Status: **confirmed result** (a check passed as defined), **failed check**, **provisional measurement** (a value
whose basis may still change: superseded intrinsics, frame counts instead of time, a fallback method), **pending** (not
yet measurable or not recorded), **recorded** (a number to keep, with no pass or fail). Coverage figures before R3.14
use nominal timestamps (frame index / frame rate); R3.14 recomputes them from the video's own timestamps (D135).

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 1 | Self-check | `SELFCHECK PASS`: 12 keyframes, scale_error 0.0024, accuracy_median 0.018 m, fscore_5cm 0.915 | confirmed result | workstation, 15 Nov |
| 1 | Tests | laptop, R3.13: 33 passed, 8 skipped (R3.14: 40 passed, 11 skipped expected) | confirmed result | laptop, 17 Nov |
| 1 | PyTorch on the workstation | no value in the records | pending | — |
| 2 | Oracle plumbing (r002) | 86 of 160 keyframes calibrated | provisional measurement | workstation, 15 Nov |
| 2 | DA-V2 against DA3 on fr1 (r006–r010) | no method passes screening on fr1; DA3 worse than DA-V2 on fr1, better on fr3; `enough_valid_keyframes` false for every fr1 arm, cause not investigated | failed check (screening) | workstation, 15 Nov |
| 2 | Model revisions | DA3's fr3 runs used the unpinned default revision, not recorded | pending | — |
| 2b | Export 960×540 | ok; ONNX against PyTorch max_abs 0.00025, median_rel 4.2e-07; sha256 fb0015f4…cafc0fd0b | confirmed result | workstation, 15 Nov |
| 2b | Export 640×480 | result not reported | pending | — |
| 3 | Build | `done ->` (laptop, R3.13) | confirmed result | 17 Nov |
| 3 | `qtrack`, `qrecon_quest` on the headset | both ran in every headset run since 15 Nov (the usage-line check itself was not recorded) | confirmed result | runs r001 onward |
| 3 | No LGPL solver libraries | no value in the records | pending | — |
| 4a | Boards found (house video) | board 01 and board 02 found | confirmed result | 16 Nov |
| 4a | Calibration (house, board 01, 0–20 s) | rms 0.238 px, 40 views, coverage 0.25, distances 0.90–1.74 m, focal 1,217.6 px | provisional measurement, superseded by D134 (1,315 px) | 16 Nov |
| 4a | Calibration (12:04:50, board 01, 0–54 s) | rms 0.224 px, focal 1,637.3 px (std 1.1%), coverage 0.17, distances 1.60–1.97 m | provisional measurement, superseded by D134 | 16 Nov |
| 4a | Tracking frames, board frames (house) | 4,522; 532 | confirmed result | 16 Nov |
| 4a | Extracted frames are the frames named (12:11:18, board frames and 15 fps) | all 12 sampled frames exact, both folders | confirmed result | 18 Nov |
| 4b | Tracked share, house r001 (first minute, 1,217.6 px) | 0.642 of all frames; map from 7.6 s; lost at 46.2 s | failed check (frame counts) | 16 Nov |
| 4b | Tracked share, 12:04:50 r010 (first minute, 1,315 px) | 0.871 of all frames; 1.00 from initialisation (784 tracked, 0 lost, 116 initialising) | provisional measurement: meets D135's criterion on frame counts; the time-based figure is pending | 16 Nov |
| 4b | Anchor, house r002 and r004/r005 (3 px limit) | refused: held-out 8.95 px (r002) | failed check, under the limit D133 replaced | 16 Nov |
| 4b | Anchor, house r006 (1,217.6 px, D133) | ok; corner fit 3.1 mm, held-out 4.6 mm | confirmed result under superseded intrinsics | 16 Nov |
| 4b | Anchor, 12:04:50 r009 (1,637.3 px) | ok; corner fit 6.6 mm, held-out 2.1 mm, 46 frames | confirmed result under superseded intrinsics | 16 Nov |
| 4b | Anchor, 12:04:50 r011 (first minute, 1,315 px) | refused: corner fit 16.4 mm (held-out 5.0 mm) | failed check | 17 Nov |
| 4b | Anchor, 12:04:50 whole walk (1,315 px; r001 strict, r004 best effort) | corner fit 14.0 mm rms (max 39.7 mm); held-out 6.1 mm median; r004 used it with a warning | failed check | 17 Nov |
| 4b | Keyframes calibrated | house r006: 20 of 20; 12:04:50 r009: 20 of 20 | confirmed result | 16 Nov |
| 4b | Depth time per keyframe, median | 5.28 s (r006), 5.35 s (r009), 5.52–5.58 s (r004–r006, 17 Nov) | recorded | — |
| 5 | Coverage, 12:04:50 whole walk (r012, 1,315 px) | whole recording 0.222 (77.0 of 347.5 s); initialisation 8.87 s; from initialisation 0.228; longest gap 75.49–337.09 s (261.6 s, the only one) | provisional measurement (nominal timestamps) | r012, computed 18 Nov |
| 5 | Bedroom (board 01): valid keyframes | 81 of 100 (0.81, r004) | meets the 0.8 criterion; metric assessment pending: its anchor failed its check | 17 Nov |
| 5 | Living/dining (board 02): valid keyframes | 61 of 92 (0.66, r005) | failed check | 17 Nov |
| 5 | Living/dining (board 02): anchor | r005 anchored by board 02 with no failed check (values in its anchor.json); r007 on the end stretch: corner fit 4.8 mm rms, held-out 2.4 mm median | confirmed result | 17 Nov |
| 5 | Both rooms: tape distances, "no wall appears twice" | no reference measurements; no procedure yet (O43) | pending: Level 5 not passed for either room (D137) | — |
| 5 | Board 02 in board 01's frame | 5.19 m, 10.6° (r007: board 01 from single views, camera still; scale taken as 1) | provisional measurement | 17 Nov |
| 5 | Closure (bedroom, r004) | 0.081 m, 31.8°, scale drift −0.079; its anchor failed its check | provisional measurement | 17 Nov |
| 5 | Peak memory, total time | r004 871 MB, 580 s; r005 864 MB, 526 s; r006 889 MB, 488 s | recorded | 17 Nov |
| 5 | Merged cloud `house_both_rooms` | 435,710 points; extent 8.92 × 3.89 × 10.16 m; rooms placed correctly relative to each other (project lead's view) | provisional measurement | 17 Nov |
