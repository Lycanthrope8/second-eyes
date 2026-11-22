# Results-sheet entries, drafted 18 November 2026 (runs of 18 November added at the end)

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
| 5 | Coverage, 12:04:50 whole walk (r012, 1,315 px), nominal timestamps | whole recording 0.222 (77.0 of 347.5 s); initialisation 8.87 s; from initialisation 0.228; longest gap 75.49–337.09 s (261.6 s, the only one) | superseded by the source-timestamp row below | r012, computed 18 Nov |
| 5 | Bedroom (board 01): valid keyframes | 81 of 100 (0.81, r004) | meets the 0.8 criterion; metric assessment pending: its anchor failed its check | 17 Nov |
| 5 | Living/dining (board 02): valid keyframes | 61 of 92 (0.66, r005) | failed check | 17 Nov |
| 5 | Living/dining (board 02): anchor | r005 anchored by board 02 with no failed check (values in its anchor.json); r007 on the end stretch: corner fit 4.8 mm rms, held-out 2.4 mm median | confirmed result | 17 Nov |
| 5 | Both rooms: tape distances, "no wall appears twice" | no reference measurements; no procedure yet (O43) | pending: Level 5 not passed for either room (D137) | — |
| 5 | Board 02 in board 01's frame | 5.19 m, 10.6° (r007: board 01 from single views, camera still; scale taken as 1) | provisional measurement | 17 Nov |
| 5 | Closure (bedroom, r004) | 0.081 m, 31.8°, scale drift −0.079; its anchor failed its check | provisional measurement | 17 Nov |
| 5 | Peak memory, total time | r004 871 MB, 580 s; r005 864 MB, 526 s; r006 889 MB, 488 s | recorded | 17 Nov |
| 5 | Merged cloud `house_both_rooms` | 435,710 points; extent 8.92 × 3.89 × 10.16 m; rooms placed correctly relative to each other (project lead's view) | provisional measurement | 17 Nov |

## Runs of 18 November (R3.14: map restarts, automatic board windows)

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 5 | Coverage, 12:04:50 whole walk, one map (r012), source timestamps | whole recording 0.222 (77.0 of 347.5 s); initialisation 8.95 s; from initialisation 0.228; longest gap 75.55–337.10 s (261.5 s, the only one) | recorded | 18 Nov |
| 4b/5 | Coverage, 12:04:50 with map restarts (r004) | whole recording 0.804 (279.3 of 347.5 s); initialisation 6.95 s; from initialisation 0.820; longest gap 75.82–100.09 s (24.3 s); 7 gaps; 4.8 s tracked in maps stella discarded | failed check against Level 4b's 0.9 from initialisation (D135) | 18 Nov |
| 4b/5 | Coverage, 12:11:18 with map restarts (r001) | whole recording 0.696 (255.9 of 367.5 s); initialisation 10.13 s; from initialisation 0.716; longest gap 176.27–208.74 s (32.5 s); 7 gaps; 10.3 s tracked in maps stella discarded | failed check against 0.9 (D135) | 18 Nov |
| 5 | Tracked time in maps that saw a board | 12:04:50: 189.8 of 279.3 s (maps 0, 3, 6, 8); 12:11:18: 96.2 of 255.9 s (maps 0, 11). The rest is in maps that never saw a board and cannot be placed (O42) | recorded | 18 Nov |
| 5 | Anchors (best effort) | passed: r006, r007, r008. Failed check: r002 (corner fit); r003 (corner fit; board 02's viewing split into two windows); r005 (held-out views; board 01's viewing split into two windows) | failed check for r002, r003, r005 | 18 Nov |
| 5 | Board link from r003 | board 01 at 2.61 m from board 02 (single views, spread 7.1 mm), against 5.19 m from r007 of 17 Nov; r003's own anchor failed its check | failed check (the links disagree) | 18 Nov |
| 5 | Valid keyframes per reconstruction | r002 47/54 (0.87), r003 41/60 (0.68), r005 5/49 (0.10), r006 35/60 (0.58), r007 11/21 (0.52), r008 14/21 (0.67) | recorded | 18 Nov |
| 5 | Bedroom (board 01), merge `house_all` | valid keyframes 66 of 124 (0.53); one of three anchors passed | failed check (0.8); metric assessment pending (D137) | 18 Nov |
| 5 | Living/dining (board 02), merge `house_all` | valid keyframes 87 of 141 (0.62); two of three anchors passed | failed check (0.8); metric assessment pending (D137) | 18 Nov |
| 5 | Merged cloud `house_all` | 258,351 points; extent 7.59 × 3.75 × 9.64 m; one block is visibly tilted against the rest in a side view | provisional measurement | 18 Nov |
| 5 | Time and memory on the headset | depth 5.21–5.23 s per keyframe (median); reconstruction peak 829–861 MB; tracking peak 138–162 MB; tracking at capture pace (370 s and 350 s wall) | recorded | 18 Nov |

## Runs of 18 November, R3.15 (best-window anchoring), reusing the same tracking runs

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 5 | Anchor, 12:11:18 map 11 (r010) | board 02, merged window 330.1–338.1 s: passed (no failed check) | confirmed result | 18 Nov |
| 5 | Anchor, 12:04:50 map 0 (r011) | board 01, all windows together: held-out views disagree; valid keyframes 5 of 49 (0.10), as with R3.14 | failed check; cause not found | 18 Nov |
| 5 | Links between board 01 and board 02 | r010 (18 Nov): 2.72 m; r007 (17 Nov): 5.19 m. They differ by 2.90 m and 5.6°. Both come from one map tracked from board 02 to board 01 (about 26 s of walking), so tracker scale drift on the way is the likely cause; the merge used r007's | failed check (the links disagree) | 18 Nov |
| 5 | Merged cloud `house_all2` | 257,381 points; extent 7.86 × 3.86 × 9.64 m; valid keyframes 0.53 (board 01's room) and 0.62 (board 02's room) | failed check (0.8) for both rooms; metric assessment pending (D137) | 18 Nov |

## Placement by shared views, R3.16 (D139)

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 1 | Tests (laptop, R3.16) | 46 passed, 13 skipped, 1 failed: the synthetic placement's scale came out 3.0% off against a 3% limit, because ORB keypoints differ slightly between OpenCV builds (under 3% in Claude's sandbox). The limit is 5% from R3.17 | failed check, test corrected | 18 Nov |
| 5 | Board-less maps placed | 5 of 9 in three rounds: h121118 maps 5, 7 and 10, h120450 maps 2 and 5, all in board 02's frame; located frames agree within 5.7–8.5 cm rms and 0.8–3.4° rms. Not placed: h121118 maps 1, 4 and 8, h120450 map 1 (none of their frames matched a reference) | recorded | 18 Nov |
| 5 | h121118 map 5 (r015) | scale 24.3; 18 of 60 keyframes usable; part of it lands about 12 m east and 7.5 m north of the rest, so the tracker's scale drifted inside that map. R3.17 checks the camera speed a placement implies and trims or refuses such maps | failed check | 18 Nov |
| 5 | Usable keyframes of placed maps | r015 0.30, r016 0.57, r017 0.54, r018 0.92, r019 0.77 | recorded | 18 Nov |
| 5 | Merged cloud `house_all3` | 381,742 points; extent 13.83 × 3.86 × 16.56 m, the misplaced part included; the placements mostly add the living/dining area | provisional measurement | 18 Nov |

## Placement with the speed check, R3.17 (`--redo --include-failed-anchors --replace-failed`)

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 5 | h121118 map 5 | same fit (scale 24.3); the whole map would imply impossible speeds, so only 159.3–176.1 s were kept (0.9 m/s); r020: 25 of 30 keyframes usable (r015 had 18 of 60); the far-off part is gone | recorded | 18 Nov |
| 5 | h121118 map 7 | whole map kept (0.16 m/s); r021: 25 of 42 keyframes usable | recorded | 18 Nov |
| 5 | h120450 map 0 (bedroom) | its failed board anchor replaced by shared views with h121118 map 0: 41 of 60 located frames agree within 8.6 cm rms and 3.6° rms; r022: 53 of 60 keyframes usable, 172,764 points (r011 had 5 of 49) | recorded | 18 Nov |
| 5 | No longer placed | h121118 map 10 and h120450 map 5 (their R3.16 placements went through map 5's drifted part); h120450 map 2 (located frames span 0.25–0.29 m, under the 0.3 m that fixes the scale). Seven maps unplaced in all | recorded | 18 Nov |
| 5 | Bedroom (board 01), `house_all4` | usable keyframes 114 of 135 (0.84) | meets the 0.8 criterion; metric assessment pending (D137, O43) | 18 Nov |
| 5 | Living/dining (board 02), `house_all4` | usable keyframes 138 of 213 (0.65) | failed check (0.8) | 18 Nov |
| 5 | Agreement where runs overlap, `house_all4` | bedroom: r014 vs r009 6.5 cm median (76% of points within 30 cm), r014 vs r022 6.1 cm, r022 vs r009 14.7 cm (89%); living/dining: r020 vs r012 17.4 cm (65%), r021 vs r012 21.1 cm (64%), r013 vs r012 16.7 cm (39%); floor heights differ by up to about 30 cm between runs | provisional measurement (consistency between runs, not accuracy) | Claude, 18 Nov |
| 5 | Merged cloud `house_all4` | 476,824 points; extent 8.78 × 3.89 × 10.43 m; nothing far outside the house | provisional measurement | 18 Nov |
| 5 | Board links | still 2.72 m (r010) against 5.19 m (r007, used); r010's own reconstruction lies mostly between the rooms | failed check (the links disagree) | 18 Nov |

## The 10 Nov house video and the three-recording merge (R3.18), 21 November

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 4b/5 | Coverage, house video h191353 (r001) | whole recording 0.771 (232.5 of 301.5 s); initialisation 7.74 s; from initialisation 0.792; longest gap 162.81–182.68 s (19.9 s); 8 gaps; 4.0 s tracked in discarded maps; 9 maps saved, 2 discarded | failed check against 0.9 (D135) | 21 Nov |
| 5 | House video, maps with a board | map 0 (bedroom): r002, board 01, no failed check, 50 of 52 keyframes usable. Map 9 (living/dining): r004, board 02, no failed check, 20 of 30. Map 10: r005, board 01 seen only at the end of a 56 s map; 2,540 points that land 11–21 m from the house (scale drift inside the map). Map 5: no board fit, left out | confirmed result for r002 and r004; failed check for r005 and map 5 | 21 Nov |
| 5 | Placement across the three recordings | 3 more maps placed by shared views: house video maps 3 and 4, 12:04:50 map 2; 9 maps remain unplaced | recorded | 21 Nov |
| 5 | A third board link | none: no house-video map saw both boards (maps 9 and 10 are separated by a 2.4 s loss) | recorded | 21 Nov |
| 5 | Which board link | with the 17 Nov link (5.18 m) about 19% of the living/dining points stand on the bedroom's floor; with the 18 Nov link (2.72 m) 41%, and living-room walls run through the bedroom in a top view | provisional measurement; the project lead's check by eye is awaited | Claude, 21 Nov |
| 5 | Rooms, `house_all5` | bedroom 204 of 247 keyframes usable (0.83); living/dining 204 of 321 (0.64) | bedroom meets 0.8, living/dining fails; metric assessment pending (D137) | 21 Nov |
| 5 | Merged cloud `house_all5` | 619,732 points; extent 13.96 × 4.92 × 23.63 m, stretched by r005 | provisional measurement | 21 Nov |
| 5 | Time | house video (5 min 1 s): 23 min to process; placement across three recordings about 10 min | recorded | 21 Nov |

## Speed check on board-anchored maps, R3.19, 21 November

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 5 | House video map 10 | its board anchor implied 2.4 m/s over the whole map; rebuilt for 287.5–301.4 s (r009): 0.65 m/s, 20 of 25 keyframes usable, 70,663 points (r005 had 2,540 points 11–21 m away) | recorded | 21 Nov |
| 5 | Other board-anchored maps | implied speeds 0.29–1.34 m/s; none trimmed | recorded | 21 Nov |
| 5 | Bedroom agreement across recordings, `house_all6` | house video map 0 against 12:11:18 map 0, 12:04:50 maps 0 and 8: 4.6, 6.6 and 7.7 cm median (60–85% of points within 30 cm); trimmed map 10 against map 0: 5.2 cm | provisional measurement (consistency between runs, not accuracy) | Claude, 21 Nov |
| 5 | Living/dining agreement, `house_all6` | house video map 9 against 12:04:50 map 3, both anchored by board 02: 21.1 cm median, 27% of points within 30 cm | provisional measurement | Claude, 21 Nov |
| 5 | Rooms, `house_all6` | bedroom 184 of 212 keyframes usable (0.87); living/dining 204 of 321 (0.64) | bedroom meets 0.8, living/dining fails; metric assessment pending (D137) | 21 Nov |
| 5 | Merged cloud `house_all6` | 687,855 points; extent 8.78 × 3.89 × 10.43 m; floor overlap with the 17 Nov link 19%, with the 18 Nov link 42% | provisional measurement | 21 Nov |

## Comparison with the MASt3R-SLAM reference (D141), 21 November

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 5 | Reference | `registered.ply`, 18,590,183 points; thinned to 2 cm cells 1,699,837 points and to 5 cm cells 219,464, both keeping all of its surfaces within 5 cm; already in board 01's frame (refinement 0.76°, 4.6 cm) | recorded | Claude, 21 Nov |
| 5 | `house_all6` against the reference | median 12.9 cm; precision 0.32 and 0.45, recall 0.21 and 0.34, F-score 0.25 and 0.39 at 5 and 10 cm | provisional measurement (comparison, not ground truth) | Claude, 21 Nov |
| 5 | Per map | best: house video map 0 and 12:04:50 map 8, 2.9 cm median (83–84% within 10 cm); house video map 10 4.5 cm; 12:11:18 map 5 5.8 cm; house video map 9 7.8 cm. Worst: 12:11:18 map 7 37.9 cm, 12:11:18 map 0 36.5 cm (its anchor failed the corner-fit check), 12:04:50 map 0 25.7 cm (placed against 12:11:18 map 0 with `--include-failed-anchors`) | provisional measurement | Claude, 21 Nov |
| 5 | Without the failed-anchor map and the map placed through it | median 5.7 cm; precision 0.64, recall 0.33, F-score 0.44 at 10 cm | provisional measurement | Claude, 21 Nov |
| 5 | Placement of the two rooms | fitted room by room to the reference, the bedroom maps move 16 cm and 4.0°, the living/dining maps 7 cm and 2.5° | provisional measurement; supports D140 | Claude, 21 Nov |

## Tracking presets (R3.20 sweep) and the keyframe cap, 21 November

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 4b | 12:11:18, four presets (maps saved, discarded, breaks, tracked share whole and from initialisation, longest map) | baseline 9, 2, 8, 0.744, 0.764, 65.4 s; fps30 9, 1, 8, 0.787, 0.806, 65.0 s; fps30_features 7, 0, 6, 0.892, 0.909, 109.9 s; fps30_features_keep 9, 0, 8, 0.849, 0.864, 107.6 s | provisional measurement (one run each; runs vary) | 21 Nov |
| 4b | Tracking cost, fps30_features | 22.6 ms per frame median, 37.4 ms at the 95th percentile (33.3 ms between frames); 386 s for the 367.5 s video; peak 230 MB | recorded | 21 Nov |
| 5 | Keyframe cap | the headset program takes keyframes in time order and stops at the cap, so a reconstruction capped at 60 keyframes (at most two a second) covered only the first part of a map longer than about 40 s; four runs in `house_all6` stopped at 60 (r005, r010, r012, r022). From R3.21, paths placed by a fixed similarity, including every one-map run, get their keyframes spread over the whole path | failed check, fixed for fixed-similarity paths | Claude, 21 Nov, from the program's code |

## First one-map run, 12:11:18 (R3.21), 22 November

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 5 | Pieces (fps30_features tracking, r012) | 7. Map 5 anchored by board 02 (188.6–197.1 s, checks passed); map 0: no board fit; map 6 (110 s): board 02 at 276.1–280.6 s, corner fit failed, its path implying 2.95 m/s (kept 266.1–348.1 s); maps 1–4: no board | recorded | 22 Nov |
| 5 | Join | 3 pieces in one map, 134 s of the 367 s video (37%): map 5 (board), map 0 by shared views with the house video's map 0 (14 of 17 located frames, 4.2 cm rms), map 4 by shared views with the house video's map 4 (trimmed to 159.6–176.2 s). Left out: map 6 (located frames spanned only 0.28 m, under the 0.3 m that fixes scale, and its anchor failed), maps 1–3 (no frame matched a placed piece) | failed check against one map for the whole video (D142) | 22 Nov |
| 5 | One reconstruction (r005) | 200 keyframes spread over 205 candidates, 113 usable (0.565); 162,889 points; 1,016 s on the headset; peak 898 MB | recorded | 22 Nov |
| 5 | Against the MASt3R-SLAM reference | the 12:11:18 one map: median 13.5 cm, 42% within 10 cm; `house_one` (with the other two recordings' earlier maps): median 11.7 cm, F-score 0.38 at 10 cm | provisional measurement (comparison, not ground truth) | 22 Nov |
| 5 | A third board link | map 6's anchor-only run saw board 01 at 4.19 m from board 02 (single views, from an anchor whose corner fit failed); not used (D140) | recorded | 22 Nov |
| — | House video and 12:04:50 | tracking stopped before it started: the foreground-app probe failed because no app was resumed on the headset; fixed in R3.22 (the probe no longer stops a run) | failed check, fixed | 22 Nov |

## One map per video for all three recordings (R3.22), 22 November

| Level | Check | Value | Status | Source |
|---|---|---|---|---|
| 4b | Tracking with fps30_features | house video: 0.843 of the recording (0.859 from initialisation), 7 maps; 12:04:50: 0.882 (0.890), 7 maps; 12:11:18 (sweep run): 0.892 (0.909), 7 maps | provisional measurement | 22 Nov |
| 5 | One map per video | house video: 4 pieces, all anchored by boards whose checks passed, 173 s (57%); 12:11:18: 4 pieces (board 02, and three by shared views), 279 s (76%); 12:04:50: 5 pieces (three boards, two by shared views), 284 s (82%) | recorded | 22 Nov |
| 5 | Left out | house video maps 1, 4, 6 (82 s); 12:11:18 maps 1–3 (50 s); 12:04:50 maps 1–2 (26 s): no frame matched a placed piece (O44) | recorded | 22 Nov |
| 5 | One reconstruction per video | 200, 192 and 192 keyframes; usable 0.60, 0.43 and 0.46 (the keyframes, spread evenly over twice as many candidates, were too far from their neighbours for depth calibration; grouped from R3.23); about 17 min each on the headset; peak 873–918 MB | recorded | 22 Nov |
| 5 | Against the MASt3R-SLAM reference, inside its own extent | `house_one2`: median 6.9 cm, precision 0.60, recall 0.26, F-score 0.36 at 10 cm, with 23% of its points outside the reference's extent; `house_all6`, the 14-map merge, on the same terms: median 11.6 cm, precision 0.47, recall 0.34, F-score 0.39, 5% outside | provisional measurement (comparison, not ground truth) | Claude, 22 Nov |
| 5 | Points outside the reference's extent | mostly beyond its far end (x 7–10 m, z −7 to −11 m in board 01's frame), at heights from the floor up; 45% of 12:04:50's outside points lie within 15 cm of the house video's; part may be space the reference does not cover, part drift in long pieces | provisional measurement | Claude, 22 Nov |
| 5 | Aligning the three videos to each other by their overlap | median 15.0 → 13.9 cm (5 cm cells): the remaining error is mostly inside each video | provisional measurement | Claude, 22 Nov |
| 5 | A fourth board link | 12:04:50 map 9 saw board 02 at 6.08 m from board 01 (board fit 1.5 mm); the links now read 2.72, 4.19, 5.18 and 6.08 m; D140 stands | recorded | 22 Nov |
