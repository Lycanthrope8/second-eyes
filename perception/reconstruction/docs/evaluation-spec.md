# Quest-only reconstruction: evaluation specification v1.2

Second Eyes, Quest-only path. v1 of 14 November, revised twice on 15 November (changes at the end). Separate from Second Eyes
Edge and its workstation prototype.

This document fixes how the reconstruction of drone video is evaluated **before any result is examined**. It is
frozen by `scripts/qrecon.py freeze`, which records its hash together with the configuration files; the harness
refuses to run a validation sequence if either has changed since.

## 1. What is being decided

How the Quest turns a single-camera video into camera poses in metres and a point cloud that SpatialLM, and later
a compressed SpatialLM, can read. The provisional decision (see `decision-record.md`):

| Role | Choice |
|---|---|
| Tracking | stella_vslam on the Quest CPU, patched to drop LGPL code (D132), anchored to the room reference: a printed AprilTag or ChArUco board of measured size |
| Dense depth, baseline | Depth Anything V2 Small, one keyframe at a time |
| Dense depth, challenger | DA3-Small on overlapping groups of keyframes, given the tracker's poses |
| Dense depth, resource fallback | ZipDepth |
| Workstation comparison | MASt3R-SLAM. A comparison only, never ground truth, and not ported |
| Tracking upgrade, later | RD-VIO through XRSLAM, only after camera and motion-sensor timing is shown to be usable |

## 2. Stages

Every run uses the same stages; only the source of poses and the depth method change.

1. **Source.** Frames with timestamps and intrinsics (`qrecon/sources.py`): a public sequence, a replayed video
   file, later the CM4 stream.
2. **Poses.** Reference poses, or a tracker's trajectory. Tracking runs on every frame; mapping uses keyframes.
3. **Keyframes.** One rule for every method: a new keyframe after 0.10 m of translation or 10° of rotation, and
   never sooner than 0.5 s after the last.
4. **Depth.** The method under test, on the keyframes.
5. **Metric calibration.** Sparse landmarks triangulated from the poses in use.
   - Depth Anything V2 returns a relative, disparity-like value `d`. Scale and shift are fitted in inverse depth,
     `1/z = a·d + b`, robustly, and tested on landmarks kept out of the fit.
   - DA3 returns depth already scaled to the supplied poses by its official API. One robust scale against the
     landmarks is computed and reported; it is not applied in the primary arm.
   - A keyframe without enough landmark support is rejected, not guessed.
6. **Consistency.** Depth edges are removed; a depth value is kept only if a neighbouring keyframe agrees within 5%.
7. **Fusion.** One bounded voxel map (2.5 cm), a voxel counting only when seen at least twice.

Every stage exists twice, with the same parameters: in the desktop harness (`qrecon/`, Python) and in the headset
program (`questpipe/qrecon_quest`, C++). Their agreement is tested on synthetic data before either is trusted.

Occupancy for flight planning is a separate output with stricter criteria and is **not** covered by this
specification; nothing here clears space for flight.

## 3. Pinned components

| Component | Pin | Licence |
|---|---|---|
| Depth Anything V2, code | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` | Apache-2.0 |
| Depth Anything V2 Small, weights | `depth_anything_v2_vits.pth`; Hugging Face revision and SHA-256 recorded at download | Apache-2.0 (Small only) |
| Depth Anything 3, code | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | Apache-2.0 |
| DA3-Small, weights | `depth-anything/DA3-SMALL`; revision recorded at download (Astra reported `e08cab6`) | Apache-2.0 |
| stella_vslam | tag `0.7.0` (`525231147319bcd31242f981078c36d9272727b4`), patched by `tracking/patch_stella.py` (D132). Never a version before 0.3 | BSD-2-Clause |
| g2o | `20230223_git` (`e8df2004e07ea8f5b8e6a8b9f2dc067b45b45036`), built without CSparse, CHOLMOD, viewer and other LGPL/GPL parts | BSD |
| Eigen | 3.4.0 | MPL-2.0 (D83 exception, D132) |
| yaml-cpp | `yaml-cpp-0.7.0` (`0579ae3d976091d7d664aa9d2527e0d0cff25763`) | MIT |
| OpenCV (headset) | Android SDK 4.10.0, prebuilt | Apache-2.0 |
| SQLite | 3.46.1 amalgamation | public domain |
| FBoW and its ORB vocabulary | FBoW as stella's submodule; vocabulary `708407ae4cd59219b996eda7ed1f4562c7ae106a` | MIT |
| ZipDepth | tag `v1.0` (`91f3fd21e131641f51e8d35736d1958350180e3a`) | MIT |

Downloads without a pinned hash have their SHA-256 recorded in `tracking/out/android-arm64/build.json` on first build;
they are pinned from there.

For every model the run manifest records: the file's SHA-256, the revision, and **the parameter count of the
loaded model**. The model card, the paper and the checkpoint disagree on DA3-Small's size, so no size is stated
until that count exists.

Before the freeze, `arms.da3s.revision` in `configs/qrecon.yaml` is set to the recorded revision.

## 4. Runs

| Run | Poses | Depth | Isolates |
|---|---|---|---|
| A | reference | measured depth | The reference surface, and whether the scene is readable at all |
| B-dav2s | reference | DA-V2 Small | Depth and calibration with pose error removed |
| B-da3s_w4, B-da3s_w8 | reference | DA3-Small, windows of 4 and 8 keyframes, stride half the window | The challenger under the same conditions |
| C-… | tracker | the same three | Tracking and reconstruction error together |
| D-zipdepth | tracker | ZipDepth | Only if a resource test rejects both other methods |
| oracle | reference | reference depth, distorted | Plumbing check. Never a result |

A keyframe that appears in two DA3 windows takes the prediction from the window where it is nearest the centre.

## 5. Data and their roles

| Data | Role | Reference |
|---|---|---|
| TUM RGB-D `freiburg3_long_office_household` | **Development.** Parameters may be tuned here | Depth and motion-capture trajectory |
| TUM RGB-D `freiburg1_room` | **Validation.** Runs only after the freeze | Depth and motion-capture trajectory |
| Your house video (Galaxy S26 Ultra, 1080p, 30 fps, 5:40, focus and stabilisation locked) | Replay input for tracking, depth, fusion and resource use; camera calibration from its board views | ChArUco 01 at the start sets metric scale; ChArUco 02 in the second room measures scale drift and the room link. No dense reference |
| Pi Camera 3 recording (when the hardware arrives) | The real camera | Room tag, tape-measured references |

RGB is the only algorithm input on the public sequences; their depth and trajectory are used to score and nothing else.

**Selection, not final performance.** Both public sequences are used to choose between methods. Their numbers are
reported as selection results and never as the system's final accuracy. Final assessment needs recordings from the
drone's camera in prepared rooms, kept apart from anything used for tuning.

## 6. Measurement definitions

**Reference surface.** Measured depth of the selected keyframes, fused with the reference poses by the same
voxel map as every other run.

**Evaluated region.** Pixels whose measured depth is valid and between 0.4 m and 4.0 m. Predicted points from
other pixels are not scored; their share is reported as `outside_region_share`.

**Alignment.** None is fitted in a scored comparison. Clouds and depths are compared as produced.
- With reference poses, the frame and scale are the reference's own.
- With a tracker's poses, scale and frame come from the room tag. The public sequences have no tag, so for them
  a similarity is fitted to reference positions, either over the first 10 keyframes (showing drift after an
  anchor at the start) or over the whole trajectory. Both are written into the run as **DIAGNOSTIC**, and such
  a run is no evidence of metric scale.

| Measure | Definition |
|---|---|
| Scale error | abs(median over keyframes of median(predicted depth / measured depth) − 1), on the evaluated region |
| Surface accuracy | Distance from each predicted point to the nearest reference point: median, 90th and 95th percentile |
| Precision, recall, F-score | Share of predicted points within τ of the reference; share of reference points within τ of the prediction; their harmonic mean. τ = 5 cm and 10 cm |
| Coverage | Recall, plus per keyframe the share of evaluated pixels that survive calibration and filtering |
| Valid keyframes | Share of keyframes whose calibration is accepted |
| Depth error | Per keyframe: mean relative error, median and 95th-percentile absolute error, share within 5 cm |
| Trajectory error (runs C) | Position error after the diagnostic alignment: RMSE, median, maximum |
| Calibration support | Landmarks, inliers, image cells covered, depth spread, held-out error |
| DA3 landmark scale | The single scale that would bring DA3's depth onto the landmarks; ideally 1 |
| Scale drift between references | For a recording with two references: the scale at which the second board fits in the first board's metric frame, minus 1 |
| Reference check | Reprojection error of held-out views of the reference (every fourth view is kept out of the anchoring fit) |

Both clouds are thinned to one point per 2 cm voxel before distances are taken.

A low median can hide missing or wrong surfaces, so the median is never reported without recall and the tails.

## 7. Screening limits and decision rules

Provisional engineering limits, not published guarantees and not research thresholds.

| Check | Limit |
|---|---|
| Scale error | ≤ 3% |
| Surface accuracy, median | ≤ 5 cm |
| Valid keyframes | ≥ 95% |

A method passes screening only if all three hold on the validation sequence, and its recall, tails and
`outside_region_share` are reported beside them.

| Result | Decision |
|---|---|
| DA3 improves geometry with reference poses and keeps the improvement with tracker poses | DA3-Small goes to its Quest resource test |
| DA3 does not improve geometry | Keep DA-V2 Small; stop work on that DA3 configuration |
| Both fail because tracking breaks | Work on tracking or capture conditions; a different depth model will not repair a trajectory |
| A method produces useful geometry | It gets its own Quest resource test. DA-V2 Small is tested on the headset before ZipDepth is considered |
| The smallest useful configuration exceeds the headset's budget or disturbs rendering | Reject that configuration |

**“Improves” is fixed here as:** F-score at 5 cm higher than DA-V2 Small's by at least 0.05 on the validation
sequence, with the 95th-percentile accuracy no more than 10% worse, and screening passed.

### Values chosen in this specification, for review before the freeze

Astra set the 3% and 5 cm limits, the DA3 window sizes and the order of tests. These were chosen here because a
fixed procedure needs them:

- the improvement margin (0.05 F-score at 5 cm) and the tail tolerance (10%);
- the valid-keyframe limit (95%), carried from Astra's first proposal;
- the evaluated region (0.4–4.0 m) and the 2 cm thinning;
- the keyframe rule, the calibration support rules and the 5% consistency tolerance in `configs/qrecon.yaml`;
- the choice of the two TUM sequences and which is development and which validation.

## 8. Order of work

1. Self-check and tests pass on the workstation.
2. Run A, then the B arms, on the development sequence. Tune parameters there only.
3. Record model revisions, hashes and parameter counts. **Freeze.**
4. Run A and the B arms on the validation sequence. Apply section 7.
5. Runs C once a tracker trajectory exists for the same sequences.
6. Quest resource test for each method with useful geometry (section 9).

## 9. Quest resource test

A small on-headset harness, not the full integration, run inside the Second Eyes app's normal load (MR rendering
and the detector). For each depth method, over at least ten minutes:

| Measure | How |
|---|---|
| Latency per keyframe or window | Median and 95th percentile, sustained |
| Processing time per minute of video | From the above and the keyframe rate |
| Peak process memory | The app's total, not the model's weights |
| Effect on rendering | Frame rate and stale frames with the method running and idle |
| Thermal behaviour | The headset's thermal state over the run |
| Map growth | Voxels and memory against keyframes; must level off |

Language generation is paused during reconstruction. Limits for memory and rendering are taken from the repository's
existing Gate A definitions; where those don't cover a measure, the numbers are recorded and the lead decides before
integration. A phone benchmark or a desktop timing is never used as an estimate for the headset.

### On the headset, before the app integration

Tracking and the reconstruction stages first run on the headset as command-line programs started over adb, while
the Second Eyes app runs in the foreground with its detector (its normal load). The depth model runs on the CPU
through ONNX Runtime. Running it on the GPU inside the app is a later step (O38). The PC only decodes a recorded video
into frames and copies them over; with the drone, frames will arrive from the CM4.

### Tracking on the headset

Before the depth methods, the tracker itself is measured on the headset: `tools/track_on_quest.py` runs it over adb while
the Second Eyes app runs in the foreground with its detector (its normal load). Recorded: tracking time per frame
(median, 95th percentile), share of frames tracked, lost events, CPU time, peak memory, processing time per minute of
video, and the thermal state before and after.

## 10. Evidence levels

Every statement about this pipeline carries one of these labels:

| Label | Means |
|---|---|
| Synthetic | Passed the self-check and unit tests on generated data. Shows the plumbing is right, nothing more |
| Synthetic, both implementations | The headset program and the desktop harness reach the same result on the same generated input |
| Public data | Measured on the TUM sequences under this specification |
| Quest | Measured on the headset under section 9 |
| Drone camera | Measured on recordings from the CM4 and Pi Camera 3 in a prepared room |

Nothing is claimed at a level it hasn't reached.

## Changes from v1 (15 November)

- The room reference may be a ChArUco board as well as an AprilTag; the house video uses two ChArUco boards. The
  AprilTag library pin is dropped (OpenCV's detector covers both).
- Tracker pins and the stella patch (D132) added to section 3.
- The house video's role in section 5: calibration from its board views, metric scale from board 01, scale drift from
  board 02.
- Two measures added to section 6: scale drift between references and the held-out reference check.
- Section 9 gains the tracking test on the headset.

None of this changes the public-sequence comparison (runs A and B). If you froze v1 already, freeze again with v1.1:
validation hasn't started.

## Changes from v1.1 (15 November)

- Section 2: every stage exists in the desktop harness and in the headset program, and the two are tested against
  each other.
- Section 9: the headset tests start as command-line programs alongside the running app, with the depth model on
  the CPU (ONNX Runtime); the in-app GPU route follows.
- Section 10: a level for agreement between the two implementations.

The public-sequence comparison is unchanged.
