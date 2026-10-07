# Profiling

How the cost of a build is measured on the headset, so every measurement is taken the same way and any two can be compared. A1.6 uses it for the empty app's reference; later steps compare against that reference.

## What is recorded

- **OVR Metrics Tool** (Meta's) records one row per second to a CSV on the headset: frame rate, stale frames, CPU and GPU load and levels, the app's memory including its graphics memory, battery temperature and power draw.
- **`tools/profile.py record`** runs on the PC and asks the headset every 30 seconds for its thermal state and temperatures, its battery state and its process list (`dumpsys thermalservice`, `dumpsys battery`, `top`). It stores the raw text in `runs/<run ID>/raw/sampler.jsonl` and, at the end, pulls the OVR Metrics CSV written during the recording.
- The app's memory comes from the CSV, not from `dumpsys meminfo`, because asking the app for its memory makes it do extra work in the middle of the measurement.
- The app itself is unchanged, so we measure the build and not the measuring. OVR Metrics does cost something, about one CPU core while it's active, and the summary reports that separately.
- `analysis/profile.py` reads both. Only CSV rows inside the recording window count.

## Procedure for one measurement

1. Reboot the headset, put it on, and wait about two minutes in Home.
2. In OVR Metrics Tool, turn on CSV recording and leave its overlay off.
3. Start the app. Sit with the controllers down and your hands in view.
4. On the PC, with everything committed:
   ```
   python tools/runs.py new A1 --purpose "..." --operator <initials> --headset
   python tools/profile.py record <run ID> --minutes 10
   ```
5. While it records, look around naturally and keep the headset on (see below).
6. When it finishes, quit the app, then pull its log and summarize:
   ```
   python tools/logs.py pull <run ID>
   python analysis/profile.py summary <run ID>
   ```
7. Turn OVR Metrics' CSV recording off again.

The headset stays plugged in (D22), so the summary reports the charging state instead of battery drain, and the heat readings include some warmth from charging. Each run's notes should say "plugged in".

## Keep the headset on

Taking the headset off and putting it back on while the app runs currently makes the app set up its graphics memory again without releasing the old copy: about 1.6 GB more each time (open item O14). If the headset comes off during a recording, the run is not valid: quit and restart the app, and record a new run. The summary flags gaps in the OVR Metrics rows and sudden memory jumps, so this can't go unnoticed.

## A1.7d: what the model costs (D50)

Six runs of one build, in two blocks. Each block starts with steps 1 and 2 of the procedure above: a freshly rebooted
headset, worn, two minutes in Home, and OVR Metrics' CSV recording on. Within a block the headset stays on, and each run
is its own app session: start the app, set the condition, record, quit the app, pull its log. The panel starts with the
model off (D50).

| Run | Block | Condition | Set it by | Minutes |
|---|---|---|---|---|
| 1 | 1 | model off: the app's reference, which A1.6 would have taken (D25) | starting the app | 10 |
| 2 | 1 | model loaded, idle | Load model, then waiting for Ready | 10 |
| 3 | 1 | generating, 150 steps per frame (Meta's default) | Load model, then Repeat | 10 |
| 4 | 2 | model off again: the reference's second repeat (D23, D24) | starting the app | 10 |
| 5 | 2 | generating, 50 steps per frame | Load model, Steps to 50, then Repeat | 5 |
| 6 | 2 | generating, 15 steps per frame | Load model, Steps to 15, then Repeat | 5 |

Each run's purpose starts with `A1.7d run <number>:`. The app is closed between runs, so the headset may come off
then; never while the app runs (O14). In runs 1, 2 and 4 the summary's flags must be none, or the run is redone; in
runs 3, 5 and 6 a flag may come from the model itself, so it's reported, not redone, unless the headset came off.

Repeat sends the fixed prompt again 5 s after each answer, so the model works most of the time. Start
`tools/profile.py record` only once the condition is set. For the generating runs, `grounding/check_headset.py` checks
every answer against the PC references and summarizes the timing by steps per frame. `analysis/profile.py table` puts
all six runs in one table, and `compare` checks run 1 against run 4.

## A1.8a: backend and weights (D53)

Two runs in one block, from a freshly rebooted headset as above, compared with A1.7d's run 5 (CPU, 16-bit weights,
50 steps per frame, `20260930_A1_r021`). The panel's Backend and Weights apply at Load; 50 steps per frame is the
default from A1.8a on.

| Run | Condition | Set it by | Minutes |
|---|---|---|---|
| 7 | GPU backend, 16-bit weights | Backend to GPU, Load model, then Repeat | 5 |
| 8 | CPU backend, 32-bit weights (pushed with adb) | Weights to 32-bit, Load model, then Repeat | 5 |

If the 32-bit model can't load, or Android closes the app while it loads, that is run 8's result: pull the log (its
`mark` names what was loading) and keep the headset's own record of the kill (`adb logcat`).

## A1.8c: llama.cpp in the app (D58)

Four runs of one build in one block, from a freshly rebooted headset as above, each its own app session. Compare with
A1.7d's runs, which used Meta's runner in the same app.

| Run | Condition | Set it by | Minutes |
|---|---|---|---|
| 9 | model off: this build's reference | starting the app | 5 |
| 10 | llama.cpp loaded, idle | Runtime to llama.cpp, Load model | 5 |
| 11 | llama.cpp answering and scoring, 2 threads | Runtime to llama.cpp, Load model, then Repeat | 10 |
| 12 | the same, 4 threads | as run 11, with Threads set to 4 first | 5 |

`check_headset.py` checks every answer against the PC references and every set of scores against PyTorch's (D56),
with `--reference 20260929_A1_r016 20260930_A1_r026`. Recording from before Load also captures the load; the summary
then reports its memory jump on the memory line instead of flagging it.

## A1.10a: what the camera costs (D79; D80 proposed)

Two runs in one block, one app session each, with the same build: run A with the camera off, run B with it on. The
camera starts only when X on the left controller is pressed (docs/logging.md lists its events). Grant the
camera-permission prompt in an earlier session (the smoke test), so neither measured session shows it.

Create each run before starting the app: `tools/logs.py pull` files a session that started before its run was created
under `raw/before_run/`. While A2.2d is staged and A1.10a uncommitted, `tools/runs.py new` marks runs `git_dirty: true`;
record what was uncommitted with `python tools/runs.py note <run ID> "..."`.

1. Reboot the headset, put it on and wait about two minutes in Home. Turn on OVR Metrics' CSV recording, overlay off.
2. Run A, camera off:
   ```
   python tools/runs.py new A1 --purpose "A1.10a camera off: camera component disabled" --operator <initials> --headset
   ```
   Start the app, sit with the controllers down and your hands in view, wait one minute, then
   `python tools/profile.py record <run A> --minutes 10`. Quit the app (Meta button, then Quit), then
   `python tools/logs.py pull <run A>` and `python analysis/profile.py summary <run A>`.
3. Run B, camera on: create the run the same way (purpose "A1.10a camera on: left camera, default resolution,
   preview off"), start the app, press X, wait one minute, record ten minutes, quit, pull and summarize as for run A.
4. `python analysis/camera_frames.py <run B>` (and `<run A>`, which should show the camera never playing), then
   `python analysis/profile.py compare <run A> <run B>`.
5. Turn OVR Metrics' CSV recording off.

Keep the headset on throughout each session (O14). Each run's notes record "plugged in" (D22), the requested and
delivered resolution, the frame signal (`updated_flag`, `timestamp` or `none`) and that the preview was off.

## A1.10c: what the detector costs (D84 to D86)

Three runs with one build that has the detector set up (`docs/detector.md`). Y on the left controller turns the
detector on: it loads, warms up, runs its self-checks (about ten inferences on the test images), then detects on camera
frames back to back until Y turns it off and releases the model. Create each run before starting the app, as in A1.10a.

1. Run A, parity (not profiled): `python tools/runs.py new A1 --purpose "A1.10c parity: self-checks on GPU then CPU"
   --operator <initials> --headset`. Start the app, press Y, wait for the self-checks (about 20 s), press Y again;
   press the left thumbstick (backend now CPU), press Y, wait, press Y; quit, pull, then
   `python -m perception.detector parity <run A>` and `python analysis/detector_runs.py <run A>`.
2. Run B, the cost on the GPU backend: reboot and settle as in A1.10a, OVR Metrics' CSV recording on, create the run
   (purpose "A1.10c detector on camera frames, GPU backend, back to back"), start the app, press X (camera on), then
   Y (detector on). While it settles for one minute, face furniture or the props with no people in view and pull the
   left trigger once (a snapshot). Then `python tools/profile.py record <run B> --minutes 10`, quit, pull,
   `python analysis/profile.py summary <run B>`, `python analysis/camera_frames.py <run B>`,
   `python analysis/detector_runs.py <run B>` and `python analysis/profile.py compare 20261005_A1_r039 <run B>`
   (camera on, no detector).
3. Run C, the CPU backend for comparison: as run B, but press the left thumbstick before Y, and record three minutes.
4. The snapshot: `adb pull /sdcard/Android/data/com.secondeyes.quest/files/snapshots runs/<run B>/raw/snapshots`, then
   `python -m perception.detector snapshot <run B> --image runs/<run B>/raw/snapshots/<name>.png --id <name>`.
   Snapshots stay out of Git with the rest of `raw/`.

Keep the headset on throughout each session (O14). Each run's notes record "plugged in" (D22), the backend, the rate
setting (0: back to back) and anything unusual, such as frame drops when the detector started.

## A1.10c with slicing: tuning and confirmation (D88)

The criteria: under 1% stale frames over the operational phase and over the buckets overlapping inference (OVR
Metrics, its clock calibrated against the app's frame times; the bucket figure is a proxy, O26), and capture-to-result
latency p95 of 500 ms or less. `python analysis/detector_phases.py <run>` prints both, per detector load.

Once: push the r041 snapshot so the self-checks include it.
```
adb shell mkdir -p /sdcard/Android/data/com.secondeyes.quest/files/parity
adb push runs/20261006_A1_r041/raw/snapshots/b20c60c3_001.png /sdcard/Android/data/com.secondeyes.quest/files/parity/
```

1. Tuning, one run with three segments. Reboot and settle as in A1.10a, OVR Metrics' CSV recording on, create the run
   (purpose "A1.10c slicing tuning: 8, 16, 32 steps per frame, scan mode, GPU"), start the app, press X. With the
   detector off, push the left thumbstick left or right until the overlay shows `8 steps/frame` (mode `scan`). Start
   `python tools/profile.py record <run> --minutes 13`. Press Y and leave it for 3 scans (about 3.5 minutes; the
   overlay counts them), press Y and wait for `off`; set 16, press Y, 3 scans, Y; set 32, the same. Quit, pull, then
   `python analysis/detector_phases.py <run>` and
   `python -m perception.detector parity <run> --canvas-dir runs/20261006_A1_r041/raw/snapshots`.
   Freeze the fastest setting (lowest p95 latency) that meets both criteria; if none does, stop and report.
2. Confirmation: a fresh session at the frozen setting, ten minutes of repeated scans, recorded and analyzed the same
   way (purpose "A1.10c slicing confirmation: <n> steps per frame, scan mode, GPU, 10 minutes").
3. Diagnostics (optional): a short run with burst (2 Hz for 30 s) and then continuous for a minute.

Each run's notes record "plugged in", the settings and the order of the segments (later segments start warmer).

## A1.10c cost-balanced schedule (D89)

One bounded attempt: profile, build, tune, freeze, confirm. The criteria are unchanged (p95 capture-to-result 500 ms or
less, under 1% stale frames, parity, memory). Each step's output goes to the project lead before the next.

1. Profiling session (about 6 minutes recorded). In Unity: Player Settings > Frame Timing Stats on; `GpuFrameTimes` on
   the `Detector` object; build. Reboot and settle, OVR Metrics' CSV on, create the run (purpose "A1.10c D89 4-step
   diagnostic and step profiling"), start the app, press X. With the detector off, set mode `diagnostic` (thumbstick up
   or down) and `4 steps/frame` (left or right). Start `python tools/profile.py record <run> --minutes 6`. Press Y: the
   self-checks, eight clock-sync stalls, then one scan at 0.5 Hz (about 90 s); when the overlay says
   `diagnostic scan done`, press Y and wait for `off`. Set mode `profile`, press Y: self-checks, stalls, five profiling
   passes (about 100 s); at `profile done`, press Y. Quit, pull the logs, then
   `python analysis/detector_phases.py <run>` and `python -m perception.detector schedule <run>`.
2. Tuning session: push the candidate schedules
   (`adb shell mkdir -p /sdcard/Android/data/com.secondeyes.quest/files/schedules`, then
   `adb push perception/detector/schedules/<id>.json /sdcard/Android/data/com.secondeyes.quest/files/schedules/`),
   restart the app, and run each in scan mode at 1 Hz for two scans in one recorded run, as in the slicing tuning.
   Freeze the fastest schedule that meets both criteria; commit that schedule file and record it as a decision.
3. Confirmation: a fresh ten-minute session of repeated scans with the frozen schedule, analyzed the same way.

If no schedule meets both criteria, or this needs substantial new profiling or runtime infrastructure, stop: the next
step is CPU sequential operation, after confirming CPU parity (`python -m perception.detector parity 20261006_A1_r042`).

## Comparing two runs

```
python analysis/profile.py compare <run A> <run B>
```

Two runs agree when the mean frame rate is within 0.5 fps, peak app memory within 5%, mean CPU load within 3 percentage points (D24), and our app's own GPU time within 5% (D52). Whole-GPU load is shown too, but not judged: it includes the system's boundary and compositor, which vary with where the wearer sits.

## Summary lines

| Line | Meaning |
|---|---|
| frames | mean and minimum frame rate over all seconds, stale frames, seconds below 71 fps. Seconds without a row count as frameless: OVR Metrics writes nothing while the app is frozen. A lone missing row with normal frames and no stale frames around it is OVR Metrics skipping one and doesn't count (A1.7d) |
| model | for runs with sends: the frame rate in their prompt passes, answers and pauses, and the freezes inside prompt passes, which are not flags |
| load | mean and peak CPU and GPU load, the app's GPU time per frame, the CPU and GPU levels used |
| memory | the app's memory at the start, at the end and at its peak, the graphics part, and the least free memory on the headset |
| heat | SoC and battery temperature at the start and end and at their peak, the highest thermal status, mean power |
| battery | plugged in or on battery, and the level |
| CPU of one core | the app, OVR Metrics, and the busiest other processes (the evidence for O5) |
| flags | gaps in the rows, memory jumps, app restarts, the app not running |

On this headset, the SoC's thermal thresholds start at 89 °C (status 1, light throttling), according to `dumpsys thermalservice`.
