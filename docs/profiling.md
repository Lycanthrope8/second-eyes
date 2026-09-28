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

## Comparing two runs

```
python analysis/profile.py compare <run A> <run B>
```

Two runs agree when, with the limits proposed in A1.6, the mean frame rate is within 0.5 fps, peak app memory within 5%, and mean CPU and GPU load each within 3 percentage points.

## Summary lines

| Line | Meaning |
|---|---|
| frames | mean and minimum frame rate, stale frames, seconds below 71 fps |
| load | mean and peak CPU and GPU load, the app's GPU time per frame, the CPU and GPU levels used |
| memory | the app's memory at the start, at the end and at its peak, the graphics part, and the least free memory on the headset |
| heat | SoC and battery temperature at the start and end and at their peak, the highest thermal status, mean power |
| battery | plugged in or on battery, and the level |
| CPU of one core | the app, OVR Metrics, and the busiest other processes (the evidence for O5) |
| flags | gaps in the rows, memory jumps, app restarts, the app not running |

On this headset, the SoC's thermal thresholds start at 89 °C (status 1, light throttling), according to `dumpsys thermalservice`.
