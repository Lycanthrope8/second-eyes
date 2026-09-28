# Event logs

How every session log is written, so it can be read, checked and aligned later. This page explains the format; the exact rules for one line are in `schemas/log-event.v1.json`. It implements proposal S0 item 3.

## Files

- One file per app session, in JSON Lines: one JSON object per line, appended as things happen. That keeps writing cheap on the headset.
- The first line is `session.start` and the last is `session.end`. A missing `session.end` means the app crashed or the file was pulled mid-session.
- On the PC, a run's logs live in `runs/<run ID>/raw/`, which Git ignores until raw-data storage is decided (open item O9).
- On the headset, the app writes `Android/data/com.secondeyes.quest/files/logs/<UTC start>_<session ID>.jsonl`, for example `20260928T051200Z_a3f9c2e1.jsonl`, in UTF-8 without a byte-order mark.
- Events are timestamped when they happen and queued; a background thread writes them once per second, and right away when the app pauses or quits. A crash loses at most the last second (D17). The writer is `quest-app/Assets/SecondEyes/Logging/EventLog.cs`.
- A log is tied to its run when it is pulled into `runs/<run ID>/raw/` (D16): quit the app, then run `python tools/logs.py pull <run ID>`. It copies every new session log into the run, checks each copy's size, and moves the original into `logs/pulled/` on the headset, so nothing is copied twice and nothing is deleted. It refuses while the app is still running, because that session's log isn't finished.

## Every line

```json
{"seq": 41, "mono_us": 81234567890, "utc_us": 1790503200123456, "ev": "asr.text", "data": {"text": "inspect the box behind the table"}}
```

| Field | Meaning |
|---|---|
| `seq` | line counter from 0 within the file, so a lost line shows up |
| `mono_us` | device monotonic clock in microseconds (on the headset, .NET's `Stopwatch`); never jumps, so durations and, later, clock alignment use it |
| `utc_us` | wall clock as Unix time in microseconds; can jump when the clock is corrected |
| `ev` | event type in lower case, prefixed by its component: `asr.text`, `slm.output`, `drone.telemetry` |
| `data` | the event's own fields |

Both clocks are integers in microseconds because every JSON reader keeps those exact; Unix time in nanoseconds would lose digits.

## Units

A data field with a unit carries it in its name: `latency_ms`, `dist_m`, `yaw_deg`. Coordinate frames are decided when the first pose is logged (open item O3).

## Event types

Defined in v1:

| `ev` | `data` fields | When |
|---|---|---|
| `session.start` | `format` (1), `session_id`, `app_version`, `os_build` (or null) | first line |
| `session.end` | none | last line |
| `mark` | `text` | a note added during a session |
| `error` | `where`, `message` | something went wrong |
| `display.rate` | `requested_hz`, `available_hz` (list) | once at startup, after the app requests its refresh rate (A1.4a) |
| `hands.state` | `left_tracked`, `right_tracked` (true or false) | at startup, then whenever either hand starts or stops being tracked (A1.5) |
| `control.stop` | `source` (`button_b`) | when the stop button, B on the right controller, is pressed (A1.5) |

Planned, and defined by the phase that first emits them (proposal S0 item 3): speech start and end, recognized text, model input and output, candidate probabilities, the validated goal, commands sent, and drone telemetry. Sync events are defined when time sync is built (deferred, D10).

To add an event type, add its `data` schema under `$defs` in `schemas/log-event.v1.json` (the entry's name is the event type), add a row to the table above, and log the change in `notes/decisions.md`. Until then, `check` reports it as undocumented.

## Example

```
{"seq": 0, "mono_us": 81230000000, "utc_us": 1790503196110000, "ev": "session.start", "data": {"format": 1, "session_id": "a3f9c2", "app_version": "0.1.0", "os_build": null}}
{"seq": 1, "mono_us": 81231500000, "utc_us": 1790503197610000, "ev": "mark", "data": {"text": "headset on, idle in Home"}}
{"seq": 2, "mono_us": 81290000000, "utc_us": 1790503256110000, "ev": "session.end", "data": {}}
```

## Reading and checking

```
python analysis/eventlog.py check <run ID or file>
python analysis/eventlog.py summary <run ID or file>
```

`check` reports malformed lines, fields the format doesn't know, gaps in `seq`, the monotonic clock going backwards, a missing `session.start`, events after `session.end`, and undocumented event types. `summary` prints each file's session, start time, duration and event counts. Scripts in `analysis/` can reuse the reader: `from eventlog import read_events`.
