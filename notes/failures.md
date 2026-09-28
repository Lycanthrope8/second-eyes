# Failure stories

Every failure worth remembering, including ones that will not make the paper. Each entry has a run ID and, where possible, media.

| Date | Phase | What happened | Cause, if known | Run ID | Media (path) |
|---|---|---|---|---|---|
| 2026-09-28 | A1.3a | Unity's build cache `quest-app/.utmp/` was committed with the project | `.gitignore` had no rule for it (missing from Claude's ignore list); fixed by ignoring and untracking it | — | — |
| 2026-09-28 | A1.6a | `tools/profile.py` never found OVR Metrics' CSVs, so the trial recordings had no frame, load or memory data, and I first concluded OVR Metrics wasn't recording | The headset names the files `com.secondeyes.quest#...csv`, while the tool looked for `com_secondeyes_quest...`, the name of the uploaded copies (Claude's error). Fixed; `r008` and `r009` were deleted as never committed | `20260928_A1_r007` | — |
| 2026-09-28 | A1.6 | App memory went from 3.5 to 4.9 GB after the headset came off and on; free memory fell to 259 MB and the frame rate to between 1 and 65 fps | The app sets up its graphics memory again on each off-on without releasing the old copy (O14) | none: exploratory sample, OVR Metrics CSV `com_secondeyes_quest_UnityPlayerGameActivity-20260928_124355.csv` | — |
| 2026-09-28 | A1.4a | Run marked dirty by files the build changes only while it runs (`ProjectSettings.asset`, `Assets/Resources/PerformanceTestRun*.json`, `Assets/StreamingAssets/`) | The run was created before the build had finished; the files were back to normal afterwards | `20260928_A1_r004` | — |
