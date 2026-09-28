# Failure stories

Every failure worth remembering, including ones that will not make the paper. Each entry has a run ID and, where possible, media.

| Date | Phase | What happened | Cause, if known | Run ID | Media (path) |
|---|---|---|---|---|---|
| 2026-09-28 | A1.3a | Unity's build cache `quest-app/.utmp/` was committed with the project | `.gitignore` had no rule for it (missing from Claude's ignore list); fixed by ignoring and untracking it | — | — |
| 2026-09-28 | A1.4a | Run marked dirty by files the build changes only while it runs (`ProjectSettings.asset`, `Assets/Resources/PerformanceTestRun*.json`, `Assets/StreamingAssets/`) | The run was created before the build had finished; the files were back to normal afterwards | `20260928_A1_r004` | — |
