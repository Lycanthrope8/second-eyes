# A4 Delivery 1: index (19 October 2026)

51 commits on top of `833ef8d`, then this closing; 225 files. The phase is closed in `A4_closing.md`.

- **The automatic scan:** `quest-app/Assets/SecondEyes/Perception/AutoScan/` (`AutoScan.cs`, `RoomModel.cs`, `Boxes.cs`,
  `Glue/AutoScanner.cs`); the see-through shader in `quest-app/Assets/SecondEyes/Resources/`.
- **The panel:** `quest-app/Assets/SecondEyes/Grounding/ChatPanel.Scan.cs`, `.Prompt.cs`, `.Session.cs`;
  `Grounding/Interactive/InteractiveService.cs` (one scanned room per session).
- **The camera-only stand-in, kept for the drone:** `quest-app/Assets/SecondEyes/Perception/Scan/` and the localizer.
- **Settings and data:** `perception/autoscan/autoscan.default.v1.json` to `v8`, `schemas/auto-scan-config.v1.json` to
  `v8`, `perception/bedroom/category-map.bedroom.v2.json` to `v5`, `size-priors.bedroom.v1.json` and `v2`.
- **The contract:** `grounding/contract/checks.py`, `ContractCheck.cs`, scene v3's perceived profile.
- **The PC tools:** `tools/localize` (the observation-log replay and the complete comparison), `grounding/quest/
  scan_device.py` (pull and view), `grounding/quest/perception_device.py` (the folder).
- **The checks:** `tools/csharp-tests/AutoScan` (106), the other C# projects, and the Python suites `test_contract_v3`,
  `test_resolution_v2`, `test_perception_folder`, `test_scan_device`.
- **The docs:** `docs/auto-scan.md`, `docs/a4-usability.md`, `docs/scene-contract.md`, `docs/perception-folder.md`,
  `docs/localization.md`, `docs/scan.md`.
- **The records:** `notes/decisions.md` (D123 to D130), `notes/failures.md`, this index, `A4_closing.md` and the closing
  runbook `A4_runbook_closing.md`.
