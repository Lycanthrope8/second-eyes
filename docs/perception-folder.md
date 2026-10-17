# The perception folder (A4)

The files the headset's scan and the session's scanned rooms need, which the accepted parser folder (A2.7, D115)
doesn't hold. They travel as their own folder, verified on the headset as the parser folder is: file by file against
`manifest.json`, whose own SHA-256 identifies the folder in every capture record.

| Folder path | From the repository | What for |
|---|---|---|
| `schemas/capture.v1.json`, `scene.v3.json` | `schemas/` | the scan's record and its scene |
| `schemas/grounding-query.v2.json`, `grounding-result.v2.json`, `command-context.v2.json` | `schemas/` | a scanned room's queries, results and typed commands |
| `schemas/size-priors.v1.json`, `localization-config.v1.json`, `scan-config.v1.json`, `category-map.v1.json` | `schemas/` | checking the files below |
| `configs/scan.default.v2.json` | `perception/scan/` | when the scan takes a keyframe (`scan.md`) |
| `configs/localization.default.v1.json` | `perception/localization/` | the localization's thresholds (`localization.md`) |
| `maps/category-map.bedroom.v1.json` | `perception/bedroom/` | the bedroom's category map |
| `priors/size-priors.bedroom.v1.json` | `perception/bedroom/` | the bedroom's measured sizes |

The manifest names the four roles (`scan_config`, `localization_config`, `size_priors`, `category_map`), so a different
room or configuration is a different folder, not a code change.

## On the PC

    python -m grounding.quest.perception_device build --out <new folder>
    python -m grounding.quest.perception_device verify --folder <built folder>
    python -m grounding.quest.perception_device push --folder <built folder>

`build` checks every file first (the schemas are JSON Schemas; the configurations and size priors pass their schemas;
the map passes the contract), copies each with Git's line endings (LF), so a Windows checkout with CRLF line endings
builds the very same folder and manifest SHA-256 as any other, and writes the manifest last. `push` verifies the folder,
removes the headset's old manifest so the app never loads a half-pushed folder, pushes each file with its size checked
on the headset into `files/perception/`, the manifest last, and writes a receipt beside the folder.
`grounding/tests/test_perception_folder.py` checks the build and the verification.

## On the headset

`Start session` loads the folder when it is there (`Perception/PerceptionFolder.cs`): every file must match the
manifest, the configurations and size priors must pass their schemas, and the map the contract; otherwise no session
starts, and the panel says why. The parser folder then takes, from the perception folder, only the schemas and maps it
doesn't hold itself; its own files always come first, and without a perception folder nothing changes. The scan
needs the folder; without it the Scan button says so. `tools/csharp-tests/Contract` loads a folder built by the
Python tool and refuses a copy with one changed file.
