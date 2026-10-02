# schemas

File formats that every component follows, kept in one place so the headset app and the Python code cannot drift apart. To change a format, change its file here first and log the change in `notes/decisions.md`.

| File | Format of | Since |
|---|---|---|
| `run-config.v1.json` | `runs/<run ID>/config.yaml` | S0.2 |
| `log-event.v1.json` | one line of a session log (see `docs/logging.md`) | S0.3 |
| `scene.v1.json` | one scene record (`docs/scene-contract.md`) | A2.1a |
| `command-context.v1.json` | one typed command with the user's pose (`docs/scene-contract.md`) | A2.1a |
| `category-map.v1.json` | one category map (`docs/scene-contract.md`) | A2.1a |
| `relation-config.v1.json` | the relation library's thresholds, bands and category lists (`docs/relations.md`) | A2.1b |
| `direction-config.v1.json` | the directional relations' band and cutoffs (`docs/directions.md`) | A2.1c |
