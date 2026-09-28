# schemas

File formats that every component follows, kept in one place so the headset app and the Python code cannot drift apart. To change a format, change its file here first and log the change in `notes/decisions.md`.

| File | Format of | Since |
|---|---|---|
| `run-config.v1.json` | `runs/<run ID>/config.yaml` | S0.2 |
| `log-event.v1.json` | one line of a session log (see `docs/logging.md`) | S0.3 |
