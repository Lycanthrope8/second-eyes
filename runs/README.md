# runs

The run registry. Every run gets an ID and a folder holding its config, so every number can be traced back to exactly what produced it.

## Run IDs

`YYYYMMDD_<phase>_r<NNN>`, for example `20261001_A1_r007`.

- The date is the local date the run was created.
- The phase is one of the proposal's phase IDs; the allowed list is in `schemas/run-config.v1.json`.
- The number counts per phase and never resets, so "A1 r007" is unique on its own and means the seventh A1 run.

## Creating a run

```
python tools/runs.py new A1 --purpose "idle baseline" --operator AB
```

This makes `runs/<run ID>/config.yaml` from `_template.yaml` and fills in the ID, the creation time and the Git commit. It refuses to run outside a Git repository with at least one commit. If the repository has any uncommitted change, including new files and the config of an earlier run, it warns you and marks the run `git_dirty: true`. Files ignored by `.gitignore` don't count. So commit before you create a run.

After a Unity build, wait until it has completely finished before creating a run: while it runs, the build temporarily changes files in `quest-app/`. If `new` warns about uncommitted changes anyway, delete the new run folder, fix the cause (commit, or wait for the build), and create the run again. The number is free again because the folder was never committed.

Optional:

- `--checkpoint FILE` records the model file's name and SHA-256.
- `--headset` reads the OS build from the connected Quest over adb.

If any of these steps fails, nothing is created. Afterwards, open the config and fill the optional fields that apply to this run; leave the rest `null`.

## Checking runs

```
python tools/runs.py check 20261001_A1_r007
python tools/runs.py check --all
```

`check` reports missing or malformed fields, fields the schema doesn't know, a folder name that doesn't match its run ID or date, and two runs with the same phase and number. Run it before committing.

## What goes in a run folder

- `config.yaml`, tracked in Git.
- `raw/`, holding the run's logs and recordings. Git ignores it until raw-data storage is decided (open item O9), so the files exist only on your disk: back up anything you can't recreate. Headset logs get there with `python tools/logs.py pull <run ID>`, after you quit the app (see `docs/logging.md`).

## Config fields

| Field | Filled by | Meaning |
|---|---|---|
| `config_version` | tool | format version of this file (1) |
| `run_id`, `phase`, `created` | tool | ID, phase, local time with UTC offset |
| `purpose` | you, required | one line: why this run exists |
| `operator` | you, required | who ran it |
| `git_commit`, `git_dirty` | tool | commit, and whether the repository had uncommitted changes |
| `model_checkpoint`, `model_sha256` | you or `--checkpoint` | model file and its hash |
| `prompt_version` | you | version of the prompt template |
| `schema_version` | you | version of the model's output/scene schema (proposal §6.2) |
| `layout_id` | you | which prop layout |
| `headset_os_build` | you or `--headset` | the headset's `ro.build.fingerprint` |
| `drone_firmware` | you | drone firmware version |
| `lighting` | you | lighting conditions |
| `notes` | you | anything else; set it with `python tools/runs.py note <run ID> "<text>"`, which keeps the file's layout |

To add or change a field, change `schemas/run-config.v1.json` first and log the change in `notes/decisions.md`.
