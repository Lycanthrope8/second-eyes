# Conventions

How this repository is organized, and the rules that keep it easy to follow. The map is in the top-level README. If a rule changes, edit it here and log the change in `notes/decisions.md`.

## Rules

1. **Present versus past.** `docs/` describes how things work now and is edited in place. `notes/` records what happened; entries are dated and only appended to. When a design changes, edit the doc and add a line to `notes/decisions.md` saying what changed and why.
2. **Every top-level folder has a README.md** saying what belongs there. Update it when that changes.
3. **Folders fill when their phase starts.** Until then a component folder holds only its README. Inner structure is added when the work begins, not before.
4. **Large files never go into Git:** video, audio, Vicon recordings, point clouds, model weights and builds. `.gitignore` blocks the common types. Until raw-data storage is decided (open item O9, after Gate A), keep them outside the repository.
5. **Phase notes are written on the day a phase ends,** from `notes/phases/_template.md`, and saved as `notes/phases/<phase ID>_<short name>.md`.
6. **Every number has a source.** Each number in a note or a paper traces to a run ID (format and rules: `runs/README.md`).
7. **Running lists are kept as things happen:** `notes/decisions.md`, `notes/limitations.md` and `notes/failures.md`. Negative results go in too; they often end up in a discussion section or a rebuttal.
8. **Photograph every physical setup on the day it is built:** props, partitions, cradle, tag and markers. Keep the originals outside the repository until storage is decided (O9).
9. **Tag a Git release at every gate.** Tag names are open item O8.
10. **Suggestions are not decisions.** Something is decided only once it has been approved and logged in `notes/decisions.md`.

Anything not written here is not decided yet. The open list is in `notes/decisions.md`.
