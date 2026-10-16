# The A4 usability check (D122)

A4's exit: the project lead scans a real room with the Quest, types supported commands, and judges whether the system
grounds them to the right object or says honestly that it can't. It is a usability check, not a measurement: no
numbers are claimed from it (Gate C and the formal evaluation come at the end of the pipeline).

## The room

The bedroom, with the objects measured in `perception/bedroom/size-priors.bedroom.v1.json`: two chairs (88 and 115 cm
tall), a TV, a laptop, a keyboard, a mouse, a bottle, and the bed (no size: it is placed only when seen from two spots).
The dining table is in another room and is left out of the category map.

## The scan

1. Start a session, then press **Scan**.
2. Stand at the first spot and look around slowly. Turn about 30 degrees, then hold still until the status says
   `keyframe` or moves on. A full circle gives up to 12 keyframes.
3. Walk about a metre and a half to a second spot and do the same; a third spot helps where objects hide each other.
4. Press **Finish**. The result box lists what was found, each with how it was placed: from two or more spots, from its
   size, position unknown, or maybe the same as another.

## The commands

The positive examples are commands a scanned room can answer: a target **closest to** or **farthest from** an anchor
(`resolution.md`), between objects that were found. For example:

- `the chair closest to the bed`
- `the chair farthest from the tv`
- `the bottle closest to the laptop`
- `the mouse closest to the keyboard`

Commands with *near*, *on* or *below* need object orientations a scan doesn't estimate, so they should come back
unresolved (`insufficient_information`), not with a guess. A command about two chairs that might be one chair seen twice
should come back ambiguous.

## What counts

For each command, note what you typed, what the panel says it read (**Read as**), and the answer:

- **right**: the target is the object you meant;
- **honest**: it asks, or says it can't (ambiguous, no match, insufficient information), for a reason that fits;
- **wrong**: it names another object.

The scan's files (`files/scans/<capture ID>/`) and the event log are pulled after the check, so every answer can be
traced to its scan, its scene and its query.
