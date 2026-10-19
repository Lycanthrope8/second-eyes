# The A4 usability check (D122)

A4's exit: the project lead scans a real room with the Quest, types supported commands, and judges whether the system
grounds them to the right object or says honestly that it can't. It is a usability check, not a measurement: no
numbers are claimed from it (Gate C and the formal evaluation come at the end of the pipeline).

## The room

The bedroom, with the objects measured in `perception/bedroom/size-priors.bedroom.v1.json`: two chairs (88 and 115 cm
tall), a TV, a laptop, a keyboard, a mouse, a bottle, and the bed (no size: it is placed only when seen from two spots).
The closet's mirrored doors can stay uncovered: their reflections land behind the wall, outside Meta's room, and are left
out. The map v3 (`category-map.bedroom.v3.json`) keeps the detector's couch as a couch whose parser word is sofa, names the
detector's dining table (the desks) a table, and maps Meta's Space Setup labels.

## The scan

The automatic scan (`auto-scan.md`) runs by itself from the moment the app starts.

1. Start the app. Meta's Space Setup opens: look around the room until it has the walls and has wrapped the furniture
   in boxes, then finish. The app resumes.
2. Look around the room at a normal pace, including down at desks and tables. Walking around helps: furniture names
   its box sooner, and every object's box fits more closely as you see more of its sides. Each object appears as a
   see-through block with bright edges (Meta's boxes blue, the scan's green, unnamed grey), its name over its top.
3. Start a session. The scan's objects join it; the panel's Scan button reads Pause or Resume, and the right thumbstick
   hides the panel when it is in view and brings it in front otherwise. Typed commands always go to your scanned room;
   the Scene button and presets are for the dataset scenes.

## The commands

The positive examples are commands a scanned room can answer: a target **closest to** or **farthest from** an anchor
(`resolution.md`), between objects that were found. For example:

- `the chair closest to the bed`
- `the chair farthest from the tv`
- `the bottle closest to the laptop`
- `the mouse closest to the keyboard`

Since the automatic scan's v3 settings (D125), every object is a box with an inferred turn about the vertical and a size,
so commands with *near*, *on* or *below* are evaluated on those boxes; they are as reliable as the assumptions the boxes
state (a size prior completing a side, a small thing taking its desk's turn). A box's turn is not the object's front, so a
command that needs an object's front still comes back unresolved, not with a guess. Answers are among the objects the scan
recognized (D128). A command about two chairs that might be one chair seen twice should come back ambiguous.

## What counts

For each command, note what you typed, what the panel says it read (**Read as**), and the answer:

- **right**: the target is the object you meant;
- **honest**: it asks, or says it can't (ambiguous, no match, insufficient information), for a reason that fits;
- **wrong**: it names another object.

The scan's files (`files/scans/<capture ID>/`) and the event log are pulled after the check, so every answer can be
traced to its scan, its scene and its query.
