# The A4 usability check (D122)

A4's exit: the project lead scans a real room with the Quest, types supported commands, and judges whether the system
grounds them to the right object or says honestly that it can't. It is a usability check, not a measurement: no
numbers are claimed from it (Gate C and the formal evaluation come at the end of the pipeline).

## The room

The bedroom, with the objects measured in `perception/bedroom/size-priors.bedroom.v1.json`: two chairs (88 and 115 cm
tall), a TV, a laptop, a keyboard, a mouse, a bottle, and the bed (no size: it is placed only when seen from two spots).
The closet's mirrored doors are covered while scanning: reflections would be placed as objects behind them.
The dining table is in another room and is left out of the category map.

## The scan

1. Cover the mirrors. Start a session, then press **Scan**. The guidance appears in view, below the frame rate.
2. At the first spot, face part of the room and hold still until the controller buzzes. Then turn about 30 degrees and
   hold still again, watching the level row fill. At a desk or table, also look down at it and hold still: that fills
   the down row, and brings what stands on it into the picture. Skip directions facing a wall right beside you.
3. Walk 2 to 3 m to another side of the room and do the same. Objects seen from two spots that far apart are placed
   from both views, without their sizes; a third spot helps where objects hide each other.
4. Bring the panel with the right thumbstick and press **Finish**. The result box lists what was found, each with how
   it was placed: from two or more spots, from its size, position unknown, or maybe the same as another.

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
