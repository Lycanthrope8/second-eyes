# The automatic scan (A4)

How the headset decides, on its own, when to take a keyframe during a scan (decision A of stage 3: automatic, so no
step can be forgotten or doubled). The rule is `quest-app/Assets/SecondEyes/Perception/Scan/KeyframeRule.cs`,
Unity-free; its settings are `schemas/scan-config.v1.json`, and the defaults are
`perception/scan/scan.default.v1.json`. Each keyframe it asks for becomes one detector inference on one camera frame,
copied with its own timestamp and pose, and one keyframe of the capture record (`capture-record.md`).

## The rule

Fed the head's pose once per rendered frame (position in `quest_tracking`, y up; yaw in degrees, 0 facing +z,
positive turning right), the rule asks for a keyframe:

- only while tracking holds, no inference is running, and the head has held still for `steady_s`: over that time
  its yaw varied by at most `steady_deg` (largest minus smallest) and its position stayed within `steady_m` of where
  it is now, so the image is sharp and the pose settled;
- the first as soon as the head is still;
- at the same spot, when the head faces a direction at least `step_deg - step_margin_deg` from every keyframe already
  taken there, so turning back adds nothing;
- at a new spot, when the head has moved `new_position_m` or more across the floor from the spot's first keyframe;
  that keyframe starts the next position.

A keyframe that couldn't be taken (the camera not delivering, say) leaves the rule as it was. The rule also gives the
panel a hint: hold still, turn about 30 degrees, move to a new spot, or the scan is complete.

## The settings

The defaults are A4's first values, set by reasoning, not tuned; a change is a new `config_id`.

| Setting | Default | What it does |
|---|---|---|
| `step_deg` | 30 | the turn between keyframes at one spot (the capture record's nominal step) |
| `step_margin_deg` | 5 | how much short of a full step still counts as a new direction |
| `steady_s` | 0.5 | how long the head must hold still |
| `steady_deg` | 3 | how much its yaw may vary while still |
| `steady_m` | 0.05 | how far it may drift while still |
| `new_position_m` | 1.0 | how far across the floor makes a new spot |
| `max_keyframes_per_position` | 12 | keyframes at one spot at most (a full circle) |
| `max_positions` | 4 | spots in one scan at most |

## Checks

`tools/csharp-tests/Scan` runs the rule on simulated head movements at 72 Hz, holds, turns and walks, with
expectations worked out by hand:

1. A full circle in 30-degree turns gives twelve keyframes at one spot, each facing its own direction.
2. Turning back to the first direction adds nothing.
3. Turning back and forth between two directions gives two keyframes.
4. A slow continuous turn is never still.
5. A tremor of 0.5 degrees still counts as holding still; a sway of 2.5 degrees does not.
6. Walking 1.2 m and holding still starts a new position; walking 0.6 m does not.
7. A keyframe that comes due while the detector is busy waits, then comes once.
8. Nothing is taken while tracking is lost.
9. Directions are measured correctly across the +-180 degree seam.
10. Recording a keyframe that wasn't due is refused.
11. and 12. The per-position and per-scan limits hold, and the hint then says to move or to finish.

Ignoring a busy detector makes case 7 fail, which shows the cases bite.

## Not settled yet

How still a real head holds, and so whether `steady_deg` and `steady_m` suit a person, is measured in stage 3's probe
trial: it logs the head's yaw and position while you scan.
