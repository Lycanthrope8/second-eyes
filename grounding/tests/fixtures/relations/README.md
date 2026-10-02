# Relation test fixtures

Test data for `grounding/tests/test_relations.py` (A2.1b, D67). Not training data.

`cases.json` holds the reviewed cases: each call, the hand arithmetic behind it, and the expected outcome, with expected measures compared within 1e-6. The expectations were calculated by hand from the definitions in `docs/relations.md` and approved on 2026-10-02 before the library existed; irrational values were evaluated with a calculator, never with the library. Case 7's lamp distance is the corrected sqrt(4.0749) = 2.0186381548, and cases 34 and 34b, a tilted cube, were added at approval.

Scenes: `a17` and `a17r` are the contract fixtures `scene.a17.annotated.json` and `scene.a17.restricted.json`. The three below are new, contract-valid and annotated; boxes are 0.2 m cubes and axis-aligned unless noted.

| Scene | Objects |
|---|---|
| `scene.rel.json` | obj_001 table 1.2 × 0.8 × 0.75 at (0, 0, 0.375); obj_002, 003, 004 boxes at (0, 0, z) with z = 0.85, 1.25, 0.10 (on, above, under the table); obj_005, 006, 007 boxes at x = 1.0, 1.2, 1.3, z = 0.10; obj_008 container, a 0.5 m cube at (2.0, 0, 0.25); obj_009, 010, 011 boxes at (2.0, 0, z) with z = 0.15, 0.42, 0.45; obj_012 box at (0, 1.0, 0.10), yaw 45 degrees; obj_013 table at (0, -3, 0.375), tilted 10 degrees about x; obj_014 box at (0, -3, 0.90); obj_015 box at (0.4, 0, 0.90) |
| `scene.rank.json` | anchor obj_001 at the origin; chairs obj_002, 003, 004 at x = 1, 2, 3; obj_005 at y = 1.04; obj_006 at y = -1.02; obj_007 and obj_008 with unknown centres; sizes and rotations unknown |
| `scene.tilt.json` | obj_001 unit cube at the origin, rotated 45 degrees about x; obj_002 axis-aligned unit cube at (0, 0, 1.6); obj_003 the first cube without its rotation |

| Case | Scene | Call | Arithmetic | Expected |
|---|---|---|---|---|
| 1 | a17 | `between(obj_001, obj_004, obj_005)` | L = 1.56; a = 1.2168/1.56 = 0.78 in [0.02, 1.54]; p = sqrt(0.05^2 + 0.08^2) = 0.0943398113 <= 0.292 (w = max(0.30, 0.312)); table y 1.10-1.70 vs chairs y 1.93-2.43 and 0.37-0.87: overlaps 0 | TRUE |
| 2 | a17r | `between(obj_001, obj_004, obj_005)` | centre conditions as in case 1; the overlap limit needs rotations, which are withheld | UNKNOWN |
| 3 | a17 | `between(obj_002, obj_004, obj_005)` | a = 1.248/1.56 = 0.80; p = sqrt(0.65^2 + 0.25^2) = 0.6964194139 > 0.332 | FALSE |
| 3r | a17r | `between(obj_002, obj_004, obj_005)` | the lateral condition is FALSE from centres alone, so withheld rotations don't matter | FALSE |
| 4 | a17 | `near(obj_002, obj_001)` | box_1's 30 degree footprint vertex at x = 2.173205 crosses the table edge x = 2.00 and the vertical extents overlap, so the boxes intersect: d = 0 | TRUE |
| 4r | a17r | `near(obj_002, obj_001)` | rotations withheld | UNKNOWN |
| 5 | a17 | `below(obj_002, obj_001)` | separated: s = 0.40 - 0 > 0.01, FALSE; furniture: offset 0, margin 0.40 - 0.74 = -0.34, overlap triangle 0.5 * 0.4 * 0.1732050808 = 0.0346410162 m^2 over 0.16 m^2 = 0.2165063509 < 0.45, FALSE | FALSE |
| 6 | a17 | `near(obj_003, obj_001)` | box_2's rotation is not in the source | UNKNOWN |
| 7 | a17 | `closest rank 1 to obj_001 of obj_004, obj_005; possible obj_006` | both chairs at sqrt(0.6173) = 0.7856844150, tied; the lamp at sqrt(4.0749) = 2.0186381548 changes neither combination | ambiguous obj_004, obj_005 |
| 7r | a17r | `closest rank 1 to obj_001 of obj_004, obj_005; possible obj_006` | centres are oracle in both profiles | ambiguous obj_004, obj_005 |
| 8a | rel | `near(obj_005, obj_001)` | x gap 0.9 - 0.6 = 0.30 <= 0.48 | TRUE |
| 8b | rel | `near(obj_006, obj_001)` | x gap 1.1 - 0.6 = 0.50, in (0.48, 0.52] | UNKNOWN |
| 8c | rel | `near(obj_007, obj_001)` | x gap 1.2 - 0.6 = 0.60 > 0.52 | FALSE |
| 9 | rel | `near(obj_012, obj_001)` | diamond vertex at y = 1.0 - 0.1414213562 = 0.8585786438; d = 0.4585786438 <= 0.48 (unrotated: 0.50) | TRUE |
| 10 | rel | `above(obj_003, obj_001)` | g = 1.15 - 0.75 = 0.40 >= 0.06; o = 0.04/0.04 = 1 | TRUE |
| 11a | rel | `above(obj_002, obj_001)` | g = 0.75 - 0.75 = 0 < 0.04 | FALSE |
| 11b | rel | `on(obj_002, obj_001)` | gate passes (table, tilt 0); -0.04 <= 0 <= 0.04; o = 1 | TRUE (top_face_support) |
| 12 | rel | `on(obj_003, obj_001)` | g = 0.40 > 0.06 | FALSE |
| 13a | rel | `on(obj_015, obj_001)` | g = 0.80 - 0.75 = 0.05, in (0.04, 0.06] | UNKNOWN |
| 13b | rel | `above(obj_015, obj_001)` | g = 0.05, in [0.04, 0.06) | UNKNOWN |
| 14 | rel | `on(obj_002, obj_004)` | a box is not a support; UNKNOWN before geometry | UNKNOWN |
| 15 | rel | `on(obj_014, obj_013)` | tilt 10 degrees > 5 | UNKNOWN |
| 16 | rel | `below(obj_004, obj_001)` | separated: s = 0.20 > 0.01, FALSE; furniture: offset 0 <= 0.08, margin 0.20 - 0.75 = -0.55, o = 1 | TRUE (beneath_furniture) |
| 17 | rel | `below(obj_001, obj_003)` | separated: s = 0.75 - 1.15 = -0.40; o = 0.04/0.04 = 1 | TRUE (separated) |
| 18 | rel | `below(obj_005, obj_001)` | separated: s = 0.20, FALSE; furniture: o = 0, FALSE | FALSE |
| 19a | rel | `inside(obj_009, obj_008)` | z' in [-0.20, 0.00]; e = max(0.10 - 0.25, 0.20 - 0.25) = -0.05 | TRUE (outer_box_containment) |
| 19b | rel | `inside(obj_010, obj_008)` | z' max 0.27; e = 0.02, in (0.01, 0.03] | UNKNOWN |
| 19c | rel | `inside(obj_011, obj_008)` | z' max 0.30; e = 0.05 > 0.03 | FALSE |
| 20 | rel | `inside(obj_004, obj_001)` | a table is not a container; UNKNOWN before geometry | UNKNOWN |
| 21 | rel | `between(obj_001, obj_002, obj_004)` | L = 0.75; a = 0.35625/0.75 = 0.475; p = 0; vs obj_002 v = 0.75 - 0.75 = 0, not counted; vs obj_004 v = 0.20 > 0 and o = 1 > 0.55 | FALSE |
| 22 | rel | `between(obj_003, obj_002, obj_004)` | a = (0.40 * -0.75)/0.75 = -0.40 < -0.02 | FALSE |
| 23 | rel | `between(obj_008, obj_010, obj_011)` | L = 0.03, below the 0.22 needed | UNKNOWN |
| 24 | rel | `between(obj_001, obj_001, obj_002)` | the IDs are not distinct | FALSE |
| 25a | rank | `closest rank 1 to obj_001 of obj_002, obj_003, obj_004` | gaps of 1.00 | resolved obj_002 |
| 25b | rank | `closest rank 2 to obj_001 of obj_002, obj_003, obj_004` | gaps of 1.00 | resolved obj_003 |
| 25c | rank | `closest rank 3 to obj_001 of obj_002, obj_003, obj_004` | gaps of 1.00 | resolved obj_004 |
| 25d | rank | `farthest rank 1 to obj_001 of obj_002, obj_003, obj_004` | gaps of 1.00 | resolved obj_004 |
| 26 | rank | `closest rank 1 to obj_001 of obj_002, obj_005` | gap 1.04 - 1.00 = 0.04 <= 0.05 | ambiguous obj_002, obj_005 |
| 27 | rank | `closest rank 2 to obj_001 of obj_002` | one candidate for rank 2 | no_match |
| 28 | rank | `closest rank 1 to obj_001 of obj_002, obj_007` | obj_007 has no centre | insufficient |
| 29 | rank | `closest rank 3 to obj_001 of obj_002; possible obj_003, obj_004` | {} no_match; {obj_003} no_match; {obj_004} no_match; {obj_003, obj_004} distances 1/2/3, obj_004 resolved; the combinations differ (one-at-a-time checking would wrongly give no_match) | insufficient |
| 30 | rank | `closest rank 1 to obj_001 of obj_002; possible obj_003, obj_004` | every combination resolves obj_002 (next gap >= 1.00) | resolved obj_002 |
| 31 | rank | `closest rank 1 to obj_001 of obj_002; possible obj_006` | {} resolves obj_002; {obj_006} gap 0.02, ambiguous | insufficient |
| 32 | rank | `closest rank 1 to obj_008 of obj_002, obj_003` | the anchor has no centre | insufficient |
| 33 | rank | `closest rank 1 to obj_002 of obj_002, obj_003` | the anchor is excluded; obj_003 at 1.00 | resolved obj_003 |
| 34 | tilt | `near(obj_001, obj_002)` | A's top edge at z = sqrt(2)/2 = 0.7071067812; B's bottom at 1.1; d = 1.1 - 0.7071067812 = 0.3928932188 <= 0.48 | TRUE |
| 34b | tilt | `near(obj_003, obj_002)` | the same cube without its rotation: d = 1.1 - 0.5 = 0.60 > 0.52 | FALSE |

Geometry checks, also hand-calculated:

| Scene | Object | Arithmetic | Expected |
|---|---|---|---|
| tilt | obj_001 | a unit cube turned 45 degrees about x: z from -sqrt(2)/2 to sqrt(2)/2; footprint 1 by sqrt(2); up axis tilted 45 degrees | z_min -0.7071067812, z_max 0.7071067812, footprint_area 1.4142135624, tilt_deg 45.0 |
| tilt | obj_002 | axis-aligned unit cube centred at z = 1.6 | z_min 1.1, z_max 2.1, footprint_area 1.0, tilt_deg 0.0 |
| rel | obj_013 | the table turned 10 degrees about x | tilt_deg 10.0 |
