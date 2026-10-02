# Direction test fixtures

Test data for `grounding/tests/test_directions.py` (A2.1c, D68). Not training data.

`cases.json` holds the 43 fixed rows of Section 7 of the A2.1c implementation brief, as 90 predicate calls. They were transcribed before `grounding/relations/directions.py` existed. Expected scores are the brief's arithmetic expressions, which are authoritative; the test evaluates them with its own small arithmetic reader, never with the library. Truth values are exact; scores and measures are compared within 1e-9 m.

Every scene and command below passes the contract validator together with `../contract/valid/category-map.fx.json`. Object centres are at z = 0 and the user at z = 1.6 unless noted; sizes, rotations and categories are neutral values the directions never read.

| Scene | Content |
|---|---|
| `scene.h.*.json` (fixture.dir.h) | targets for the user-heading frame: obj_001 (1,0), obj_002 (0,1), obj_003 (0,-1), obj_004 (1,1), obj_005 (4/5,-3/5); (s,0) for s = .03 to -.03 as obj_006 to obj_012; (0,s) as obj_013 to obj_018 with obj_009 at s = 0; obj_019 with an unknown centre |
| `scene.v.*.json` (fixture.dir.v) | anchor obj_001 (0,2) with targets obj_002 (1,2), obj_003 (0,3) and obj_004 (0,2); anchor obj_005 (3,4) with obj_006 (3.8,3.4); obj_007, an anchor with an unknown centre; anchors at (0,.02), (0,.03) and (0,.01) with targets at x = 1 (obj_008 to obj_013); obj_014, an anchor at (0,2) and z = 1.2; obj_015 with an unknown centre |
| `scene.i.*.json` (fixture.dir.i) | anchor obj_001 (0,2) facing (0,1,0) and obj_004 facing (0,-1,0), with targets obj_002 (0,3) and obj_003 (1,2); anchor obj_005 (0,0) facing (.36,.48,.8) with obj_006 (.8,-.6); anchors at the origin facing (0,0,1), (.1,0,sqrt(.99)), (.28,0,.96) and (.06,0,sqrt(1-.06^2)) (obj_007, obj_009, obj_010, obj_011) with target obj_008 (1,0); obj_012, an anchor with no front but a known rotation; obj_013, an anchor with a front but no size or rotation; obj_014 with an unknown centre |
| `scene.scope.annotated.json` | one target whose centre carries the scene's `local_note` assumption |

Each of the three scenes comes in the annotated and the restricted profile (`*.annotated.json`, `*.restricted.json`), with the same objects; the restricted one withholds rotations, fronts and colours and gives sizes from a prior.

Commands (`command.<key>.json`): `h.base` and `v.base`, a user at the origin facing +y; `h.east` and `v.east`, facing +x; `h.diag`, facing (3/5,4/5); `h.drift`, heading (0,1.0000005); `h.nopos` and `v.nopos`, no user position; `h.nohead`, no heading but a known rotation; `v.nohead`, neither heading nor rotation; `v.north`, a user at (0,4); `v.at_anchor` and `v.at_anchor_low`, a user at (0,2) at z = 1.6 and 0.3; `i.nopose`, a command with no pose; `scope.c001`, a user whose position carries the command's own `local_note`.

| Row | Section | Inputs | Calls and expectations |
|---|---|---|---|
| D01 | 7.1 | H; T=(1,0) | H `right(obj_001)` [h, h.base]: TRUE, score 1 |
| D02 | 7.1 | H; T=(1,0) | H `left(obj_001)` [h, h.base]: FALSE, score -1 (failed:left) |
| D03 | 7.1 | H; T=(0,1) | H `in_front_of(obj_002)` [h, h.base]: TRUE, score 1 |
| D04 | 7.1 | H; T=(0,-1) | H `behind(obj_003)` [h, h.base]: TRUE, score -(-1) |
| D05 | 7.1 | H; T=(1,1) | H `right(obj_004)` [h, h.base]: TRUE, score 1<br>H `in_front_of(obj_004)` [h, h.base]: TRUE, score 1 |
| D06 | 7.1 | H; H=(1,0); T=(1,0) | H `right(obj_001)` [h, h.east]: UNKNOWN, score 1*0+0*(-1) (boundary:right)<br>H `in_front_of(obj_001)` [h, h.east]: TRUE, score 1 |
| D07 | 7.1 | V; T=(1,2) | V `right(obj_002, obj_001)` [v, v.base]: TRUE, score 1<br>V `left(obj_002, obj_001)` [v, v.base]: FALSE, score -1 (failed:left) |
| D08 | 7.1 | V; T=(0,3) | V `behind(obj_003, obj_001)` [v, v.base]: TRUE, score -(2-3)<br>V `in_front_of(obj_003, obj_001)` [v, v.base]: FALSE, score -1 (failed:in_front_of) |
| D09 | 7.1 | V; U=(0,4); T=(0,3) | V `in_front_of(obj_003, obj_001)` [v, v.north]: TRUE, score 0*0+1*1 |
| D10 | 7.1 | V; U=(0,4); T=(1,2) | V `right(obj_002, obj_001)` [v, v.north]: FALSE, score 1*(-1)+0*0 (failed:right)<br>V `left(obj_002, obj_001)` [v, v.north]: TRUE, score 1 |
| D11 | 7.1 | V; T=(1,2); change H from (0,1) to (1,0) | V `right(obj_002, obj_001)` [v, v.base]: TRUE, score 1<br>V `right(obj_002, obj_001)` [v, v.east]: TRUE, score 1 |
| D12 | 7.1 | I; T=(0,3) | I `in_front_of(obj_002, obj_001)` [i]: TRUE, score 3-2 |
| D13 | 7.1 | I; T=(1,2) | I `right(obj_003, obj_001)` [i]: TRUE, score 1 |
| D14 | 7.1 | I; F=(0,-1,0); T=(0,3) | I `behind(obj_002, obj_004)` [i]: TRUE, score -(0*0+1*(-1)) |
| D15 | 7.1 | I; F=(0,-1,0); T=(1,2) | I `right(obj_003, obj_004)` [i]: FALSE, score -1 (failed:right)<br>I `left(obj_003, obj_004)` [i]: TRUE, score 1 |
| D16 | 7.1 | H; H=(3/5,4/5); T=(4/5,-3/5) | H `right(obj_005)` [h, h.diag]: TRUE, score 16/25+9/25<br>H `in_front_of(obj_005)` [h, h.diag]: UNKNOWN, score 12/25-12/25 (boundary:in_front_of) |
| D17 | 7.1 | V; A=(3,4); T=(3.8,3.4) | V `right(obj_006, obj_005)` [v, v.base]: TRUE, score .64+.36<br>V `in_front_of(obj_006, obj_005)` [v, v.base]: UNKNOWN, score -.48+.48 (boundary:in_front_of) |
| D18 | 7.1 | I; A=(0,0); F=(.36,.48,.8); T=(.8,-.6) | I `right(obj_006, obj_005)` [i]: TRUE, score 1<br>I `in_front_of(obj_006, obj_005)` [i]: UNKNOWN, score 0 (boundary:in_front_of) |
| D19 | 7.2 | B; s=.03; repeated on H with T=(0,s) | H `right(obj_006)` [h, h.base]: TRUE, score .03<br>H `left(obj_006)` [h, h.base]: FALSE, score -(.03) (failed:left)<br>H `in_front_of(obj_013)` [h, h.base]: TRUE, score .03<br>H `behind(obj_013)` [h, h.base]: FALSE, score -(.03) (failed:behind) |
| D20 | 7.2 | B; s=.02; repeated on H with T=(0,s) | H `right(obj_007)` [h, h.base]: UNKNOWN, score .02 (boundary:right)<br>H `left(obj_007)` [h, h.base]: UNKNOWN, score -(.02) (boundary:left)<br>H `in_front_of(obj_014)` [h, h.base]: UNKNOWN, score .02 (boundary:in_front_of)<br>H `behind(obj_014)` [h, h.base]: UNKNOWN, score -(.02) (boundary:behind) |
| D21 | 7.2 | B; s=.01; repeated on H with T=(0,s) | H `right(obj_008)` [h, h.base]: UNKNOWN, score .01 (boundary:right)<br>H `left(obj_008)` [h, h.base]: UNKNOWN, score -(.01) (boundary:left)<br>H `in_front_of(obj_015)` [h, h.base]: UNKNOWN, score .01 (boundary:in_front_of)<br>H `behind(obj_015)` [h, h.base]: UNKNOWN, score -(.01) (boundary:behind) |
| D22 | 7.2 | B; s=0; repeated on H with T=(0,s) | H `right(obj_009)` [h, h.base]: UNKNOWN, score 0 (boundary:right)<br>H `left(obj_009)` [h, h.base]: UNKNOWN, score -(0) (boundary:left)<br>H `in_front_of(obj_009)` [h, h.base]: UNKNOWN, score 0 (boundary:in_front_of)<br>H `behind(obj_009)` [h, h.base]: UNKNOWN, score -(0) (boundary:behind) |
| D23 | 7.2 | B; s=-.01; repeated on H with T=(0,s) | H `right(obj_010)` [h, h.base]: UNKNOWN, score -.01 (boundary:right)<br>H `left(obj_010)` [h, h.base]: UNKNOWN, score -(-.01) (boundary:left)<br>H `in_front_of(obj_016)` [h, h.base]: UNKNOWN, score -.01 (boundary:in_front_of)<br>H `behind(obj_016)` [h, h.base]: UNKNOWN, score -(-.01) (boundary:behind) |
| D24 | 7.2 | B; s=-.02; repeated on H with T=(0,s) | H `right(obj_011)` [h, h.base]: UNKNOWN, score -.02 (boundary:right)<br>H `left(obj_011)` [h, h.base]: UNKNOWN, score -(-.02) (boundary:left)<br>H `in_front_of(obj_017)` [h, h.base]: UNKNOWN, score -.02 (boundary:in_front_of)<br>H `behind(obj_017)` [h, h.base]: UNKNOWN, score -(-.02) (boundary:behind) |
| D25 | 7.2 | B; s=-.03; repeated on H with T=(0,s) | H `right(obj_012)` [h, h.base]: FALSE, score -.03 (failed:right)<br>H `left(obj_012)` [h, h.base]: TRUE, score -(-.03)<br>H `in_front_of(obj_018)` [h, h.base]: FALSE, score -.03 (failed:in_front_of)<br>H `behind(obj_018)` [h, h.base]: TRUE, score -(-.03) |
| D26 | 7.2 | V; T and A distinct IDs, both at (0,2) | V `right(obj_004, obj_001)` [v, v.base]: UNKNOWN, score 0 (boundary:right)<br>V `left(obj_004, obj_001)` [v, v.base]: UNKNOWN, score 0 (boundary:left)<br>V `in_front_of(obj_004, obj_001)` [v, v.base]: UNKNOWN, score 0 (boundary:in_front_of)<br>V `behind(obj_004, obj_001)` [v, v.base]: UNKNOWN, score 0 (boundary:behind) |
| D27 | 7.2 | V and I; target_id = anchor_id, an existing object | V `right(obj_001, obj_001)` [v, v.base]: FALSE (not_distinct)<br>V `left(obj_001, obj_001)` [v, v.base]: FALSE (not_distinct)<br>V `in_front_of(obj_001, obj_001)` [v, v.base]: FALSE (not_distinct)<br>V `behind(obj_001, obj_001)` [v, v.base]: FALSE (not_distinct)<br>I `right(obj_001, obj_001)` [i]: FALSE (not_distinct)<br>I `left(obj_001, obj_001)` [i]: FALSE (not_distinct)<br>I `in_front_of(obj_001, obj_001)` [i]: FALSE (not_distinct)<br>I `behind(obj_001, obj_001)` [i]: FALSE (not_distinct) |
| D28 | 7.3 | H; T=(1,0), target centre unknown; also missing target centre in V and I | H `right(obj_019)` [h, h.base]: UNKNOWN (missing_centre:obj_019)<br>V `right(obj_015, obj_001)` [v, v.base]: UNKNOWN (missing_centre:obj_015)<br>I `right(obj_014, obj_001)` [i]: UNKNOWN (missing_centre:obj_014) |
| D29 | 7.3 | H; T=(1,0), user position unknown; also missing user position in V | H `right(obj_001)` [h, h.nopos]: UNKNOWN (missing_user_position:fx.dir.h.nopos)<br>V `right(obj_002, obj_001)` [v, v.nopos]: UNKNOWN (missing_user_position:fx.dir.v.nopos) |
| D30 | 7.3 | H; T=(1,0), heading unknown, rotation known | H `right(obj_001)` [h, h.nohead]: UNKNOWN (missing_heading:fx.dir.h.nohead) |
| D31 | 7.3 | V; T=(1,2), anchor centre unknown | V `right(obj_002, obj_007)` [v, v.base]: UNKNOWN (missing_centre:obj_007) |
| D32 | 7.3 | V; T=(1,2), heading and user rotation unknown | V `right(obj_002, obj_001)` [v, v.nohead]: TRUE, score 1 |
| D33 | 7.3 | I; T=(0,3), semantic front unknown, box rotation known | I `in_front_of(obj_002, obj_012)` [i]: UNKNOWN (missing_semantic_front:obj_012) |
| D34 | 7.3 | I; T=(0,3), box size and rotation unknown, semantic front known | I `in_front_of(obj_002, obj_013)` [i]: TRUE, score 3-2 |
| D35 | 7.3 | H; T=(1,0), restricted profile | H `right(obj_001)` [h.r, h.base]: TRUE, score 1 |
| D36 | 7.3 | V; T=(0,3), restricted profile | V `behind(obj_003, obj_001)` [v.r, v.base]: TRUE, score -(2-3) |
| D37 | 7.3 | I; T=(0,3), restricted profile with front withheld | I `in_front_of(obj_002, obj_001)` [i.r]: UNKNOWN (missing_semantic_front:obj_001) |
| D38 | 7.4 | V; U=(0,2), A=(0,2), T=(1,2); repeated with user z = 0.3 and anchor z = 1.2 | V `right(obj_002, obj_001)` [v, v.at_anchor]: UNKNOWN (degenerate_viewer_anchor)<br>V `right(obj_002, obj_014)` [v, v.at_anchor_low]: UNKNOWN (degenerate_viewer_anchor) |
| D39 | 7.4 | V; U=(0,0), A=(0,.02), T=(1,.02); plus the strictly-below cutoff L=.01 | V `right(obj_009, obj_008)` [v, v.base]: UNKNOWN (degenerate_viewer_anchor)<br>V `right(obj_013, obj_012)` [v, v.base]: UNKNOWN (degenerate_viewer_anchor) |
| D40 | 7.4 | V; U=(0,0), A=(0,.03), T=(1,.03) | V `right(obj_011, obj_010)` [v, v.base]: TRUE, score 1 |
| D41 | 7.4 | I; A=(0,0), F=(0,0,1), T=(1,0) | I `in_front_of(obj_008, obj_007)` [i]: UNKNOWN (vertical_semantic_front) |
| D42 | 7.4 | I; A=(0,0), F=(.1,0,sqrt(.99)), T=(1,0); plus the strictly-below cutoff F=(.06,0,...) | I `in_front_of(obj_008, obj_009)` [i]: UNKNOWN (vertical_semantic_front)<br>I `in_front_of(obj_008, obj_011)` [i]: UNKNOWN (vertical_semantic_front) |
| D43 | 7.4 | I; A=(0,0), F=(.28,0,.96), T=(1,0) | I `in_front_of(obj_008, obj_010)` [i]: TRUE, score 1 |
