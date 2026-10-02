# Contract test fixtures

Test data for `grounding/tests/test_contract.py` (A2.1a, D66). Not training data.

`valid/` holds the a17 scene in both evidence profiles, two commands and the category map; every file passes, alone and together. The object centres come from A1's a17 prompts, which add box_1 to the example memory in proposal Revision 3, section 4.3; the user's position and 4 degree heading come from section 4.3. The sizes, rotations, fronts, colours and the unmapped floor lamp are illustrative test values, and the size priors in the restricted scene are made up for the test, not built from data.

`invalid/` holds one file per rule. Each is a valid record with one change, or for two record-type cases a minimal file, named `<expected code>__<case>.json`, and must fail with exactly that code. Cross-record cases are checked together with the partner file named below. "One-object scene" is `fixture.min`: the a17 table alone, in the profile shown.

| File | Made from | Change | Checked with |
|---|---|---|---|
| `E_PARSE_ENCODING__byte_order_mark.json` | `valid/command.a17.c001.json` | the file starts with a UTF-8 byte-order mark |  |
| `E_PARSE_JSON__truncated.json` | `valid/command.a17.c001.json` | the file stops halfway |  |
| `E_PARSE_DUPLICATE_KEY__text_twice.json` | `valid/command.a17.c001.json` | the command object has "text" twice |  |
| `E_NONFINITE__nan.json` | `valid/command.a17.c001.json` | the position's x is NaN |  |
| `E_NONFINITE__overflow.json` | `valid/command.a17.c001.json` | the position's x is 1e400 |  |
| `E_NONFINITE__huge_integer.json` | `valid/command.a17.c001.json` | the position's x is an integer with 400 digits |  |
| `E_RECORD_TYPE__unknown_record_type.json` | `valid/command.a17.c001.json` | record_type is "command" |  |
| `E_RECORD_TYPE__future_version.json` | `valid/command.a17.c001.json` | schema_version is 2 |  |
| `E_RECORD_TYPE__null_file.json` | none | the file holds only `null` (regression, A2.1a correction) |  |
| `E_RECORD_TYPE__list_record_type.json` | none | `{"record_type": [], "schema_version": 1}` (regression, A2.1a correction) |  |
| `E_SCHEMA_REQUIRED__missing_user_pose.json` | `valid/command.a17.c001.json` | user_pose is missing |  |
| `E_SCHEMA_REQUIRED__known_without_value.json` | `valid/command.a17.c001.json` | the known position has no value |  |
| `E_SCHEMA_REQUIRED__known_without_evidence.json` | `valid/command.a17.c001.json` | the known position has no evidence |  |
| `E_SCHEMA_ADDITIONAL__unknown_with_value.json` | `valid/command.a17.c001.json` | the unknown snapshot time also carries a value |  |
| `E_SCHEMA_ADDITIONAL__answer_in_command.json` | `valid/command.a17.c001.json` | the command carries an answer field, target_id |  |
| `E_SCHEMA_TYPE__revision_as_string.json` | `valid/command.a17.c001.json` | scene_revision is the string "0" |  |
| `E_SCHEMA_ENUM__unknown_reason.json` | `valid/command.a17.c001.json` | an unknown's reason is "missing" |  |
| `E_SCHEMA_RANGE__zero_size.json` | one-object scene (annotated) | a size component is 0 |  |
| `E_SCHEMA_PATTERN__category_bearing_id.json` | one-object scene (annotated) | object_id is "table_1" |  |
| `E_SCHEMA_UNIQUE__repeated_colour.json` | one-object scene (annotated) | colours are ["brown", "brown"] |  |
| `E_SCHEMA_LENGTH__two_component_centre.json` | one-object scene (annotated) | center_m has two numbers |  |
| `E_SCHEMA_CONST__centimetres.json` | one-object scene (annotated) | units is "cm" |  |
| `E_SCHEMA_CONST__dataset_conversion.json` | one-object scene (annotated) | conversion is "from_source", which v1 doesn't accept yet |  |
| `E_DUPLICATE_ID__two_objects.json` | one-object scene (annotated) | two objects are obj_001 |  |
| `E_DUPLICATE_REGISTRY__repeated_source.json` | `valid/command.a17.c001.json` | sources lists fx twice |  |
| `E_UNRESOLVED_SOURCE__evidence_source.json` | `valid/command.a17.c001.json` | the position's evidence names source fx.missing |  |
| `E_UNRESOLVED_ASSUMPTION__evidence_assumption.json` | `valid/command.a17.c001.json` | the position's evidence names assumption registration_exact, which isn't listed |  |
| `E_EVIDENCE_SOURCE__annotated_from_prior.json` | one-object scene (annotated) | the centre is annotated but names a prior_table source |  |
| `E_EVIDENCE_SOURCE__measured_fixture_value.json` | one-object scene (annotated) | the centre's evidence kind is measured |  |
| `E_QUAT_NORM__scaled.json` | one-object scene (annotated) | the rotation is [0, 0, 0, 1.01] |  |
| `E_QUAT_YAW_TILT__tilted.json` | one-object scene (annotated) | a yaw_only rotation of unit length with x = 0.1 |  |
| `E_QUAT_AXIS_ALIGNED__yawed.json` | one-object scene (annotated) | an axis_aligned rotation that is a 30 degree yaw |  |
| `E_UNIT_VECTOR__front_length_2.json` | one-object scene (annotated) | the semantic front is [0, 2, 0] |  |
| `E_UNIT_VECTOR__heading_not_unit.json` | `valid/command.a17.c001.json` | the heading is [1, 1] |  |
| `E_PROFILE_WITHHELD__known_rotation.json` | one-object scene (restricted) | a restricted scene with a known rotation |  |
| `E_PROFILE_SIZE__annotated_size.json` | one-object scene (restricted) | a restricted scene with the annotated size |  |
| `E_PROFILE_ANNOTATED__assumed_size.json` | one-object scene (annotated) | an annotated scene with a prior's size |  |
| `E_POSE_RESERVED__quest_head.json` | `valid/command.a17.c001.json` | pose_kind is quest_head |  |
| `E_POSE_NONE_KNOWN__known_position.json` | `valid/command.a17.c002.json` | pose_kind none with a known position |  |
| `E_POSE_EVIDENCE__authored_measured.json` | `valid/command.a17.c001.json` | the authored position's evidence kind is measured |  |
| `E_POSE_EVIDENCE__dataset_measured.json` | `valid/command.a17.c001.json` | a dataset_viewpoint pose whose position's evidence kind is measured |  |
| `E_HEADING_SOURCE__camera_yaw_on_authored.json` | `valid/command.a17.c001.json` | an authored pose's heading_source is camera_yaw |  |
| `E_POSE_REVISION__pose_revision.json` | `valid/command.a17.c001.json` | the pose's scene_revision is 1, the command's 0 |  |
| `E_MAP_VOCAB__model_outside_vocabulary.json` | `valid/category-map.fx.json` | an entry maps stool to model label stool |  |
| `E_MAP_DUPLICATE__repeated_label.json` | `valid/category-map.fx.json` | chair is listed twice |  |
| `E_SCENE_MISMATCH__revision.json` | `valid/command.a17.c001.json` | the command refers to revision 1 of fixture.a17 | `valid/scene.a17.annotated.json` |
| `E_FRAME_MISMATCH__frame.json` | `valid/command.a17.c001.json` | the pose's frame is room | `valid/scene.a17.annotated.json` |
| `E_CATEGORY_UNMAPPED__standard_label.json` | one-object scene (annotated) | the table's standard label is desk | `valid/category-map.fx.json` |
| `E_DUPLICATE_RECORD__same_scene.json` | `valid/scene.a17.annotated.json` | a second copy of scene.a17.annotated.json | `valid/scene.a17.annotated.json` |
