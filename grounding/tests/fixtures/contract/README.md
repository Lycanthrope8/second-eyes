# Contract test fixtures

Test data for `grounding/tests/test_contract.py` (A2.1a, D66). Not training data.

`valid/` holds the a17 scene in both evidence profiles, the a17 scene again as a format-v2 dataset scene (D74), two commands and the category map, and, since A4, a format-v3 perceived scene of a bedroom with its own category map (`scene.bedroom.perceived_v3.json`, `category-map.bedroom.json`: illustrative test values, not measurements); every file passes, alone and together. The object centres come from A1's a17 prompts, which add box_1 to the example memory in proposal Revision 3, section 4.3; the user's position and 4 degree heading come from section 4.3. The sizes, rotations, fronts, colours and the unmapped floor lamp are illustrative test values, and the size priors in the restricted scene are made up for the test, not built from data.

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
| `E_FRAME_SOURCE__fixture_kind_source.json` | `valid/scene.a17.dataset_v2.json` | the conversion's source `fx` is a fixture source, not a dataset |  |
| `E_FRAME_SOURCE__unresolved_source.json` | `valid/scene.a17.dataset_v2.json` | the conversion names source `dataset_a`, which isn't in sources |  |
| `E_FRAME_CONVERSION__other_source_frame.json` | `valid/scene.a17.dataset_v2.json` | the conversion's source frame is `dataset.native`, not the scene frame `scene` |  |
| `E_CATEGORY_UNMAPPED__standard_label.json` | one-object scene (annotated) | the table's standard label is desk | `valid/category-map.fx.json` |
| `E_DUPLICATE_RECORD__same_scene.json` | `valid/scene.a17.annotated.json` | a second copy of scene.a17.annotated.json | `valid/scene.a17.annotated.json` |
| `E_PROFILE_PERCEIVED__assumed_centre.json` | `valid/scene.bedroom.perceived_v3.json` | obj_001's centre is assumed from the size table; a perceived centre is inferred from the scan |  |
| `E_PROFILE_PERCEIVED__single_position_without_assumption.json` | `valid/scene.bedroom.perceived_v3.json` | obj_002, placed from one position, names no assumption for its centre |  |
| `E_PROFILE_PERCEIVED__measured_rotation.json` | `valid/scene.bedroom.perceived_v3.json` | obj_001's rotation is measured; a perceived rotation is only inferred from the scan, as its 3D box's turn |  |
| `E_ASSOCIATION__duplicate_not_listed_back.json` | `valid/scene.bedroom.perceived_v3.json` | obj_003 no longer lists obj_002 back, though obj_002 lists it |  |
| `E_ASSOCIATION__single_position_from_two_positions.json` | `valid/scene.bedroom.perceived_v3.json` | obj_005 is single_position but was seen from two positions |  |
| `E_FRAME_SOURCE__v3_size_table_conversion.json` | `valid/scene.bedroom.perceived_v3.json` | the conversion names the size table, not the sensor capture |  |
| `E_EVIDENCE_SOURCE__v3_object_from_size_table.json` | `valid/scene.bedroom.perceived_v3.json` | obj_001's observation names the size table as its source |  |
| `E_SCHEMA_ENUM__v3_dataset_source.json` | `valid/scene.bedroom.perceived_v3.json` | sources gain a dataset source, a kind v3 doesn't take |  |
| `E_SCHEMA_CONST__v3_identity_conversion.json` | `valid/scene.bedroom.perceived_v3.json` | the conversion's matrix is the identity, not the basis change |  |
| `E_ASSOCIATION__more_positions_than_keyframes.json` | `valid/scene.bedroom.perceived_v3.json` | obj_001 was seen from 3 positions in only 2 keyframes |  |
| `E_ASSOCIATION__multi_position_from_one_position.json` | `valid/scene.bedroom.perceived_v3.json` | obj_005 is multi_position but was seen from one position |  |
| `E_ASSOCIATION__duplicate_of_itself.json` | `valid/scene.bedroom.perceived_v3.json` | obj_002 lists itself as a possible duplicate |  |
| `E_ASSOCIATION__duplicate_outside_scene.json` | `valid/scene.bedroom.perceived_v3.json` | obj_002 lists obj_009, which isn't in the scene |  |
| `E_ASSOCIATION__duplicate_without_partner.json` | `valid/scene.bedroom.perceived_v3.json` | obj_002 is a possible duplicate that names no partner (so obj_003 isn't listed back either) |  |
| `E_PROFILE_PERCEIVED__unknown_size_not_estimated.json` | `valid/scene.bedroom.perceived_v3.json` | obj_001's size is unknown as not_estimated; a perceived size is a prior or no_permitted_prior |  |
| `E_PROFILE_PERCEIVED__capture_time_inferred.json` | `valid/scene.bedroom.perceived_v3.json` | obj_001's capture time is inferred; a capture time is measured |  |
| `E_RECORD_TYPE__command_v2_without_typed_source.json` | a command labelled format 2 without a `user_input` source: format 2 is only for commands typed on the headset, so it is unsupported, as before |
| `E_SCHEMA_ENUM__command_v1_user_input_source.json` | a format-1 command with a `user_input` source, which only format 2 has |
