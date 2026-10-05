"""Semantic and cross-record rules of the scene contract, v1 (A2.1a, D66).

The JSON Schemas in schemas/ check structure; these functions check what a schema can't: finite numbers, identities
and references, geometry, evidence provenance, the evidence profiles, the user pose, category maps, and agreement
between records validated together. validate.py runs them only on records that passed their schema. Rules and codes:
docs/scene-contract.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# v1 validation tolerance for stored geometry: unit quaternions and vectors, yaw-only tilt, identity rotations (D66).
# A data-integrity choice, not a relation threshold; relation thresholds and tolerances stay open (spec §10.2).
TOLERANCE = 1e-6

CODES = {
    # parsing and record type (validate.py)
    "E_PARSE_ENCODING": "the file is not UTF-8 without a byte-order mark",
    "E_PARSE_JSON": "the file is not well-formed JSON",
    "E_PARSE_DUPLICATE_KEY": "an object repeats a key",
    "E_NONFINITE": "a number is NaN, infinite, or too large for a double",
    "E_RECORD_TYPE": "not a JSON object with a supported record_type and schema_version",
    # schema (validate.py), by JSON Schema keyword
    "E_SCHEMA_REQUIRED": "a required field is missing (for a known value: its value or evidence)",
    "E_SCHEMA_ADDITIONAL": "a field the format doesn't define (for an unknown value: a value)",
    "E_SCHEMA_TYPE": "a value has the wrong JSON type",
    "E_SCHEMA_ENUM": "a value is not one of the allowed words",
    "E_SCHEMA_CONST": "a value differs from the only value v1 allows",
    "E_SCHEMA_PATTERN": "a string doesn't match its pattern",
    "E_SCHEMA_LENGTH": "an array or string has the wrong length",
    "E_SCHEMA_RANGE": "a number is out of range",
    "E_SCHEMA_UNIQUE": "an array repeats an item that must be unique",
    "E_SCHEMA_OTHER": "another schema rule failed",
    # semantic rules within one record
    "E_DUPLICATE_ID": "two objects in a scene share an object_id",
    "E_DUPLICATE_REGISTRY": "a sources or assumptions list repeats an ID",
    "E_UNRESOLVED_SOURCE": "a source ID is not in the record's sources list",
    "E_UNRESOLVED_ASSUMPTION": "an assumption ID is not in the record's assumptions list",
    "E_EVIDENCE_SOURCE": "an evidence kind or unknown reason doesn't fit the kind of source it names",
    "E_QUAT_NORM": "a quaternion is not of unit length",
    "E_QUAT_YAW_TILT": "a yaw-only quaternion rotates about x or y",
    "E_QUAT_AXIS_ALIGNED": "an axis-aligned rotation is not the identity",
    "E_UNIT_VECTOR": "a semantic front or heading is not of unit length",
    "E_PROFILE_WITHHELD": "a restricted scene shows orientation, semantic front or colours",
    "E_PROFILE_SIZE": "a restricted scene's size is neither a category prior nor unknown for lack of one",
    "E_PROFILE_ANNOTATED": "an annotated scene contains an assumed value or a profile's unknown",
    "E_POSE_RESERVED": "pose_kind quest_head is reserved; v1 doesn't accept live headset poses",
    "E_POSE_NONE_KNOWN": "pose_kind none with a known pose component",
    "E_POSE_EVIDENCE": "an authored or dataset pose component isn't annotated by a source of its kind",
    "E_HEADING_SOURCE": "the heading's source doesn't fit the pose kind",
    "E_POSE_REVISION": "the pose's scene_revision differs from the command's",
    "E_MAP_VOCAB": "a category map entry's model label is not in its model vocabulary",
    "E_MAP_DUPLICATE": "a category map lists a source label twice",
    # cross-record rules, for records validated together
    "E_DUPLICATE_RECORD": "two records have the same identity",
    "E_SCENE_MISMATCH": "a command's scene is present, but not at the command's revision",
    "E_FRAME_MISMATCH": "a command's pose frame differs from its scene's frame",
    "E_FRAME_SOURCE": "a scene v2 conversion's source_id doesn't name a dataset source in sources",
    "E_FRAME_CONVERSION": "a scene v2 identity conversion's source_frame_id isn't the scene's frame_id",
    "E_CATEGORY_UNMAPPED": "an object's raw label, standard and model labels are not an entry of the scene's map",
    "W_NOT_CROSS_CHECKED": "warning: a referenced scene or map was not among the records, so that check didn't run",
}

# Which source kinds may stand behind each evidence kind (known values) and each reason (unknown values), in v1.
KNOWN_SOURCE_KINDS = {
    "annotated": {"dataset", "fixture"},
    "inferred": {"dataset", "fixture", "prior_table"},
    "assumed": {"prior_table"},
    "measured": set(),          # v1 has no source kind for live perception
}
UNKNOWN_SOURCE_KINDS = {
    "not_in_source": {"dataset", "fixture"},
    "unmapped_label": {"dataset", "fixture"},
    "source_invalid": {"dataset", "fixture"},
    "withheld_by_profile": {"profile"},
    "no_permitted_prior": {"prior_table", "profile"},
    "not_estimated": None,      # any source kind
}
OBJECT_SOURCE_KINDS = {"dataset", "fixture"}
POSE_SOURCE_KIND = {"authored": "fixture", "dataset_camera": "dataset", "dataset_viewpoint": "dataset"}
POSE_HEADING_SOURCE = {"authored": "authored", "dataset_camera": "camera_yaw", "dataset_viewpoint": "camera_yaw"}


@dataclass(frozen=True)
class Issue:
    file: str
    path: str
    code: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.code.startswith("E_")

    def line(self) -> str:
        return f"{self.file}: {self.path}: {self.code}: {self.message}"


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def _numbers(value, path):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _numbers(v, f"{path}[{i}]")


def _norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


class _Record:
    """One record's issues, with its sources and assumptions lists."""

    def __init__(self, file: str, record: dict):
        self.file, self.record, self.issues = file, record, []
        for path, x in _numbers(record, "$"):
            if not _finite(x):
                self.add(path, "E_NONFINITE", "number is too large for a double")
        self.sources, self.assumptions = {}, set()
        for i, s in enumerate(record.get("sources", [])):
            if s["source_id"] in self.sources:
                self.add(f"$.sources[{i}].source_id", "E_DUPLICATE_REGISTRY", f"source {s['source_id']!r} repeated")
            else:
                self.sources[s["source_id"]] = s["kind"]
        for i, a in enumerate(record.get("assumptions", [])):
            if a["assumption_id"] in self.assumptions:
                self.add(f"$.assumptions[{i}].assumption_id", "E_DUPLICATE_REGISTRY",
                         f"assumption {a['assumption_id']!r} repeated")
            else:
                self.assumptions.add(a["assumption_id"])

    def add(self, path: str, code: str, message: str) -> None:
        self.issues.append(Issue(self.file, path, code, message))

    def source_kind(self, path: str, source_id: str):
        """The kind of a named source, or None after reporting that it doesn't resolve."""
        if source_id not in self.sources:
            self.add(path, "E_UNRESOLVED_SOURCE", f"source {source_id!r} is not in sources")
            return None
        return self.sources[source_id]

    def references(self, path: str, w: dict):
        """Resolve a value's source and assumptions; returns the source's kind, or None."""
        if w["state"] == "known":
            for j, a in enumerate(w["evidence"]["assumptions"]):
                if a not in self.assumptions:
                    self.add(f"{path}.evidence.assumptions[{j}]", "E_UNRESOLVED_ASSUMPTION",
                             f"assumption {a!r} is not in assumptions")
            return self.source_kind(f"{path}.evidence.source", w["evidence"]["source"])
        return self.source_kind(f"{path}.source", w["source"])

    def provenance(self, path: str, w: dict, kind) -> None:
        """Check that an evidence kind or unknown reason fits the kind of its source (v1 table)."""
        if kind is None:
            return
        if w["state"] == "known":
            ev = w["evidence"]["kind"]
            if kind not in KNOWN_SOURCE_KINDS[ev]:
                why = ("v1 has no source kind for measured values" if ev == "measured"
                       else f"{ev} evidence can't come from a {kind} source")
                self.add(f"{path}.evidence", "E_EVIDENCE_SOURCE", why)
        else:
            allowed = UNKNOWN_SOURCE_KINDS[w["reason"]]
            if allowed is not None and kind not in allowed:
                self.add(f"{path}.source", "E_EVIDENCE_SOURCE",
                         f"reason {w['reason']} can't come from a {kind} source")

    def quaternion(self, path: str, w: dict) -> None:
        if w["state"] != "known":
            return
        q, support = w["value"], w["support"]
        n = _norm(q)
        if abs(n - 1.0) > TOLERANCE:
            self.add(f"{path}.value", "E_QUAT_NORM", f"norm {n:.9g} is not 1 within {TOLERANCE:g}")
        elif support == "yaw_only" and (abs(q[0]) > TOLERANCE or abs(q[1]) > TOLERANCE):
            self.add(f"{path}.value", "E_QUAT_YAW_TILT", f"x = {q[0]:g}, y = {q[1]:g} must be 0 for yaw_only")
        elif support == "axis_aligned" and (max(abs(q[0]), abs(q[1]), abs(q[2])) > TOLERANCE
                                            or abs(abs(q[3]) - 1.0) > TOLERANCE):
            self.add(f"{path}.value", "E_QUAT_AXIS_ALIGNED", f"{q} is not the identity")

    def unit_vector(self, path: str, w: dict) -> None:
        if w["state"] == "known":
            n = _norm(w["value"])
            if abs(n - 1.0) > TOLERANCE:
                self.add(f"{path}.value", "E_UNIT_VECTOR", f"length {n:.9g} is not 1 within {TOLERANCE:g}")


def _object_values(obj: dict, base: str):
    g, a, o = obj["geometry"], obj["attributes"], obj["observation"]
    return [
        (f"{base}.category", obj["category"]),
        (f"{base}.geometry.center_m", g["center_m"]),
        (f"{base}.geometry.size_m", g["size_m"]),
        (f"{base}.geometry.rotation_xyzw", g["rotation_xyzw"]),
        (f"{base}.semantic_front", obj["semantic_front"]),
        (f"{base}.attributes.colours", a["colours"]),
        (f"{base}.observation.capture_time", o["capture_time"]),
        (f"{base}.observation.detector_confidence", o["detector_confidence"]),
        (f"{base}.observation.geometry_uncertainty", o["geometry_uncertainty"]),
    ]


def _scene(r: _Record) -> None:
    scene, seen = r.record, {}
    profile = scene["evidence_profile"]
    if scene["schema_version"] == 2:  # the dataset-frame representation (D74); v1 frames carry no conversion record
        frame = scene["coordinate_frame"]
        conversion = frame["conversion"]
        if r.sources.get(conversion["source_id"]) != "dataset":  # read directly: an unresolved ID gives only this code
            r.add("$.coordinate_frame.conversion.source_id", "E_FRAME_SOURCE",
                  f"conversion source {conversion['source_id']!r} is not a dataset source in sources")
        if conversion["source_frame_id"] != frame["frame_id"]:
            r.add("$.coordinate_frame.conversion.source_frame_id", "E_FRAME_CONVERSION",
                  f"an identity conversion's source frame {conversion['source_frame_id']!r} must be the scene frame "
                  f"{frame['frame_id']!r}")
    for i, obj in enumerate(scene["objects"]):
        base = f"$.objects[{i}]"
        if obj["object_id"] in seen:
            r.add(f"{base}.object_id", "E_DUPLICATE_ID", f"{obj['object_id']} is also objects[{seen[obj['object_id']]}]")
        else:
            seen[obj["object_id"]] = i
        for path, sid in [(f"{base}.source_ref.source_id", obj["source_ref"]["source_id"]),
                          (f"{base}.observation.source", obj["observation"]["source"])]:
            kind = r.source_kind(path, sid)
            if kind is not None and kind not in OBJECT_SOURCE_KINDS:
                r.add(path, "E_EVIDENCE_SOURCE", f"objects come from a dataset or fixture, not a {kind} source")
        for path, w in _object_values(obj, base):
            r.provenance(path, w, r.references(path, w))
        r.quaternion(f"{base}.geometry.rotation_xyzw", obj["geometry"]["rotation_xyzw"])
        r.unit_vector(f"{base}.semantic_front", obj["semantic_front"])
        if profile == "restricted":
            for path, w in [(f"{base}.geometry.rotation_xyzw", obj["geometry"]["rotation_xyzw"]),
                            (f"{base}.semantic_front", obj["semantic_front"]),
                            (f"{base}.attributes.colours", obj["attributes"]["colours"])]:
                if not (w["state"] == "unknown" and w["reason"] == "withheld_by_profile"):
                    r.add(path, "E_PROFILE_WITHHELD", "the restricted profile withholds this field")
            size = obj["geometry"]["size_m"]
            if not ((size["state"] == "known" and size["evidence"]["kind"] == "assumed")
                    or (size["state"] == "unknown" and size["reason"] == "no_permitted_prior")):
                r.add(f"{base}.geometry.size_m", "E_PROFILE_SIZE",
                      "restricted sizes are a category prior (assumed) or unknown (no_permitted_prior)")
        else:
            for path, w in _object_values(obj, base):
                if ((w["state"] == "known" and w["evidence"]["kind"] == "assumed")
                        or (w["state"] == "unknown" and w["reason"] in ("withheld_by_profile", "no_permitted_prior"))):
                    r.add(path, "E_PROFILE_ANNOTATED", "the annotated profile has no assumed values or profile unknowns")


def _command(r: _Record) -> None:
    cmd = r.record
    pose = cmd["user_pose"]
    kind = pose["pose_kind"]
    values = [(f"$.user_pose.{k}", pose[k]) for k in ("position_m", "rotation_xyzw", "heading_xy", "snapshot_time")]
    if pose["scene_revision"] != cmd["scene_revision"]:
        r.add("$.user_pose.scene_revision", "E_POSE_REVISION",
              f"{pose['scene_revision']} differs from the command's {cmd['scene_revision']}")
    if kind == "quest_head":
        r.add("$.user_pose.pose_kind", "E_POSE_RESERVED", "quest_head is reserved for a later version")
        return
    for path, w in values:
        source_kind = r.references(path, w)
        if w["state"] == "unknown":
            r.provenance(path, w, source_kind)
        elif kind == "none":
            r.add(path, "E_POSE_NONE_KNOWN", "pose_kind none has no known components")
        elif w["evidence"]["kind"] != "annotated" or (source_kind is not None
                                                       and source_kind != POSE_SOURCE_KIND[kind]):
            r.add(f"{path}.evidence", "E_POSE_EVIDENCE",
                  f"a {kind} pose is annotated by a {POSE_SOURCE_KIND[kind]} source in v1")
    heading = pose["heading_xy"]
    if kind != "none" and heading["state"] == "known" and heading["heading_source"] != POSE_HEADING_SOURCE[kind]:
        r.add("$.user_pose.heading_xy.heading_source", "E_HEADING_SOURCE",
              f"a {kind} pose's heading comes from {POSE_HEADING_SOURCE[kind]}")
    r.quaternion("$.user_pose.rotation_xyzw", pose["rotation_xyzw"])
    r.unit_vector("$.user_pose.heading_xy", heading)


def _category_map(r: _Record) -> None:
    vocabulary, seen = set(r.record["model_vocabulary"]), {}
    for i, e in enumerate(r.record["entries"]):
        if e["model"] not in vocabulary:
            r.add(f"$.entries[{i}].model", "E_MAP_VOCAB", f"{e['model']!r} is not in model_vocabulary")
        if e["source_label"] in seen:
            r.add(f"$.entries[{i}].source_label", "E_MAP_DUPLICATE",
                  f"{e['source_label']!r} is also entries[{seen[e['source_label']]}]")
        else:
            seen[e["source_label"]] = i


def check_record(file: str, record: dict) -> list:
    """Semantic rules for one record that has passed its schema."""
    r = _Record(file, record)
    {"scene": _scene, "command_context": _command, "category_map": _category_map}[record["record_type"]](r)
    return r.issues


def check_cross(records: list) -> list:
    """Rules between records validated together; records is a list of (file, record) that passed alone."""
    issues = []
    scenes, by_scene_id, commands, maps = {}, {}, {}, {}
    for file, rec in records:
        rt = rec["record_type"]
        if rt == "scene":
            key, table = (rec["scene_id"], rec["scene_revision"], rec["evidence_profile"]), scenes
        elif rt == "command_context":
            key, table = rec["command_id"], commands
        else:
            key, table = rec["map_id"], maps
        if key in table:
            issues.append(Issue(file, "$", "E_DUPLICATE_RECORD", f"{rt} {key} is also in {table[key][0]}"))
            continue
        table[key] = (file, rec)
        if rt == "scene":
            by_scene_id.setdefault(rec["scene_id"], []).append(rec)
    for file, cmd in commands.values():
        candidates = by_scene_id.get(cmd["scene_id"], [])
        same = [s for s in candidates if s["scene_revision"] == cmd["scene_revision"]]
        if not candidates:
            issues.append(Issue(file, "$.scene_id", "W_NOT_CROSS_CHECKED", f"scene {cmd['scene_id']} not given"))
        elif not same:
            issues.append(Issue(file, "$.scene_revision", "E_SCENE_MISMATCH",
                                f"scene {cmd['scene_id']} is given only at other revisions"))
        elif any(s["coordinate_frame"]["frame_id"] != cmd["user_pose"]["frame_id"] for s in same):
            issues.append(Issue(file, "$.user_pose.frame_id", "E_FRAME_MISMATCH",
                                f"{cmd['user_pose']['frame_id']!r} is not the scene's frame"))
    for file, scene in scenes.values():
        found = maps.get(scene["category_map"])
        if found is None:
            issues.append(Issue(file, "$.category_map", "W_NOT_CROSS_CHECKED",
                                f"category map {scene['category_map']} not given"))
            continue
        entries = {(e["source_label"], e["standard"], e["model"]) for e in found[1]["entries"]}
        for i, obj in enumerate(scene["objects"]):
            c = obj["category"]
            if c["state"] == "known":
                triple = (obj["source_ref"]["source_label"], c["value"]["standard"], c["value"]["model"])
                if triple not in entries:
                    issues.append(Issue(file, f"$.objects[{i}].category", "E_CATEGORY_UNMAPPED",
                                        f"{triple} is not an entry of {scene['category_map']}"))
    return issues
