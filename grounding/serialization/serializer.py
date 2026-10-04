"""Render validated scene and command-context records into coordinates_v2 or coordinates_relations_v2 (A2.1d, D69).

The boundary, in order (implementation brief section 3):
1. options are checked;
2. private copies of the records are validated together with the contract validator;
3. both configurations are loaded from their files for this call only;
4. the augmented work is counted from n, before any relation table, box or evaluator exists, and the cap is enforced;
5. the document is rendered, and its fully wrapped input is optionally token-checked.

Relation states come only from the accepted libraries (grounding.relations); the one computation done here is the
approved centre distance, norm(sub(center_b, center_a)). Nothing is cached between calls.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from ..contract import validate as contract
from ..relations import directions as D
from ..relations import geometry as g
from ..relations import predicates as P
from . import codec
from . import constants as K

STATE = {P.TRUE: "T", P.FALSE: "F", P.UNKNOWN: "U"}
MEASUREMENT_KINDS = ("exact", "test_double")


class SerializationInputError(ValueError):
    """Options, records or configuration the serializer can't accept. No model input was made.

    issues: dicts with label (which record or option), path, code and message, like the contract validator's.
    """

    def __init__(self, issues):
        self.issues = [dict(i) for i in issues]
        super().__init__("; ".join(f"{i['label']}: {i['path']}: {i['code']}: {i['message']}" for i in self.issues))


class TokenCheckError(RuntimeError):
    """The caller's token check failed or returned something invalid; the budget was not checked."""


@dataclass(frozen=True)
class TokenCheck:
    wrap: object                # callable(document: str) -> complete model input: str
    count_tokens: object        # callable(complete input: str) -> nonnegative int
    max_input_tokens: int       # already excludes any output or scoring reserve the caller keeps
    tokenizer_identity: str
    wrapper_identity: str
    measurement_kind: str       # "exact" or "test_double"


@dataclass(frozen=True)
class SerializationResult:
    status: str                 # ok, work_budget_exceeded or token_budget_exceeded
    format: str
    serializer_version: int
    static_prefix: object       # str, or None unless status is ok
    dynamic_suffix: object
    document: object
    metadata: dict


def _issue(label, path, code, message):
    return {"label": label, "path": path, "code": code, "message": message}


def canonical_sha256(value) -> str:
    """Snapshot identity: SHA-256 of JSON with sorted keys, compact separators, UTF-8, finite numbers only."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------------------- options
def _check_options(fmt, cap, token_check, category_maps, config_paths):
    issues = []
    if fmt not in K.FORMATS:
        issues.append(_issue("format", "$", "E_OPTION_FORMAT", f"{fmt!r} is not one of {', '.join(K.FORMATS)}"))
    plain_int = isinstance(cap, int) and not isinstance(cap, bool)
    if fmt == K.AUGMENTED and not (plain_int and cap >= 0):
        issues.append(_issue("max_relation_work_units", "$", "E_OPTION_WORK_CAP",
                             f"{K.AUGMENTED} needs a nonnegative integer cap, got {cap!r}"))
    elif fmt == K.COORDINATES and not (cap is None or (plain_int and cap >= 0)):
        issues.append(_issue("max_relation_work_units", "$", "E_OPTION_WORK_CAP",
                             f"the cap must be None or a nonnegative integer, got {cap!r}"))
    if token_check is not None:
        tc = token_check
        problems = []
        if not isinstance(tc, TokenCheck):
            problems.append("must be a TokenCheck")
        else:
            if not callable(tc.wrap) or not callable(tc.count_tokens):
                problems.append("wrap and count_tokens must be callable")
            if isinstance(tc.max_input_tokens, bool) or not isinstance(tc.max_input_tokens, int) \
                    or tc.max_input_tokens < 0:
                problems.append(f"max_input_tokens must be a nonnegative integer, got {tc.max_input_tokens!r}")
            for name in ("tokenizer_identity", "wrapper_identity"):
                if not isinstance(getattr(tc, name), str) or not getattr(tc, name).strip():
                    problems.append(f"{name} must be a nonempty string")
            if tc.measurement_kind not in MEASUREMENT_KINDS:
                problems.append(f"measurement_kind must be one of {MEASUREMENT_KINDS}, got {tc.measurement_kind!r}")
        issues += [_issue("token_check", "$", "E_OPTION_TOKEN_CHECK", p) for p in problems]
    if not isinstance(category_maps, (list, tuple)) or not all(isinstance(m, dict) for m in category_maps):
        issues.append(_issue("category_maps", "$", "E_OPTION_CATEGORY_MAPS", "must be a list or tuple of records"))
    for name, path in config_paths.items():
        if not isinstance(path, (str, Path)):
            issues.append(_issue(name, "$", "E_OPTION_CONFIG_PATH", f"must be a path, got {type(path).__name__}"))
    if issues:
        raise SerializationInputError(issues)


# ---------------------------------------------------------------------------------------------------- validation
def _validate(scene, command, maps):
    """Validate the private snapshots together; returns the warnings, raises on any error."""
    issues = []
    for label, record, kind in (("scene", scene, "scene"), ("command", command, "command_context")):
        if not isinstance(record, dict) or record.get("record_type") != kind:
            issues.append(_issue(label, "$.record_type", "E_RECORD_ROLE", f"the {label} argument must be a {kind} "
                                 "record"))
    for i, m in enumerate(maps):
        if m.get("record_type") != "category_map":
            issues.append(_issue(f"category_map[{i}]", "$.record_type", "E_RECORD_ROLE",
                                 "category_maps must hold category_map records"))
    if issues:
        raise SerializationInputError(issues)
    records = [("scene", scene), ("command", command)] + [(f"category_map[{i}]", m) for i, m in enumerate(maps)]
    found = contract.validate(records)
    errors = [_issue(i.file, i.path, i.code, i.message) for i in found if i.is_error]
    warnings = [_issue(i.file, i.path, i.code, i.message) for i in found if not i.is_error]
    if not errors:  # serializer-level checks, run only on records the validator accepted
        # The validator only warns when a command names another scene; here the two must match.
        if command["scene_id"] != scene["scene_id"] or command["scene_revision"] != scene["scene_revision"]:
            errors.append(_issue("command", "$.scene_id", "E_SCENE_MISMATCH",
                                 f"command is for {command['scene_id']} revision {command['scene_revision']}, "
                                 f"the scene is {scene['scene_id']} revision {scene['scene_revision']}"))
        if command["user_pose"]["frame_id"] != scene["coordinate_frame"]["frame_id"]:
            errors.append(_issue("command", "$.user_pose.frame_id", "E_FRAME_MISMATCH",
                                 f"pose frame {command['user_pose']['frame_id']} is not the scene's "
                                 f"{scene['coordinate_frame']['frame_id']}"))
        # D70: supplying maps means asking for cross-validation, so the scene's own map must be among them.
        # No maps at all keeps the validator's W_NOT_CROSS_CHECKED warning.
        supplied = sorted(m["map_id"] for m in maps)
        if maps and scene["category_map"] not in supplied:
            errors.append(_issue("scene", "$.category_map", "E_CATEGORY_MAP_MISMATCH",
                                 f"the scene needs category map {scene['category_map']}; supplied: "
                                 f"{', '.join(supplied)}"))
    if errors:
        raise SerializationInputError(errors)
    return warnings


def _load_configs(relation_path, direction_path):
    loaded = {}
    for label, path, loader in (("relation_config", relation_path, P.load_config),
                                ("direction_config", direction_path, D.load_direction_config)):
        try:
            before = Path(path).read_bytes()
            config = loader(Path(path))
            after = Path(path).read_bytes()
        except (OSError, ValueError, jsonschema.exceptions.ValidationError) as e:
            raise SerializationInputError([_issue(label, str(path), "E_CONFIG", f"{type(e).__name__}: {e}")]) from e
        if before != after:  # the hash in metadata must describe the bytes the loader parsed
            raise SerializationInputError([_issue(label, str(path), "E_CONFIG", "the file changed while loading")])
        loaded[label] = (config, _sha256_bytes(before), str(path))
    return loaded


# ---------------------------------------------------------------------------------------------------- projection
def _known(w):
    return w["value"] if w["state"] == "known" else None


def _conditional(w) -> bool:
    return w["state"] == "known" and (w["evidence"]["kind"] == "assumed" or bool(w["evidence"]["assumptions"]))


def _object_row(o):
    gm = o["geometry"]
    wrappers = {"category": o["category"], "colours": o["attributes"]["colours"], "center_m": gm["center_m"],
                "size_m": gm["size_m"], "rotation_xyzw": gm["rotation_xyzw"], "semantic_front": o["semantic_front"]}
    category = _known(o["category"])
    colours = _known(o["attributes"]["colours"])
    rotation = gm["rotation_xyzw"]
    return [o["object_id"], category["model"] if category is not None else None,
            sorted(colours) if colours is not None else None, _known(gm["center_m"]), _known(gm["size_m"]),
            _known(rotation), rotation["support"] if rotation["state"] == "known" else None,
            _known(o["semantic_front"]), [c for c in K.EVIDENCE_COLUMNS if _conditional(wrappers[c])]]


def _pose_record(command):
    pose = command["user_pose"]
    heading = pose["heading_xy"]
    return {"pose_kind": pose["pose_kind"], "position_m": _known(pose["position_m"]), "heading_xy": _known(heading),
            "heading_source": heading.get("heading_source") if heading["state"] == "known" else None,
            "conditional_fields": [n for n in K.POSE_CONDITIONAL_FIELDS if _conditional(pose[n])]}


def _header(fmt):
    return {"header": {"serializer_version": K.SERIALIZER_VERSION, "format": fmt, "axes": dict(K.AXES),
                       "object_columns": list(K.OBJECT_COLUMNS), "unknown": K.UNKNOWN_TEXT,
                       "conditional_fields": K.CONDITIONAL_FIELDS_TEXT, "coverage": K.COVERAGE}}


def _semantics(rel_cfg, dir_cfg):
    directions = {"direction_band_m": dir_cfg.direction_band_m,
                  "viewer_anchor_min_horizontal_m": dir_cfg.viewer_anchor_min_horizontal_m,
                  "semantic_front_min_horizontal_norm": dir_cfg.semantic_front_min_horizontal_norm}
    return {"semantics": {"definitions": K.DEFINITIONS,
                          "parameters": {"thresholds": dict(sorted(rel_cfg.thresholds.items())),
                                         "bands": dict(sorted(rel_cfg.bands.items())),
                                         "directions": dict(sorted(directions.items()))},
                          "categories": {k: sorted(rel_cfg.categories[k]) for k in ("support", "furniture",
                                                                                     "container")}}}


# -------------------------------------------------------------------------------------------- relation evidence
def _ref(ref, scene_key):
    if isinstance(ref, D.DirectionFieldRef):
        record_type, record_id, heading_source = ref.record_type, list(ref.record_id), ref.heading_source
    else:
        record_type, record_id, heading_source = "scene", list(scene_key), None
    return {"record_type": record_type, "record_id": record_id, "object_id": ref.object_id, "field": ref.field,
            "state": ref.state, "kind": ref.kind, "source": ref.source, "assumptions": list(ref.assumptions),
            "reason": ref.reason, "heading_source": heading_source}


def _consulted_conditional(result) -> bool:
    return any(r.state == "known" and (r.kind == "assumed" or bool(r.assumptions)) for r in result.inputs)


def _evidence(name, relation, frame, ids, result, scene_key):
    inputs = [_ref(r, scene_key) for r in result.inputs]
    entry = {"block": name, "relation": relation, "frame": frame, "objects": ids, "state": STATE[result.value],
             "conditional": _consulted_conditional(result), "inputs": inputs,
             "assumptions": sorted({(i["record_type"], tuple(i["record_id"]), a) for i in inputs
                                    if i["state"] == "known" for a in i["assumptions"]}),
             "reasons": list(result.reasons), "measures": dict(result.measures)}
    entry["assumptions"] = [[t, list(r), a] for t, r, a in entry["assumptions"]]
    if hasattr(result, "basis"):
        entry["basis"] = result.basis
    if hasattr(result, "requested_score_m"):
        entry["requested_score_m"] = result.requested_score_m
    return entry


def _call(rel, drel, command, ids, relation, frame, cell):
    if frame is None:
        return getattr(rel, relation)(*(ids[i] for i in cell))
    if frame == "object_intrinsic":
        return drel.evaluate(relation, ids[cell[0]], frame=frame, anchor_id=ids[cell[1]])
    if frame == "user_heading":
        return drel.evaluate(relation, ids[cell[0]], frame=frame, command=command)
    return drel.evaluate(relation, ids[cell[0]], frame=frame, anchor_id=ids[cell[1]], command=command)


def _relation_block(name, relation, frame, domain, ids, rel, drel, command, scene_key, tally):
    cells = {}
    for cell in codec.legal_cells(domain, len(ids)):
        result = _call(rel, drel, command, ids, relation, frame, cell)
        if result.value not in STATE:
            raise RuntimeError(f"{name}: a result outside the shared Truth enum")
        cells[cell] = (STATE[result.value], _consulted_conditional(result))
        tally["evidence"].append(_evidence(name, relation, frame, [ids[i] for i in cell], result, scene_key))
    tally["calls"][name] = len(cells)
    tally["states"][name] = {s: sum(1 for v, _ in cells.values() if v == s) for s in "TFU"}
    tally["conditional"][name] = sum(1 for _, c in cells.values() if c)
    return codec.encode_block(relation, frame, domain, len(ids), cells)


def _distance_block(objects, ids, scene_key, tally):
    centres = {o["object_id"]: o["geometry"]["center_m"] for o in objects}
    cells, performed = {}, 0
    for i, j in codec.legal_cells("unordered_pairs", len(ids)):
        wa, wb = centres[ids[i]], centres[ids[j]]
        if wa["state"] == "known" and wb["state"] == "known":
            value = g.norm(g.sub(wb["value"], wa["value"]))
            performed += 1
        else:
            value = None
        cells[(i, j)] = (value, _conditional(wa) or _conditional(wb))
        refs = [{"record_type": "scene", "record_id": list(scene_key), "object_id": oid, "field": "geometry.center_m",
                 "state": w["state"], "kind": w["evidence"]["kind"] if w["state"] == "known" else None,
                 "source": w["evidence"]["source"] if w["state"] == "known" else w["source"],
                 "assumptions": list(w["evidence"]["assumptions"]) if w["state"] == "known" else [],
                 "reason": None if w["state"] == "known" else w["reason"], "heading_source": None}
                for oid, w in ((ids[i], wa), (ids[j], wb))]
        tally["evidence"].append({"block": "center_distance_m", "objects": [ids[i], ids[j]], "distance_m": value,
                                  "conditional": cells[(i, j)][1], "inputs": refs,
                                  "assumptions": [["scene", list(scene_key), a] for r in refs for a in r["assumptions"]]})
    slots = len(cells)
    tally["calls"]["center_distance_m"] = slots
    tally["distances"] = {"slots": slots, "performed": performed, "missing": slots - performed}
    tally["conditional"]["center_distance_m"] = sum(1 for _, c in cells.values() if c)
    return codec.encode_distances(len(ids), cells)


# --------------------------------------------------------------------------------------------------- the entry point
def serialize(scene_record, command_record, *, format, relation_config_path, direction_config_path,
              max_relation_work_units=None, category_maps=(), token_check=None) -> SerializationResult:
    """Render one document; see the module docstring and docs/serialization.md."""
    started = time.perf_counter()
    _check_options(format, max_relation_work_units, token_check, category_maps,
                   {"relation_config_path": relation_config_path, "direction_config_path": direction_config_path})
    scene, command = copy.deepcopy(scene_record), copy.deepcopy(command_record)
    maps = [copy.deepcopy(m) for m in category_maps]
    warnings = _validate(scene, command, maps)
    configs = _load_configs(relation_config_path, direction_config_path)
    rel_cfg, dir_cfg = configs["relation_config"][0], configs["direction_config"][0]
    validated = time.perf_counter()

    objects = sorted(scene["objects"], key=lambda o: o["object_id"])
    ids = [o["object_id"] for o in objects]
    n = len(ids)
    augmented = format == K.AUGMENTED
    scene_key = (scene["scene_id"], scene["scene_revision"], scene["evidence_profile"])
    planned = K.work_slots(n) if augmented else 0
    planned_by_block = {}
    if augmented:
        for name, _, _, domain in K.STATIC_BLOCKS + K.DYNAMIC_BLOCKS:
            planned_by_block[name] = codec.legal_count(domain, n)
        planned_by_block["center_distance_m"] = codec.legal_count("unordered_pairs", n)
    meta = {
        "format": format, "serializer_version": K.SERIALIZER_VERSION,
        "scene": {"scene_id": scene["scene_id"], "scene_revision": scene["scene_revision"],
                  "evidence_profile": scene["evidence_profile"], "frame_id": scene["coordinate_frame"]["frame_id"],
                  "category_map": scene["category_map"], "snapshot_sha256": canonical_sha256(scene)},
        "command": {"command_id": command["command_id"], "snapshot_sha256": canonical_sha256(command)},
        "category_maps": [{"map_id": m.get("map_id"), "snapshot_sha256": canonical_sha256(m)} for m in maps],
        "snapshot_hash_rule": "SHA-256 of JSON with sorted keys, compact separators, UTF-8, finite numbers; "
                              "provenance, not a cache key",
        "validation": {"warnings": warnings},
        "configs": {label: {"identity": cfg.identity, "file": path, "file_sha256": digest}
                    for label, (cfg, digest, path) in configs.items()},
        "effective": {"thresholds": dict(sorted(rel_cfg.thresholds.items())), "bands": dict(sorted(rel_cfg.bands.items())),
                      "categories": {k: sorted(v) for k, v in sorted(rel_cfg.categories.items())},
                      "directions": {"direction_band_m": dir_cfg.direction_band_m,
                                     "viewer_anchor_min_horizontal_m": dir_cfg.viewer_anchor_min_horizontal_m,
                                     "semantic_front_min_horizontal_norm": dir_cfg.semantic_front_min_horizontal_norm}},
        "objects": ids, "n": n,
        "work": {"planned_slots": planned, "planned_by_block": planned_by_block, "cap": max_relation_work_units,
                 "status": "within_cap" if not augmented or planned <= max_relation_work_units else "exceeded"},
    }
    timing = {"validation_and_config_s": validated - started}
    meta["timing_s"] = timing
    if augmented and planned > max_relation_work_units:
        meta["token"] = {"token_count": None, "token_budget_status": "not_checked"}
        return SerializationResult("work_budget_exceeded", format, K.SERIALIZER_VERSION, None, None, None, meta)

    tally = {"calls": {}, "states": {}, "conditional": {}, "evidence": [], "distances": None}
    static = [enc_line(_header(format)), enc_line(_semantics(rel_cfg, dir_cfg)),
              enc_line({"objects": [_object_row(o) for o in objects]})]
    if augmented:
        rel = P.Relations(scene, rel_cfg)
        drel = D.DirectionalRelations(scene, dir_cfg)
        for name, relation, frame, domain in K.STATIC_BLOCKS:
            static.append(enc_line(_relation_block(name, relation, frame, domain, ids, rel, drel, command, scene_key,
                                                   tally)))
        static.append(enc_line(_distance_block(objects, ids, scene_key, tally)))
    built_static = time.perf_counter()
    dynamic = [enc_line({"pose": _pose_record(command)})]
    if augmented:
        for name, relation, frame, domain in K.DYNAMIC_BLOCKS:
            dynamic.append(enc_line(_relation_block(name, relation, frame, domain, ids, rel, drel, command, scene_key,
                                                    tally)))
    dynamic.append(enc_line({"command": command["text"]}))
    built_dynamic = time.perf_counter()
    timing["static_build_s"] = built_static - validated
    timing["dynamic_build_s"] = built_dynamic - built_static

    static_prefix, dynamic_suffix = "".join(static), "".join(dynamic)
    document = static_prefix + dynamic_suffix
    meta["work"]["actual_calls_by_block"] = dict(tally["calls"])
    meta["work"]["actual_total"] = sum(tally["calls"].values())
    meta["work"]["distances"] = tally["distances"] or {"slots": 0, "performed": 0, "missing": 0}
    meta["states_by_block"] = tally["states"]
    meta["conditional_by_block"] = tally["conditional"]
    meta["tuple_evidence"] = tally["evidence"]
    meta["text"] = {name: {"bytes": len(text.encode("utf-8")), "sha256": _sha256_bytes(text.encode("utf-8"))}
                    for name, text in (("static_prefix", static_prefix), ("dynamic_suffix", dynamic_suffix),
                                       ("document", document))}

    if token_check is None:
        meta["token"] = {"token_count": None, "token_budget_status": "not_checked"}
        return SerializationResult("ok", format, K.SERIALIZER_VERSION, static_prefix, dynamic_suffix, document, meta)
    checking = time.perf_counter()
    try:
        wrapped = token_check.wrap(document)
    except Exception as e:  # noqa: BLE001  (the caller's callback; reported, never treated as a pass)
        raise TokenCheckError(f"wrap failed: {type(e).__name__}: {e}") from e
    if not isinstance(wrapped, str):
        raise TokenCheckError(f"wrap returned {type(wrapped).__name__}, not str")
    try:
        count = token_check.count_tokens(wrapped)
    except Exception as e:  # noqa: BLE001
        raise TokenCheckError(f"count_tokens failed: {type(e).__name__}: {e}") from e
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TokenCheckError(f"count_tokens returned {count!r}, not a nonnegative integer")
    timing["token_check_s"] = time.perf_counter() - checking
    within = count <= token_check.max_input_tokens
    meta["token"] = {"token_count": count, "max_input_tokens": token_check.max_input_tokens,
                     "token_budget_status": "within_budget" if within else "exceeded",
                     "measurement_kind": token_check.measurement_kind,
                     "tokenizer_identity": token_check.tokenizer_identity,
                     "wrapper_identity": token_check.wrapper_identity,
                     "wrapped_sha256": _sha256_bytes(wrapped.encode("utf-8")),
                     "wrapped_bytes": len(wrapped.encode("utf-8")),
                     "scope": "the complete wrapped input, counted in one call; max_input_tokens already accounts for "
                              "any output or scoring reserve",
                     "note": "a test double, not a model measurement" if token_check.measurement_kind == "test_double"
                     else "measured with the caller's tokenizer"}
    if not within:
        return SerializationResult("token_budget_exceeded", format, K.SERIALIZER_VERSION, None, None, None, meta)
    return SerializationResult("ok", format, K.SERIALIZER_VERSION, static_prefix, dynamic_suffix, document, meta)


def enc_line(value) -> str:
    return codec.enc(value) + "\n"

