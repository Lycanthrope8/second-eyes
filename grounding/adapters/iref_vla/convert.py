"""The A2.2a conversion stages (D74): scene, commands and annotations, plus the graph cross-check.

convert_scene takes only the object CSV, the region CSV and the vocabulary, so no answer can reach scene evidence.
convert_commands reads only the statements' expression keys. convert_annotations keeps every source annotation
unchanged and checks it against the scene-side CSV values; crosscheck_graph compares the source's graph. Neither may
repair or replace anything: a disagreement fails the import.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass

from . import pinned as K
from .sources import Issues, decimal_id, label, number, read_csv, read_json

GROUP_FIELDS = ("r", "g", "b", "scheme", "scheme_percentage", "scheme_average_dist")
OBJECT_HEADER = ("object_id", "region_id", "raw_label", "nyu_id", "nyu40_id", "nyu_label", "nyu40_label",
                 "object_bbox_cx", "object_bbox_cy", "object_bbox_cz", "object_bbox_xlength", "object_bbox_ylength",
                 "object_bbox_zlength", "object_bbox_heading", "object_front_heading") + tuple(
    f"object_color_{f}{n}" for n in (1, 2, 3) for f in GROUP_FIELDS)
LEADING = 15
OBJECT_WIDTHS = {LEADING + 6, LEADING + 12, LEADING + 18}  # 21, 27, 33: only complete colour groups may be omitted
REGION_HEADER = ("region_id", "region_label", "region_bbox_cx", "region_bbox_cy", "region_bbox_cz",
                 "region_bbox_xlength", "region_bbox_ylength", "region_bbox_zlength", "region_bbox_heading")
VOCABULARY_HEADER = ("nyuId", "nyuClass")
RECORD_KEYS = {"target_index", "target_class", "target_position", "target_colors", "target_size", "target_color_used",
               "target_size_used", "distractor_ids", "relation", "relation_type", "anchors", "false_statements"}
ANCHOR_KEYS = {"index", "class", "position", "color", "size", "color_used", "size_used"}
FALSE_KEYS = {"false_target_color", "false_target_class", "false_anchors"}
FALSE_ANCHOR_KEYS = {"false_anchor_color", "false_anchor_class"}
GRAPH_OBJECT_KEYS = {"object_id", "raw_label", "nyu_id", "nyu40_id", "nyu_label", "nyu40_label", "color_vals",
                     "color_labels", "color_percentages", "bbox", "center", "volume", "size", "affordances"}
ANCHOR_KEY, FALSE_ANCHOR_KEY = re.compile(r"anchor_([1-9][0-9]*)"), re.compile(r"anchor([1-9][0-9]*)")
COORDINATE_TOLERANCE_M, VOLUME_RELATIVE_TOLERANCE = 1e-9, 1e-12


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_id(source_object_id: str) -> str:
    """Source integer n becomes obj_ + n padded to at least three digits; never renumbered."""
    return f"obj_{int(source_object_id):03d}"


def command_id(text: str) -> str:
    return K.COMMAND_PREFIX + sha256(text.encode("utf-8"))


def _evidence():
    return {"kind": "annotated", "source": K.SOURCE_ID, "assumptions": []}


def _known(value):
    return {"state": "known", "value": value, "evidence": _evidence()}


def _unknown(reason="not_in_source"):
    return {"state": "unknown", "reason": reason, "source": K.SOURCE_ID}


@dataclass
class SceneStage:
    """The scene-side outputs and what later stages may read: the source-ID mapping and scene-side CSV values."""
    scene: dict
    category_map: dict
    inventory_views: dict
    object_ids: dict      # source object ID -> obj_ ID
    short_rows: dict      # source object ID -> row width, for rows whose trailing colour groups are omitted
    source: dict          # source object ID -> parsed CSV values, for cross-checks only
    region: dict          # {"id", "label"}
    hashes: dict          # role -> SHA-256 of the scene-side input bytes


# ------------------------------------------------------------------------------------------------- scene stage
def _vocabulary(data, issues):
    f = K.FILES["vocabulary"]["name"]
    rows = read_csv(f, data, VOCABULARY_HEADER, {2}, issues)
    issues.raise_if_any()
    by_id, by_label = {}, {}
    for n, (nid, name) in enumerate(rows, start=2):
        nid, name = decimal_id(nid, f, f"row {n}, nyuId", issues), label(name, f, f"row {n}, nyuClass", issues)
        if nid is None or name is None:
            continue
        if nid in by_id:
            issues.add(f, f"row {n}", "E_IREF_SOURCE_VALUE", f"NYU ID {nid} is listed twice")
        if name in by_label:
            issues.add(f, f"row {n}", "E_IREF_SOURCE_VALUE", f"NYU label {name!r} is listed twice")
        by_id[nid], by_label[name] = name, nid
    if K.UNKNOWN_SENTINEL not in by_label and not issues.items:
        issues.add(f, "rows", "E_IREF_SOURCE_VALUE", "the vocabulary has no 'unknown' sentinel")
    issues.raise_if_any()
    return by_id


def _region(data, issues):
    f = K.FILES["regions"]["name"]
    rows = read_csv(f, data, REGION_HEADER, {len(REGION_HEADER)}, issues)
    issues.raise_if_any()
    if len(rows) != 1 or rows[0][0] != K.REGION_ID:
        issues.add(f, "rows", "E_IREF_UNSUPPORTED_SOURCE", f"this increment supports exactly one region, ID "
                                                           f"{K.REGION_ID}; found {[r[0] for r in rows]}")
        issues.raise_if_any()
    row = rows[0]
    name = label(row[1], f, "row 2, region_label", issues)
    for i in range(2, len(REGION_HEADER)):
        number(row[i], f, f"row 2, {REGION_HEADER[i]}", issues)
    issues.raise_if_any()
    return {"id": row[0], "label": name}


def _object_rows(data, issues):
    f = K.FILES["objects"]["name"]
    rows = read_csv(f, data, OBJECT_HEADER, OBJECT_WIDTHS, issues)
    issues.raise_if_any()
    parsed, short = {}, {}
    for n, row in enumerate(rows, start=2):
        def at(i, n=n):
            return f"row {n}, {OBJECT_HEADER[i]}"
        for i, cell in enumerate(row):
            if cell == "":
                issues.add(f, at(i), "E_IREF_SOURCE_VALUE", "empty field")
        if any(cell == "" for cell in row):
            continue
        sid = decimal_id(row[0], f, at(0), issues)
        p = {"region": decimal_id(row[1], f, at(1), issues), "raw": label(row[2], f, at(2), issues),
             "nyu_id": decimal_id(row[3], f, at(3), issues), "nyu40_id": decimal_id(row[4], f, at(4), issues),
             "nyu_label": label(row[5], f, at(5), issues), "nyu40_label": label(row[6], f, at(6), issues),
             "centre": [number(row[i], f, at(i), issues) for i in (7, 8, 9)],
             "lengths": [number(row[i], f, at(i), issues) for i in (10, 11, 12)],
             "heading": number(row[13], f, at(13), issues),
             "front": None if row[14] == "_" else number(row[14], f, at(14), issues), "slots": []}
        for i in (10, 11, 12):
            v = p["lengths"][i - 10]
            if v is not None and v <= 0:
                issues.add(f, at(i), "E_IREF_SOURCE_VALUE", f"a box side length must be positive, not {v!r}")
        for g in range((len(row) - LEADING) // 6):
            cells, base = row[LEADING + 6 * g:LEADING + 6 * (g + 1)], LEADING + 6 * g
            if cells[3] == "_":
                if any(c != "_" for c in cells):
                    issues.add(f, at(base + 3), "E_IREF_SOURCE_VALUE",
                               "colour slot without a label has other values; a placeholder group must be all '_'")
                p["slots"].append(None)
                continue
            p["slots"].append(label(cells[3], f, at(base + 3), issues))
            for j in (0, 1, 2):
                c = decimal_id(cells[j], f, at(base + j), issues)
                if c is not None and int(c) > 255:
                    issues.add(f, at(base + j), "E_IREF_SOURCE_VALUE", f"RGB component {c} is above 255")
            share = number(cells[4], f, at(base + 4), issues)
            if share is not None and not 0 <= share <= 1:
                issues.add(f, at(base + 4), "E_IREF_SOURCE_VALUE", f"colour share {share!r} is outside [0, 1]")
            dist = number(cells[5], f, at(base + 5), issues)
            if dist is not None and dist < 0:
                issues.add(f, at(base + 5), "E_IREF_SOURCE_VALUE", f"colour distance {dist!r} is negative")
        p["slots"] += [None] * (3 - len(p["slots"]))  # omitted trailing groups are absent slots
        if sid is None:
            continue
        if sid in parsed:
            issues.add(f, at(0), "E_IREF_SOURCE_VALUE", f"object ID {sid} is repeated")
            continue
        parsed[sid] = p
        if len(row) < len(OBJECT_HEADER):
            short[sid] = len(row)
    issues.raise_if_any()
    for p in parsed.values():
        p["volume"] = p["lengths"][0] * p["lengths"][1] * p["lengths"][2]
    return parsed, short


def convert_scene(objects: bytes, regions: bytes, vocabulary: bytes) -> SceneStage:
    """The scene v2 record, its category map and the two inventory views, from scene-side inputs only."""
    issues = Issues()
    vocab = _vocabulary(vocabulary, issues)
    region = _region(regions, issues)
    parsed, short = _object_rows(objects, issues)
    f = K.FILES["objects"]["name"]
    for sid, p in parsed.items():
        if p["region"] != region["id"]:
            issues.add(f, f"object {sid}, region_id", "E_IREF_REFERENCE",
                       f"region {p['region']} is not in the region file (which holds {region['id']})")
    issues.raise_if_any()
    for sid, p in parsed.items():
        if p["nyu_id"] not in vocab:
            issues.add(f, f"object {sid}, nyu_id", "E_IREF_SOURCE_MISMATCH",
                       f"NYU ID {p['nyu_id']} ({p['nyu_label']!r}) is not in the pinned vocabulary")
        elif vocab[p["nyu_id"]] != p["nyu_label"]:
            issues.add(f, f"object {sid}, nyu_label", "E_IREF_SOURCE_MISMATCH",
                       f"NYU ID {p['nyu_id']} is {vocab[p['nyu_id']]!r} in the vocabulary, not {p['nyu_label']!r}")
    issues.raise_if_any()
    labels_of = {}
    for sid in sorted(parsed, key=int):
        labels_of.setdefault(parsed[sid]["raw"], {}).setdefault(parsed[sid]["nyu_label"], []).append(sid)
    for raw, found in labels_of.items():
        if len(found) > 1:
            issues.add(f, f"raw_label {raw!r}", "E_IREF_SOURCE_MISMATCH",
                       "one raw label maps to several NYU labels: " + "; ".join(f"{k!r} for objects {', '.join(v)}"
                                                                                for k, v in sorted(found.items())))
    issues.raise_if_any()

    ids = sorted(parsed, key=int)
    obj = {sid: object_id(sid) for sid in ids}
    hashes = {"objects": sha256(objects), "regions": sha256(regions), "vocabulary": sha256(vocabulary)}
    records = []
    for sid in ids:
        p = parsed[sid]
        if p["nyu_label"] == K.UNKNOWN_SENTINEL:
            category = _unknown("unmapped_label")  # never a fake known replacement class
        else:
            category = {"state": "known", "value": {"standard": p["nyu_label"], "model": p["nyu_label"]},
                        "evidence": _evidence()}
        h = p["heading"]
        colours = []
        for c in p["slots"]:
            if c is not None and c not in colours:
                colours.append(c)
        records.append({
            "object_id": obj[sid],
            "source_ref": {"source_id": K.SOURCE_ID, "source_object_id": sid, "source_label": p["raw"]},
            "category": category,
            "geometry": {"center_m": _known(list(p["centre"])), "size_m": _known(list(p["lengths"])),
                         "rotation_xyzw": {"state": "known", "value": [0.0, 0.0, math.sin(h / 2), math.cos(h / 2)],
                                           "support": "yaw_only", "evidence": _evidence()}},
            "semantic_front": _unknown() if p["front"] is None else _known([math.cos(p["front"]), math.sin(p["front"]),
                                                                           0.0]),
            "attributes": {"colours": _known(colours) if colours else _unknown()},
            "observation": {"source": K.SOURCE_ID, "capture_time": _unknown(), "detector_confidence": _unknown(),
                            "geometry_uncertainty": _unknown()}})
    paths = {r: K.FILES[r]["path"] for r in ("objects", "regions", "vocabulary")}
    scene = {
        "schema_version": 2, "record_type": "scene", "scene_id": K.SCENE_ID, "scene_revision": 0,
        "evidence_profile": "annotated", "category_map": K.MAP_ID,
        "coordinate_frame": {
            "frame_id": K.FRAME_ID, "units": "m", "handedness": "right", "up_axis": "+z",
            "origin": "the published IRef-VLA aligned-scene origin of ScanNet scene0010_01 (ScanNet's axis alignment "
                      "was applied upstream; not recentred)",
            "conversion": {"kind": "dataset_identity", "source_id": K.SOURCE_ID, "source_frame_id": K.FRAME_ID,
                           "source_to_scene": copy.deepcopy(K.IDENTITY_MATRIX)}},
        "sources": [{"source_id": K.SOURCE_ID, "kind": "dataset",
                     "description": "IRef-VLA's public ScanNet sample scene0010_01: published annotations of a ScanNet "
                                    "scan. Development material, not held-out evaluation data",
                     "release": f"IRef-VLA commit {K.COMMIT}; scene-side inputs: "
                                + "; ".join(f"{paths[r]} sha256 {hashes[r]}" for r in paths)}],
        "assumptions": [], "objects": records}
    category_map = {
        "schema_version": 1, "record_type": "category_map", "map_id": K.MAP_ID,
        "description": "IRef-VLA ScanNet sample scene0010_01: raw source labels to NYU labels, used as both the "
                       "standardized and the model-visible label. The model vocabulary is the whole pinned NYU "
                       "vocabulary except its 'unknown' sentinel",
        "release": f"IRef-VLA commit {K.COMMIT}; {paths['objects']} sha256 {hashes['objects']}; "
                   f"{paths['vocabulary']} sha256 {hashes['vocabulary']}",
        "model_vocabulary": sorted(v for v in vocab.values() if v != K.UNKNOWN_SENTINEL),
        "entries": [{"source_label": raw, "standard": nyu, "model": nyu} for raw, nyu in
                    sorted({(p["raw"], p["nyu_label"]) for p in parsed.values()
                            if p["nyu_label"] != K.UNKNOWN_SENTINEL})]}
    unknown = [sid for sid in ids if parsed[sid]["nyu_label"] == K.UNKNOWN_SENTINEL]
    views = {
        "format_version": 1, "record_type": "iref_inventory_views", "parent_scene_id": K.SCENE_ID,
        "parent_scene_revision": 0,
        "views": [
            {"view_id": "full_inventory",
             "policy": "every object row of the object CSV, in source-ID order: the canonical imported scene",
             "included_object_ids": [obj[s] for s in ids], "excluded": []},
            {"view_id": "source_known_nyu",
             "policy": "every object row whose NYU label is a recognized vocabulary label other than the 'unknown' "
                       "sentinel; built from the object CSV and the vocabulary only, before any statement is read",
             "included_object_ids": [obj[s] for s in ids if s not in unknown],
             "excluded": [{"object_id": obj[s], "source_object_id": s, "reason": "source_unknown_category"}
                          for s in unknown]}]}
    return SceneStage(scene=scene, category_map=category_map, inventory_views=views, object_ids=obj,
                      short_rows=short, source=parsed, region=region, hashes=hashes)


# --------------------------------------------------------------------------------------------- statement stages
def _expressions(statements: bytes, stage: SceneStage):
    f, issues = K.FILES["statements"]["name"], Issues()
    data = read_json(f, statements, issues)
    issues.raise_if_any()
    if not isinstance(data, dict) or set(data) != {"scene_name", "regions"}:
        issues.add(f, "$", "E_IREF_SOURCE_SHAPE", "expected exactly the keys scene_name and regions")
    elif data["scene_name"] != K.SCENE_NAME:
        issues.add(f, "$.scene_name", "E_IREF_UNSUPPORTED_SOURCE",
                   f"this increment imports {K.SCENE_NAME!r} only, not {data['scene_name']!r}")
    elif not isinstance(data["regions"], dict) or set(data["regions"]) != {stage.region["id"]}:
        issues.add(f, "$.regions", "E_IREF_UNSUPPORTED_SOURCE",
                   f"expected exactly region {stage.region['id']}, found {sorted(data['regions'])!r}"
                   if isinstance(data["regions"], dict) else "regions is not an object")
    elif not isinstance(data["regions"][stage.region["id"]], dict) or not data["regions"][stage.region["id"]]:
        issues.add(f, f"$.regions.{stage.region['id']}", "E_IREF_SOURCE_SHAPE",
                   "expected a nonempty object of expressions")
    issues.raise_if_any()
    return data["regions"][stage.region["id"]]


def convert_commands(statements: bytes, stage: SceneStage) -> list:
    """One v1 command per distinct expression, from the expression keys and the scene identity only."""
    expressions, issues, out = _expressions(statements, stage), Issues(), {}
    f = K.FILES["statements"]["name"]
    for text in expressions:
        if not text:
            issues.add(f, "$.regions.0", "E_IREF_SOURCE_VALUE", "an expression key is empty")
            continue
        cid = command_id(text)
        if cid in out:
            issues.add(f, "$.regions.0", "E_IREF_SOURCE_VALUE", f"two expressions share the command ID {cid}")
            continue
        out[cid] = {
            "schema_version": 1, "record_type": "command_context", "command_id": cid, "text": text,
            "scene_id": K.SCENE_ID, "scene_revision": 0,
            "sources": [{"source_id": K.SOURCE_ID, "kind": "dataset",
                         "description": "an expression of IRef-VLA's public ScanNet sample scene0010_01, unchanged; "
                                        "the source gives no user pose and no action",
                         "release": f"IRef-VLA commit {K.COMMIT}; expression keys of {K.FILES['statements']['path']}"}],
            "assumptions": [],
            "user_pose": {"pose_kind": "none", "frame_id": K.FRAME_ID, "scene_revision": 0,
                          "position_m": _unknown(), "rotation_xyzw": _unknown(), "heading_xy": _unknown(),
                          "snapshot_time": _unknown()}}
    issues.raise_if_any()
    return [out[k] for k in sorted(out)]


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _shape(f, path, rec, issues):
    """The pinned annotation shape: exact keys and types. True if it holds."""
    before = len(issues.items)

    def bad(where, message, code="E_IREF_SOURCE_SHAPE"):
        issues.add(f, f"{path}{where}", code, message)

    def keys(x, want, where):
        if not isinstance(x, dict):
            bad(where, "expected an object")
            return False
        extra, missing = sorted(set(x) - want), sorted(want - set(x))
        if extra or missing:
            bad(where, "source-format change: " + "; ".join(p for p in (f"unexpected keys {extra}" if extra else "",
                                                                      f"missing keys {missing}" if missing else "") if p))
            return False
        return True

    def text(x, where, empty_ok=False):
        if not isinstance(x, str) or (not x and not empty_ok):
            bad(where, "expected a nonempty string" if not empty_ok else "expected a string")

    def ref(x, where):
        if not isinstance(x, str):
            bad(where, "expected an object ID string")
        elif not re.fullmatch(r"0|[1-9][0-9]*", x):
            bad(where, f"{x!r} is not a canonical decimal object ID", "E_IREF_SOURCE_VALUE")

    def vector(x, where, n=3):
        if not (isinstance(x, list) and len(x) == n and all(_is_number(v) for v in x)):
            bad(where, f"expected {n} finite numbers")

    def volume(x, where):
        if not (_is_number(x) and x > 0):
            bad(where, "expected a positive finite number")

    def strings(x, where):
        if not (isinstance(x, list) and all(isinstance(v, str) and v for v in x)):
            bad(where, "expected a list of nonempty strings")

    if not keys(rec, RECORD_KEYS, ""):
        return False
    ref(rec["target_index"], ".target_index")
    text(rec["target_class"], ".target_class")
    vector(rec["target_position"], ".target_position")
    strings(rec["target_colors"], ".target_colors")
    volume(rec["target_size"], ".target_size")
    text(rec["target_color_used"], ".target_color_used", empty_ok=True)
    text(rec["target_size_used"], ".target_size_used", empty_ok=True)
    if not isinstance(rec["distractor_ids"], list):
        bad(".distractor_ids", "expected a list")
    else:
        for i, d in enumerate(rec["distractor_ids"]):
            ref(d, f".distractor_ids[{i}]")
    text(rec["relation"], ".relation")
    text(rec["relation_type"], ".relation_type")
    anchors = rec["anchors"]
    if not isinstance(anchors, dict) or not anchors or not all(ANCHOR_KEY.fullmatch(k) for k in anchors):
        bad(".anchors", "expected a nonempty object keyed anchor_1, anchor_2, ...")
    else:
        for k, a in anchors.items():
            if keys(a, ANCHOR_KEYS, f".anchors.{k}"):
                ref(a["index"], f".anchors.{k}.index")
                text(a["class"], f".anchors.{k}.class")
                vector(a["position"], f".anchors.{k}.position")
                strings(a["color"], f".anchors.{k}.color")
                volume(a["size"], f".anchors.{k}.size")
                text(a["color_used"], f".anchors.{k}.color_used", empty_ok=True)
                text(a["size_used"], f".anchors.{k}.size_used", empty_ok=True)
    false = rec["false_statements"]
    if keys(false, FALSE_KEYS, ".false_statements"):
        text(false["false_target_color"], ".false_statements.false_target_color")
        text(false["false_target_class"], ".false_statements.false_target_class")
        fa = false["false_anchors"]
        numbers = {ANCHOR_KEY.fullmatch(k).group(1) for k in anchors} if isinstance(anchors, dict) and all(
            ANCHOR_KEY.fullmatch(k) for k in anchors) else set()
        if not isinstance(fa, dict) or {m.group(1) for m in map(FALSE_ANCHOR_KEY.fullmatch, fa) if m} != numbers \
                or not all(FALSE_ANCHOR_KEY.fullmatch(k) for k in fa):
            bad(".false_statements.false_anchors", "expected one entry anchorN for each anchor_N")
        else:
            for k, v in fa.items():
                if keys(v, FALSE_ANCHOR_KEYS, f".false_statements.false_anchors.{k}"):
                    for kk in sorted(FALSE_ANCHOR_KEYS):
                        text(v[kk], f".false_statements.false_anchors.{k}.{kk}")
    return len(issues.items) == before


def _agrees(f, path, stage, sid, cls, position, size, issues):
    src = stage.source[sid]
    if cls != src["nyu_label"]:
        issues.add(f, f"{path}", "E_IREF_SOURCE_MISMATCH",
                   f"class {cls!r} but object {sid} has NYU label {src['nyu_label']!r} in the CSV")
    if max(abs(a - b) for a, b in zip(position, src["centre"])) > COORDINATE_TOLERANCE_M:
        issues.add(f, f"{path}", "E_IREF_SOURCE_MISMATCH",
                   f"position {position} but object {sid}'s CSV centre is {src['centre']}")
    if abs(size - src["volume"]) > VOLUME_RELATIVE_TOLERANCE * max(1.0, abs(src["volume"])):
        issues.add(f, f"{path}", "E_IREF_SOURCE_MISMATCH",
                   f"size {size!r} but object {sid}'s CSV box volume (lx*ly*lz) is {src['volume']!r}")


def convert_annotations(statements: bytes, stage: SceneStage) -> dict:
    """Every source annotation, unchanged, with its references mapped; checked against the scene-side CSV."""
    expressions, issues = _expressions(statements, stage), Issues()
    f = K.FILES["statements"]["name"]
    obj, entries = stage.object_ids, []
    for text, records in expressions.items():
        where = f"$.regions.{stage.region['id']}[{json.dumps(text, ensure_ascii=False)}]"
        if not isinstance(records, list) or not records:
            issues.add(f, where, "E_IREF_SOURCE_SHAPE", "expected a nonempty list of annotations")
            continue
        cid = command_id(text)
        for i, rec in enumerate(records):
            path = f"{where}[{i}]"
            if not _shape(f, path, rec, issues):
                continue
            refs = [("target_index", rec["target_index"])] + [(f"anchors.{k}.index", a["index"])
                                                              for k, a in rec["anchors"].items()] + [
                (f"distractor_ids[{j}]", d) for j, d in enumerate(rec["distractor_ids"])]
            dangling = [(w, r) for w, r in refs if r not in obj]
            for w, r in dangling:
                issues.add(f, f"{path}.{w}", "E_IREF_REFERENCE", f"object {r} is not in the object CSV")
            if dangling:
                continue
            _agrees(f, f"{path}.target", stage, rec["target_index"], rec["target_class"], rec["target_position"],
                    rec["target_size"], issues)
            for k, a in rec["anchors"].items():
                _agrees(f, f"{path}.anchors.{k}", stage, a["index"], a["class"], a["position"], a["size"], issues)
            entries.append({
                "annotation_id": f"{cid}.a{i:03d}", "command_id": cid, "source_region_id": stage.region["id"],
                "source_annotation_index": i, "source_payload": copy.deepcopy(rec),
                "mapped_references": {"target": obj[rec["target_index"]],
                                      "anchors": {k: obj[a["index"]] for k, a in rec["anchors"].items()},
                                      "distractors": [obj[d] for d in rec["distractor_ids"]]}})
    issues.raise_if_any()
    entries.sort(key=lambda e: (e["command_id"], e["source_annotation_index"]))
    return {"schema_version": 1, "record_type": "iref_annotations", "scene_id": K.SCENE_ID, "source_id": K.SOURCE_ID,
            "source_commit": K.COMMIT, "statement_sha256": sha256(statements), "entries": entries}


# ------------------------------------------------------------------------------------------------- graph check
def crosscheck_graph(graph: bytes, stage: SceneStage) -> dict:
    """Compare the source graph's objects with the object CSV; report omissions. Repairs nothing."""
    f, issues = K.FILES["graph"]["name"], Issues()
    data = read_json(f, graph, issues)
    issues.raise_if_any()
    region_keys = {"region_id", "region_name", "region_bbox", "objects", "relationships"}
    if not isinstance(data, dict) or set(data) != {"scene_name", "regions"}:
        issues.add(f, "$", "E_IREF_SOURCE_SHAPE", "expected exactly the keys scene_name and regions")
    elif data["scene_name"] != K.SCENE_NAME:
        issues.add(f, "$.scene_name", "E_IREF_UNSUPPORTED_SOURCE", f"{data['scene_name']!r} is not {K.SCENE_NAME!r}")
    elif not isinstance(data["regions"], dict) or set(data["regions"]) != {stage.region["id"]}:
        issues.add(f, "$.regions", "E_IREF_UNSUPPORTED_SOURCE", f"expected exactly region {stage.region['id']}")
    elif not isinstance(data["regions"][stage.region["id"]], dict) \
            or set(data["regions"][stage.region["id"]]) != region_keys:
        issues.add(f, f"$.regions.{stage.region['id']}", "E_IREF_SOURCE_SHAPE",
                   f"expected exactly the keys {sorted(region_keys)}")
    issues.raise_if_any()
    region = data["regions"][stage.region["id"]]
    if region["region_id"] != stage.region["id"] or region["region_name"] != stage.region["label"]:
        issues.add(f, f"$.regions.{stage.region['id']}", "E_IREF_SOURCE_MISMATCH",
                   f"region {region['region_id']!r} {region['region_name']!r} is not the region file's "
                   f"{stage.region['id']!r} {stage.region['label']!r}")
    if not isinstance(region["objects"], list):
        issues.add(f, f"$.regions.{stage.region['id']}.objects", "E_IREF_SOURCE_SHAPE", "expected a list")
    issues.raise_if_any()
    seen = set()
    for i, o in enumerate(region["objects"]):
        path = f"$.regions.{stage.region['id']}.objects[{i}]"
        if not isinstance(o, dict) or set(o) != GRAPH_OBJECT_KEYS:
            issues.add(f, path, "E_IREF_SOURCE_SHAPE", f"expected exactly the keys {sorted(GRAPH_OBJECT_KEYS)}")
            continue
        sid = o["object_id"]
        if sid not in stage.source:
            issues.add(f, f"{path}.object_id", "E_IREF_REFERENCE", f"object {sid!r} is not in the object CSV")
            continue
        if sid in seen:
            issues.add(f, f"{path}.object_id", "E_IREF_SOURCE_VALUE", f"object {sid} is repeated")
            continue
        seen.add(sid)
        src = stage.source[sid]
        for key, mine in (("raw_label", src["raw"]), ("nyu_id", src["nyu_id"]), ("nyu40_id", src["nyu40_id"]),
                          ("nyu_label", src["nyu_label"]), ("nyu40_label", src["nyu40_label"])):
            if o[key] != mine:
                issues.add(f, f"{path}.{key}", "E_IREF_SOURCE_MISMATCH", f"{o[key]!r}, but the CSV has {mine!r}")
        for key, mine in (("center", src["centre"]), ("size", src["lengths"])):
            if not (isinstance(o[key], list) and len(o[key]) == 3 and all(_is_number(v) for v in o[key])) \
                    or max(abs(a - b) for a, b in zip(o[key], mine)) > COORDINATE_TOLERANCE_M:
                issues.add(f, f"{path}.{key}", "E_IREF_SOURCE_MISMATCH", f"{o[key]!r}, but the CSV has {mine!r}")
        if not _is_number(o["volume"]) or abs(o["volume"] - src["volume"]) > VOLUME_RELATIVE_TOLERANCE * max(
                1.0, abs(src["volume"])):
            issues.add(f, f"{path}.volume", "E_IREF_SOURCE_MISMATCH",
                       f"{o['volume']!r}, but lx*ly*lz from the CSV is {src['volume']!r}")
        slots = [s if s is not None else "N/A" for s in src["slots"]]
        if o["color_labels"] != slots:
            issues.add(f, f"{path}.color_labels", "E_IREF_SOURCE_MISMATCH", f"{o['color_labels']!r}, but the CSV's "
                                                                            f"colour slots are {slots!r}")
    issues.raise_if_any()
    return {"graph_objects": len(seen), "mismatches": 0,
            "compared": ["labels and IDs", "centre", "size", "volume", "colour slots"],
            "omitted_from_graph": {stage.object_ids[s]: s for s in sorted(stage.source, key=int) if s not in seen},
            "relationship_kinds": sorted(region["relationships"]) if isinstance(region["relationships"], dict) else []}
