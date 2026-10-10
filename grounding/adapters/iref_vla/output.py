"""Building, validating and publishing an A2.2a import (D74).

run_import never writes over an existing folder. It converts everything, validates every output (the scene, map and
commands together through the shared contract validator; the annotation bundle against its schema; the closed
reference-only records and their cross-file references), writes into a new sibling folder and only then renames it
into place. Any failure removes that folder, so no apparently complete bundle is left behind. Output JSON is UTF-8,
sorted keys, two-space indentation and one final LF, written as bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import jsonschema

from ...contract import validate as contract
from .convert import K, using
from .convert import convert_annotations, convert_commands, convert_scene, crosscheck_graph
from .sources import AdapterInputError, AdapterOutputError, Issues, issue

REPO = Path(__file__).resolve().parents[3]
ROLES = ("objects", "regions", "vocabulary", "statements", "graph")
CONVERSIONS = {
    "center_m": "[object_bbox_cx, object_bbox_cy, object_bbox_cz], unchanged",
    "size_m": "[object_bbox_xlength, object_bbox_ylength, object_bbox_zlength], full side lengths, unchanged",
    "rotation_xyzw": "[0, 0, sin(h/2), cos(h/2)] with h = object_bbox_heading in radians: a positive rotation about +z "
                     "(support yaw_only); double precision, no rounding",
    "semantic_front": "[cos(f), sin(f), 0] with f = object_front_heading in radians, an atan2 heading in the aligned "
                      "scene frame; '_' gives unknown not_in_source; never derived from the box yaw",
    "colours": "colour-scheme labels in slot order, repeats removed keeping the first; '_' slots and omitted trailing "
               "groups are absent; no label at all gives unknown not_in_source, never []",
    "category": "nyu_label as standard and model label; NYU 'unknown' gives unknown unmapped_label",
    "object_id": "obj_ + the source object ID padded to at least three digits; source order",
}
EVIDENCE_BOUNDARY = ("Geometry is published IRef-VLA annotation of a ScanNet scan, in the aligned scene frame its "
                     "preprocessing produced (VLA-3D a0c023c9662bd875f0b3aeff4ff60adc1991e449): metric by ScanNet's "
                     "geometry and that preprocessing path, not measured against the physical room. The inspected code "
                     "explains the conventions; it does not certify which producer commit made the released sample.")


def encode(record) -> bytes:
    return (json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_file(path: Path, data: bytes) -> None:
    """The one place an output file is written."""
    with open(path, "xb") as f:
        f.write(data)


def _manifest(stage, data, pins):
    return {
        "format_version": 1, "record_type": "iref_source_manifest", "adapter_version": K.ADAPTER_VERSION,
        "repository": K.REPOSITORY, "commit": K.COMMIT, "url_prefix": K.URL_PREFIX,
        "pinned_check": "verified" if pins is not None else "not_requested",
        "files": [{"role": r, "repository_path": K.FILES[r]["path"], "bytes": len(data[r]),
                   "sha256": hashlib.sha256(data[r]).hexdigest()} for r in ROLES if r in data],
        "scene_id": K.SCENE_ID,
        "region": {"source_region_id": stage.region["id"], "label": stage.region["label"],
                   "object_ids": [o["object_id"] for o in stage.scene["objects"]]},
        "objects": [{"object_id": stage.object_ids[s], "source_object_id": s, "source_region_id": p["region"],
                     "raw_label": p["raw"], "nyu_id": p["nyu_id"], "nyu_label": p["nyu_label"],
                     "nyu40_id": p["nyu40_id"], "nyu40_label": p["nyu40_label"]}
                    for s, p in sorted(stage.source.items(), key=lambda kv: int(kv[0]))],
        "conversions": dict(CONVERSIONS), "evidence_boundary": EVIDENCE_BOUNDARY}


def _report(stage, commands, bundle, graph_report):
    objs = stage.scene["objects"]
    counts = {"objects": len(objs), "known_categories": sum(o["category"]["state"] == "known" for o in objs),
              "unknown_categories": sum(o["category"]["state"] == "unknown" for o in objs),
              "map_entries": len(stage.category_map["entries"]),
              "model_vocabulary": len(stage.category_map["model_vocabulary"]),
              "fronts_known": sum(o["semantic_front"]["state"] == "known" for o in objs),
              "fronts_unknown": sum(o["semantic_front"]["state"] == "unknown" for o in objs),
              "colour_lists_known": sum(o["attributes"]["colours"]["state"] == "known" for o in objs),
              "colour_lists_unknown": sum(o["attributes"]["colours"]["state"] == "unknown" for o in objs),
              "commands": None, "annotations": None, "repeated_expression_groups": None, "size_entries": None,
              "size_occurrences": None}
    relations = None
    if bundle is not None:
        per, relations, sizes, hit = {}, {}, {}, 0
        for e in bundle["entries"]:
            sp = e["source_payload"]
            per.setdefault(e["command_id"], 0)
            per[e["command_id"]] += 1
            relations[sp["relation"]] = relations.get(sp["relation"], 0) + 1
            words = [sp["target_size_used"]] + [a["size_used"] for a in sp["anchors"].values()]
            for w in words:
                if w:
                    sizes[w] = sizes.get(w, 0) + 1
            hit += any(words)
        counts.update(commands=len(commands), annotations=len(bundle["entries"]),
                      repeated_expression_groups=sum(n > 1 for n in per.values()), size_entries=hit,
                      size_occurrences=dict(sorted(sizes.items())))
        relations = dict(sorted(relations.items()))
    views = {v["view_id"]: len(v["included_object_ids"]) for v in stage.inventory_views["views"]}
    return {"format_version": 1, "record_type": "iref_import_report", "adapter_version": K.ADAPTER_VERSION,
            "runtime": {"python": platform.python_version()},
            "statements": "supplied" if bundle is not None else "not_supplied",
            "graph": "supplied" if graph_report is not None else "not_supplied",
            "counts": counts, "short_rows": dict(stage.short_rows), "views": views, "relations": relations,
            "graph_crosscheck": graph_report,
            "notes": ["Size words are kept unchanged; the current query grammar has no size predicate, so their "
                      "comparisons are not expressible there (unsupported_size_comparison).",
                      "No action and no user pose are given by the source; none is assigned."]}


def _closed(record, keys, name):
    if set(record) != set(keys):
        raise RuntimeError(f"internal error: {name} has keys {sorted(record)}, not {sorted(keys)}")


def _check_outputs(stage, commands, bundle, manifest, report, views):
    """Validate what is about to be written. A failure here is an adapter bug, not bad input."""
    records = [("scene", stage.scene), ("category_map", stage.category_map)] + [(c["command_id"], c) for c in commands or []]
    found = contract.validate(records)
    if found:
        raise RuntimeError("internal error: the shared validator rejects the adapter's output: "
                           + "; ".join(i.line() for i in found[:5]))
    scene_ids = [o["object_id"] for o in stage.scene["objects"]]
    _closed(views, ("format_version", "record_type", "parent_scene_id", "parent_scene_revision", "views"),
            "inventory-views")
    for v in views["views"]:
        _closed(v, ("view_id", "policy", "included_object_ids", "excluded"), "an inventory view")
        listed = v["included_object_ids"] + [e["object_id"] for e in v["excluded"]]
        if sorted(listed) != sorted(scene_ids) or len(set(listed)) != len(listed):
            raise RuntimeError(f"internal error: view {v['view_id']} does not partition the scene's objects")
    if type(views["format_version"]) is not int or views["parent_scene_id"] != stage.scene["scene_id"]:
        raise RuntimeError("internal error: inventory-views header")
    _closed(manifest, ("format_version", "record_type", "adapter_version", "repository", "commit", "url_prefix",
                       "pinned_check", "files", "scene_id", "region", "objects", "conversions", "evidence_boundary"),
            "source-manifest")
    if [o["object_id"] for o in manifest["objects"]] != scene_ids or type(manifest["format_version"]) is not int:
        raise RuntimeError("internal error: the manifest's objects are not the scene's")
    _closed(report, ("format_version", "record_type", "adapter_version", "runtime", "statements", "graph", "counts",
                     "short_rows", "views", "relations", "graph_crosscheck", "notes"), "import-report")
    if report["counts"]["objects"] != len(scene_ids) or type(report["format_version"]) is not int:
        raise RuntimeError("internal error: the report's counts are not the scene's")
    if bundle is not None:
        schema = json.loads((REPO / "schemas" / "iref-annotations.v1.json").read_text(encoding="utf-8"))
        errors = list(jsonschema.Draft202012Validator(schema).iter_errors(bundle))
        if errors or type(bundle["schema_version"]) is not int \
                or any(type(e["source_annotation_index"]) is not int for e in bundle["entries"]):
            raise RuntimeError(f"internal error: the annotation bundle is invalid: "
                               f"{errors[0].message[:200] if errors else 'integer fields'}")
        ids = {c["command_id"] for c in commands}
        if bundle["scene_id"] != stage.scene["scene_id"] or any(e["command_id"] not in ids for e in bundle["entries"]) \
                or any(e["mapped_references"]["target"] not in scene_ids for e in bundle["entries"]):
            raise RuntimeError("internal error: the annotation bundle refers outside the scene or commands")


def _publish(out: Path, files: dict) -> None:
    parent = out.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=parent))
    except OSError as e:
        raise AdapterOutputError([issue(str(out), "$", "E_IREF_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    try:
        for rel in files:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_file(target, files[rel])
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise AdapterOutputError([issue(str(out), "$", "E_IREF_OUTPUT_IO",
                                        f"{type(e).__name__}: {e}; no output was published")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _run_import(objects, regions, vocabulary, out, *, statements=None, graph=None, pins=K.PINNED) -> dict:
    """Import the pinned sample into a new folder `out`; returns the import report.

    pins verifies every supplied file's size and SHA-256 against Appendix A; the command line always passes it.
    pins=None skips that check, for the hand-written test fixtures only.
    """
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise AdapterInputError([issue(str(out), "$", "E_IREF_OUTPUT_EXISTS",
                                       "the output folder already exists; choose a new one (nothing is overwritten)")])
    paths = {"objects": objects, "regions": regions, "vocabulary": vocabulary, "statements": statements,
             "graph": graph}
    data, issues = {}, Issues()
    for role, path in paths.items():
        if path is None:
            continue
        try:
            data[role] = Path(path).read_bytes()
        except OSError as e:
            issues.add(str(path), "$", "E_IREF_SOURCE_SHAPE", f"cannot read the {role} file: {e}")
    issues.raise_if_any()
    if pins is not None:
        for role, b in data.items():
            pin = pins["files"][role]
            if len(b) != pin["bytes"] or hashlib.sha256(b).hexdigest() != pin["sha256"]:
                issues.add(str(paths[role]), "$", "E_IREF_SOURCE_MISMATCH",
                           f"{len(b)} bytes, sha256 {hashlib.sha256(b).hexdigest()}; the pinned {pin['path']} at "
                           f"{pins['commit'][:12]} has {pin['bytes']} bytes, sha256 {pin['sha256']}")
        issues.raise_if_any()
    stage = convert_scene(data["objects"], data["regions"], data["vocabulary"])
    commands = bundle = graph_report = None
    if "statements" in data:
        commands = convert_commands(data["statements"], stage)
        bundle = convert_annotations(data["statements"], stage)
    if "graph" in data:
        graph_report = crosscheck_graph(data["graph"], stage)
    manifest = _manifest(stage, data, pins)
    report = _report(stage, commands, bundle, graph_report)
    _check_outputs(stage, commands, bundle, manifest, report, stage.inventory_views)
    files = {"model_inputs/scene.annotated.json": encode(stage.scene),
             "model_inputs/category-map.json": encode(stage.category_map)}
    for c in commands or []:
        files[f"model_inputs/commands/{c['command_id']}.json"] = encode(c)
    files["reference_only/inventory-views.json"] = encode(stage.inventory_views)
    if bundle is not None:
        files["reference_only/iref-annotations.json"] = encode(bundle)
    files["reference_only/source-manifest.json"] = encode(manifest)
    files["reference_only/import-report.json"] = encode(report)
    _publish(out, files)
    return report


def run_import(objects, regions, vocabulary, out, *, statements=None, graph=None, pins=K.PINNED, identity=None) -> dict:
    """Import into a new folder `out` under identity (None: the pinned sample, as A2.2a); returns the import report.
    pins verifies every supplied file's size and SHA-256; pins=None skips that check (hand-written fixtures only)."""
    with using(identity):
        return _run_import(objects, regions, vocabulary, out, statements=statements, graph=graph, pins=pins)
