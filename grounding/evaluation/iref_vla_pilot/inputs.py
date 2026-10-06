"""Reading, verifying and joining the A2.3b inputs (D82).

Every input is read-only. The accepted verifiers run first (A2.3a's request and result verifiers, A2.2d's bundle
verifier); this module adds only the checks at the new boundary: the links between the pilot results, the frozen
requests and the preparation bundle; each parent and view joined through the preparation index; each saved decision
rechecked against its own offered scores; and, for scoring alone, the reference annotations. Nothing is ever joined
by row position alone, object ID alone or approximate text.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import jsonschema

from ...adapters.iref_vla import convert as ADAPTER
from ...adapters.iref_vla import pinned as K
from ...contract import validate as contract
from ...inference.iref_vla import choices as CH
from ...inference.iref_vla import prepare as PREP
from ...inference.iref_vla import protocol as PILOT
from ...inference.iref_vla import run as RUN
from ...resolution.validate import canonical_sha256
from ..iref_vla.protocol import ANNOTATION_SCHEMA_PATH, EvaluationInputError, issue, strict_json
from .policy import (FORMATS, INPUT, INTEGRITY, REFERENCE, VIEWS, fail, kind, plain_int, read_bytes, read_json,
                     read_jsonl, sha256)


def relabel(code, fn):
    """Run fn; an accepted module's input error is reported under this increment's code, its own code kept."""
    try:
        return fn()
    except EvaluationInputError as e:
        raise EvaluationInputError([issue(i["path"], code, f"{i['code']}: {i['message']}") for i in e.issues]) from None


# ------------------------------------------------------------------------------------------------ the requests
def load_requests(folder, *, sample) -> dict:
    f = Path(folder)
    if not f.is_dir():
        fail(INPUT, [(str(f), "the request folder does not exist")])
    manifest_bytes = read_bytes(f / "manifest.json", "request manifest")
    manifest = strict_json("request manifest", manifest_bytes, INPUT)
    rows = read_jsonl(f / "request-index.jsonl", "request index")
    proto_bytes = read_bytes(f / "protocol.json", "request protocol")
    if not isinstance(manifest, dict) or any(not isinstance(r, dict) for r in rows):
        fail(INPUT, [(str(f), "the request manifest and every index row must be JSON objects")])
    proto = relabel(INPUT, lambda: PILOT.load_protocol(f / "protocol.json"))
    problems = PREP.verify_request_dir(f)
    if problems:
        fail(INTEGRITY, [(str(f), m) for m in problems[:20]])
    if sample and sha256(proto_bytes) != PILOT.protocol_sha256():
        fail(INTEGRITY, [(str(f / "protocol.json"), "the requests were not prepared with the checked-in pilot protocol "
                                                    "(grounding/inference/iref_vla/pilot-protocol.v1.json)")])
    return {"dir": f, "manifest": manifest, "manifest_sha256": sha256(manifest_bytes), "protocol": proto,
            "protocol_sha256": sha256(proto_bytes), "rows": rows,
            "index_sha256": sha256(read_bytes(f / "request-index.jsonl", "request index")),
            "by_key": {(r["parent_command_id"], r["view_id"], r["format"]): r for r in rows},
            "selected": list(manifest["selection"]["selected_parent_ids"])}


# ------------------------------------------------------------------------------------------------- the bundle
def load_bundle(folder) -> dict:
    f = Path(folder)
    if not f.is_dir():
        fail(INPUT, [(str(f), "the preparation bundle does not exist")])
    src = relabel(INTEGRITY, lambda: PREP.read_bundle(f))
    cmap = strict_json("bundle category map", src["raw"]["model_records/category-map.json"], INPUT)
    if not isinstance(cmap, dict) or not isinstance(cmap.get("model_vocabulary"), list):
        fail(INPUT, [(str(f), "the bundle's category map is not a category-map object")])
    src.update(dir=f, category_map=cmap, manifest_sha256=sha256(src["raw"]["manifest.json"]),
               index_sha256=sha256(src["raw"]["preparation-index.jsonl"]))
    return src


def check_request_bundle(requests, bundle) -> None:
    """The requests name exactly this bundle: its manifest, artifacts and folder aggregates."""
    sb, bad = requests["manifest"]["source_bundle"], []
    if sb["manifest_sha256"] != bundle["manifest_sha256"]:
        bad.append(("source_bundle.manifest_sha256", f"the requests were prepared from bundle manifest "
                                                     f"{sb['manifest_sha256']}, not this bundle's {bundle['manifest_sha256']}"))
    for name, want in sorted(sb["artifacts"].items()):
        got = sha256(bundle["raw"][name]) if name in bundle["raw"] else None
        if got != want:
            bad.append((f"source_bundle.artifacts.{name}", f"recorded {want}, this bundle has {got}"))
    if sb["aggregates"] != bundle["manifest"].get("outputs", {}).get("aggregates"):
        bad.append(("source_bundle.aggregates", "the folder aggregates differ from this bundle's manifest"))
    if bad:
        fail(INTEGRITY, [(f"request manifest {p}", m) for p, m in bad])


# ---------------------------------------------------------------------------------------------- the joining
def _scene_sources_ok(scene) -> bool:
    srcs = scene.get("sources")
    return isinstance(srcs, list) and bool(srcs) and all(
        isinstance(s, dict) and s.get("source_id") == K.SOURCE_ID and isinstance(s.get("release"), str)
        and s["release"].startswith(f"IRef-VLA commit {K.COMMIT};") for s in srcs)


def _command_sources_ok(command) -> bool:
    want = f"IRef-VLA commit {K.COMMIT}; expression keys of {K.FILES['statements']['path']}"
    srcs = command.get("sources")
    return isinstance(srcs, list) and bool(srcs) and all(
        isinstance(s, dict) and s.get("source_id") == K.SOURCE_ID and s.get("release") == want for s in srcs)


def join(requests, bundle, *, sample) -> list:
    """One verified entry per selected parent and inventory view, in canonical order (selection rank, then view)."""
    cmap, out = bundle["category_map"], []
    if sample and cmap.get("map_id") != K.MAP_ID:
        fail(INTEGRITY, [("bundle category map", f"map {cmap.get('map_id')!r} is not the pinned sample's {K.MAP_ID!r}")])
    for rank, parent in enumerate(requests["selected"], 1):
        texts = {}
        for view in VIEWS:
            where = f"parent {parent} {view}"
            irow = bundle["index"].get((parent, view))
            if irow is None or irow.get("preparation_status") != "materialized":
                fail(INTEGRITY, [(where, "the preparation index has no materialized entry for this selected parent")])
            try:
                scene = strict_json("scene", PREP.inside(bundle["dir"], irow["scene_path"], INTEGRITY).read_bytes(), INPUT)
                command = strict_json("command", PREP.inside(bundle["dir"], irow["command_path"], INTEGRITY).read_bytes(), INPUT)
            except OSError as e:
                fail(INTEGRITY, [(where, f"a linked scene or command record cannot be read: {e}")])
            if not isinstance(scene, dict) or not isinstance(command, dict) or not isinstance(scene.get("objects"), list):
                fail(INPUT, [(where, "the linked scene and command must be record objects")])
            bad = []
            if canonical_sha256(scene) != irow["scene_sha256"] or canonical_sha256(command) != irow["command_sha256"]:
                bad.append("the scene or command record differs from the hash in the preparation index")
            if not (scene.get("scene_id") == irow["derived_scene_id"] == irow["selection_id"]
                    and command.get("command_id") == irow["derived_command_id"] and command.get("scene_id") == scene.get("scene_id")):
                bad.append("the scene and command IDs disagree with the preparation index")
            if not (plain_int(scene.get("scene_revision")) and scene["scene_revision"] == 0
                    and plain_int(command.get("scene_revision")) and command["scene_revision"] == 0):
                bad.append("the derived scene and command must both be at plain integer revision 0")
            if scene.get("evidence_profile") != "annotated":
                bad.append(f"evidence profile {scene.get('evidence_profile')!r}: both pilot views use annotated evidence")
            ids = [o.get("object_id") if isinstance(o, dict) else None for o in scene["objects"]]
            if ids != irow["planned_object_ids"]:
                bad.append("the scene's ordered object IDs are not the planned object IDs")
            if scene.get("category_map") != cmap.get("map_id"):
                bad.append("the scene names another category map")
            if bad:
                fail(INTEGRITY, [(where, m) for m in bad])
            found = contract.validate([("scene", scene), ("category map", cmap), ("command", command)])
            if found:
                fail(INTEGRITY, [(f"{where} {i.file} {i.path}", f"{i.code}: {i.message}") for i in found[:10]])
            if sample and not (_scene_sources_ok(scene) and _command_sources_ok(command)):
                fail(INTEGRITY, [(where, f"the scene or command does not name source {K.SOURCE_ID!r} at IRef-VLA commit "
                                         f"{K.COMMIT} with the pinned statement file")])
            reqs = {}
            for fmt in FORMATS:
                r = requests["by_key"].get((parent, view, fmt))
                m = bundle["meas"].get((parent, view, fmt))
                if r is None or m is None or m.get("status") != "rendered":
                    fail(INTEGRITY, [(f"{where} {fmt}", "the request or its rendered measurement is missing")])
                problems = []
                if (r["derived_scene_id"], r["derived_command_id"]) != (irow["derived_scene_id"], irow["derived_command_id"]):
                    problems.append("its derived scene or command ID is not the preparation index's")
                if r["object_ids"] != ids or [x[1] for x in r["mapping"][:-1]] != ids:
                    problems.append("its object IDs or alias mapping do not follow the scene's object order")
                if r["source_document_path"] != m["paths"]["document"] \
                        or r["source_document_sha256"] != m["tokens"]["document"]["sha256"]:
                    problems.append("its source document is not the bundle's rendered document")
                if problems:
                    fail(INTEGRITY, [(f"{where} {fmt} {r['request_id']}", p) for p in problems])
                reqs[fmt] = r
            a, b = (reqs[f] for f in FORMATS)
            if a["mapping"] != b["mapping"] or a["object_ids"] != b["object_ids"]:
                fail(INTEGRITY, [(where, "the two formats of one parent and view do not share one mapping")])
            texts[view] = command.get("text")
            unknown = [o["object_id"] for o in scene["objects"] if o["category"]["state"] == "unknown"]
            out.append({"rank": rank, "parent": parent, "view": view, "scene": scene, "command": command,
                        "derived_scene_id": scene["scene_id"], "derived_command_id": command["command_id"],
                        "index_row": irow, "requests": reqs, "object_ids": ids, "unknown_ids": unknown,
                        "text": command.get("text")})
        if len(set(texts.values())) != 1 or not isinstance(texts[VIEWS[0]], str):
            fail(INTEGRITY, [(f"parent {parent}", "the two views' derived commands do not carry one identical text")])
        if sample and ADAPTER.command_id(texts[VIEWS[0]]) != parent:
            fail(INTEGRITY, [(f"parent {parent}", "the adapter's text-to-ID rule does not give this parent ID from the "
                                                  "command's exact text: the text was not preserved")])
    return out


# ------------------------------------------------------------------------------------------- the pilot results
def load_pilot(folder) -> dict:
    f = Path(folder)
    if not f.is_dir():
        fail(INPUT, [(str(f), "the pilot result folder does not exist")])
    manifest_bytes = read_bytes(f / "manifest.json", "pilot manifest")
    manifest = strict_json("pilot manifest", manifest_bytes, INPUT)
    results = read_jsonl(f / "results.jsonl", "pilot results")
    summary = read_json(f / "summary.json", "pilot summary")
    if not isinstance(manifest, dict) or not isinstance(summary, dict) or any(not isinstance(r, dict) for r in results):
        fail(INPUT, [(str(f), "the pilot manifest, summary and every result row must be JSON objects")])
    for k, r in enumerate(results, 1):
        bad = [e.message for e in PREP.validator("result_row").iter_errors(r)]
        if bad or any(not plain_int(r.get(x)) for x in ("format_version", "request_index", "input_tokens")):
            fail(INPUT, [(f"pilot results line {k}", bad[0] if bad else "a version, index or count is not a plain integer")])
    problems = RUN.verify_results(f)
    if problems:
        fail(INTEGRITY, [(str(f), m) for m in problems[:20]])
    return {"dir": f, "manifest": manifest, "manifest_sha256": sha256(manifest_bytes), "results": results,
            "summary": summary, "files": {n: sha256(read_bytes(f / n, n)) for n in ("results.jsonl", "summary.json", "report.md")}}


def check_pilot_links(pilot, requests) -> None:
    m, r, bad = pilot["manifest"], requests["manifest"], []
    if m["requests_manifest_sha256"] != requests["manifest_sha256"]:
        bad.append(("requests_manifest_sha256", f"the pilot ran requests with manifest {m['requests_manifest_sha256']}, "
                                                f"not these ({requests['manifest_sha256']})"))
    if m["protocol_sha256"] != r["protocol_sha256"] or m["protocol_id"] != requests["protocol"]["protocol_id"]:
        bad.append(("protocol_sha256", "the pilot's protocol is not the requests' protocol"))
    if m["selection"] != r["selection"]:
        bad.append(("selection", "the pilot's selected parents differ from the requests'"))
    if m["source_bundle"] != r["source_bundle"]:
        bad.append(("source_bundle", "the pilot's source-bundle identities differ from the requests'"))
    if bad:
        fail(INTEGRITY, [(f"pilot manifest {p}", msg) for p, msg in bad])


RESULT_KEYS = ("protocol_id", "request_index", "request_id", "parent_command_id", "view_id", "format", "derived_scene_id",
               "derived_command_id", "source_document_sha256", "prompt_sha256", "token_ids_sha256", "input_tokens",
               "mapping")


def check_results(results, requests) -> dict:
    """Each saved result is exactly its request: same key, identities, hashes, token count and complete mapping."""
    rows = requests["rows"]
    ids = [r.get("request_id") for r in results]
    if len(results) != len(rows) or len(set(ids)) != len(ids):
        fail(INTEGRITY, [("pilot results", f"{len(results)} rows with {len(set(ids))} distinct request IDs for "
                                           f"{len(rows)} requests: rows are repeated, missing or extra")])
    bad = []
    for res, req in zip(results, rows):
        diff = [k for k in RESULT_KEYS if res.get(k) != req.get(k)]
        if diff:
            bad.append((f"pilot results {res.get('request_id')}", f"differs from request {req['request_id']} in {diff}"))
    if bad:
        fail(INTEGRITY, bad[:20])
    return {(r["parent_command_id"], r["view_id"], r["format"]): r for r in results}


def score_problems(row, protocol, tol) -> list:
    """Recheck one saved decision against its own offered scores and A2.3a's rule; [] when consistent."""
    st, ask, out = row["technical_status"], protocol["ask_code"], []
    mapping = row["mapping"]
    if st == "context_budget_exceeded":
        if any(row[k] is not None for k in ("model_choice", "choice_code", "choice_object_id", "selection_reason",
                                             "scores", "top_restricted_share", "logit_margin")):
            out.append("a context exclusion carries a choice or scores: it is never an ASK")
        return out
    if st == "empty_scene_bypass":
        if mapping != [[ask, protocol["ask_target"], protocol["code_token_ids"][ask]]] or row["choice_code"] != ask \
                or row["selection_reason"] != "empty_candidate_set" or row["scores"] is not None:
            out.append("an empty-scene bypass must be K-only, without scores")
        return out
    scores = row["scores"]
    if not isinstance(scores, list) or len(scores) != len(mapping) or any(
            [s.get("code"), s.get("target"), s.get("token_id")] != list(m) for s, m in zip(scores, mapping)):
        return ["the offered scores are not the request's mapping, in order"]
    vals = [(s["logit"], s["log_prob"], s["restricted_share"]) for s in scores]
    if any(type(x) not in (int, float) or isinstance(x, bool) or not math.isfinite(x) for v in vals for x in v):
        return ["an offered score is not a finite number"]
    logits = [v[0] for v in vals]
    shares = CH.restricted_shares(logits)
    if any(abs(v[2] - s) > tol for v, s in zip(vals, shares)):
        out.append("the stored restricted shares are not the softmax of the offered logits")
    lse = [v[0] - v[1] for v in vals]
    if any(abs(x - lse[0]) > tol for x in lse) or any(v[1] > tol for v in vals):
        out.append("the stored log-probabilities do not share one normalizer with the offered logits")
    best = max(logits)
    tied = [s["code"] for s, x in zip(scores, logits) if x == best]
    code = ask if len(tied) > 1 else tied[0]
    want = {"choice_code": code, "selection_reason": "exact_score_tie" if len(tied) > 1 else "max_offered_logit",
            "tied_codes": tied if len(tied) > 1 else [],
            "model_choice": "model_choice_ask" if code == ask else "model_choice_object",
            "choice_object_id": None if code == ask else dict((m[0], m[1]) for m in mapping)[code]}
    diff = [k for k, v in want.items() if row[k] != v]
    if diff:
        out.append(f"the stored decision contradicts its offered scores under A2.3a's rule (largest offered logit; an "
                   f"exact maximal tie gives K): {diff}")
    ranked = sorted(logits, reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else None
    if (margin is None) != (row["logit_margin"] is None) or (margin is not None and abs(row["logit_margin"] - margin) > tol):
        out.append("the stored logit margin is not the gap between the two largest offered logits")
    if row["top_restricted_share"] is None or abs(row["top_restricted_share"] - max(shares)) > tol:
        out.append("the stored top restricted share is not the largest share")
    return out


def check_scores(results, protocol, tol) -> None:
    bad = [(f"pilot results {r['request_id']}", m) for r in results for m in score_problems(r, protocol, tol)]
    if bad:
        fail(INTEGRITY, bad[:20])


# ------------------------------------------------------------------------------------------------ annotations
_ANN = {}


def _annotation_validator():
    if "v" not in _ANN:
        _ANN["v"] = jsonschema.Draft202012Validator(json.loads(ANNOTATION_SCHEMA_PATH.read_text(encoding="utf-8")))
    return _ANN["v"]


def pinned_reference_checks() -> list:
    return [f"the annotation bundle's scene_id, source_id, source_commit and statement_sha256 equal the adapter's pinned "
            f"values ({K.SCENE_ID}, {K.SOURCE_ID}, {K.COMMIT}, {K.FILES['statements']['sha256']}; "
            "grounding/adapters/iref_vla/pinned.py) and the rules protocol's sample identity",
            "every derived scene and command names that source and commit; every command's release names the pinned "
            "statement file",
            "every selected parent ID is the adapter's text-to-ID rule applied to its exact command text"]


def load_annotations(path, entries, requests, rules_protocol, *, sample, fixture_scene_id=None):
    """Validate the whole annotation bundle, then the selected commands' references; returns (sources, info)."""
    data = read_bytes(path, "annotations", REFERENCE)
    ann = strict_json("annotations", data, REFERENCE)
    if not isinstance(ann, dict):
        fail(REFERENCE, [("annotations", f"expected one annotation bundle object, not {kind(ann)}")])
    errors = sorted(_annotation_validator().iter_errors(ann), key=lambda e: [str(p) for p in e.absolute_path])
    if errors:
        fail(REFERENCE, [("annotations $" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.absolute_path),
                          e.message[:200]) for e in errors[:10]])
    entries_all = ann["entries"]
    if not plain_int(ann["schema_version"]) or any(not plain_int(e["source_annotation_index"]) for e in entries_all):
        fail(REFERENCE, [("annotations", "schema_version and every source_annotation_index must be plain integers")])
    dup = sorted(a for a, n in Counter(e["annotation_id"] for e in entries_all).items() if n > 1)
    if dup:
        fail(REFERENCE, [("annotations", f"annotation IDs listed more than once: {dup[:5]}")])
    checks = ["the whole annotation bundle validates against schemas/iref-annotations.v1.json, with plain-integer "
              "versions and indices and unique annotation IDs"]
    if sample:
        want = {"scene_id": K.SCENE_ID, "source_id": K.SOURCE_ID, "source_commit": K.COMMIT,
                "statement_sha256": K.FILES["statements"]["sha256"]}
        wrong = [k for k, v in want.items() if ann[k] != v]
        sp = rules_protocol["sample"]
        if wrong or sp["scene_id"] != K.SCENE_ID or sp["source_commit"] != K.COMMIT or sp["category_map_id"] != K.MAP_ID:
            fail(REFERENCE, [("annotations", f"not the pinned sample's annotations: {wrong or 'rules protocol sample'} "
                                             f"differ from the adapter's pinned identity")])
        checks += pinned_reference_checks()
    else:
        if ann["scene_id"] != fixture_scene_id:
            fail(REFERENCE, [("annotations $.scene_id", f"{ann['scene_id']!r} is not the declared fixture scene "
                                                        f"{fixture_scene_id!r}")])
        checks.append("fixture mode: the annotation scene_id equals the declared fixture scene; the pinned-sample "
                      "identity checks do not apply")
    groups = {}
    for e in entries_all:
        groups.setdefault(e["command_id"], []).append(e)
    by_parent = {}
    for x in entries:
        by_parent.setdefault(x["parent"], []).append(x)
    sources, bad = {}, []
    for parent in requests["selected"]:
        es = sorted(groups.get(parent, []), key=lambda e: e["source_annotation_index"])
        where = f"annotations command {parent}"
        if not es:
            bad.append((where, "a selected command has no annotation"))
            continue
        idx = [e["source_annotation_index"] for e in es]
        if len(set(idx)) != len(idx):
            bad.append((where, f"repeated source annotation indices {idx}"))
        if any(not e["annotation_id"].startswith(parent + ".a") for e in es):
            bad.append((where, "an annotation ID is not named for its command"))
        relations = {e["source_payload"].get("relation") for e in es}
        if len(relations) != 1 or not isinstance(next(iter(relations)), str) or not next(iter(relations)):
            bad.append((where, f"expected one nonempty source relation label, found {sorted(map(str, relations))}"))
        targets = sorted({e["mapped_references"]["target"] for e in es})
        if len(targets) != 1:
            bad.append((where, f"contradictory targets {targets}: a reference error, not a set of accepted targets"))
            continue
        target = targets[0]
        for x in by_parent[parent]:
            if target not in x["object_ids"]:
                bad.append((where, f"target {target} is absent from the {x['view']} subscene: a broken selection or "
                                   "reference assumption"))
            for fmt, r in x["requests"].items():
                if target not in [m[1] for m in r["mapping"][:-1]]:
                    bad.append((where, f"target {target} is not offered under an alias in {r['request_id']} ({x['view']}, {fmt})"))
        if not any(w == where for w, _ in bad):
            sources[parent] = {"target_id": target, "relation": next(iter(relations)),
                               "annotation_ids": [e["annotation_id"] for e in es], "annotation_count": len(es)}
    if bad:
        fail(REFERENCE, bad[:20])
    checks += ["each selected command has at least one annotation, unique source indices, one mapped target and one "
               "source relation label",
               "each target is present in both views' subscenes and offered under an alias in all four requests"]
    info = {"sha256": sha256(data), "hash_kind": "file_bytes", "scene_id": ann["scene_id"], "source_id": ann["source_id"],
            "source_commit": ann["source_commit"], "statement_sha256": ann["statement_sha256"],
            "records_total": len(entries_all), "records_selected": sum(len(groups.get(p, [])) for p in requests["selected"]),
            "checks": checks}
    return sources, info
