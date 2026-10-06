"""A2.3a preparation (D81): select the pilot's parents and freeze its exact requests.

Reads only the declared A2.2d bundle, the pilot protocol, the pinned tokenizer and the model description: never model
weights, CUDA, annotations or predictions. The bundle's own verifier runs first, then the narrow checks at this
boundary. Every request stores its exact prompt bytes and token IDs; the run never rebuilds a prompt.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import jsonschema

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl,
                                             issue, runtime, sha256, strict_json)
from ...preparation.iref_vla import tokens as T
from ...preparation.iref_vla.prepare import _check_overlap, verify_bundle
from .choices import (build_prompt, ceiling_results, check_boundary, choice_mapping, choices_line, context_status,
                      select_parents)
from .protocol import PROTOCOL_PATH, load_protocol

REPO = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO / "schemas" / "iref-zero-shot-pilot.v1.json"
BUNDLE_FILES = ("manifest.json", "summary.json", "report.md", "preparation-index.jsonl", "measurements.jsonl",
                "model_records/category-map.json")


@lru_cache(maxsize=None)
def validator(name):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"}
    return jsonschema.validators.validator_for(sub)(sub)


def rows_of(label, data: bytes, code) -> list:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise EvaluationInputError([issue(label, code, f"not UTF-8: {e}")]) from None
    if text and not text.endswith("\n"):
        raise EvaluationInputError([issue(label, code, "truncated: the last line has no LF")])
    return [strict_json(f"{label} line {k}", line.encode("utf-8"), code)
            for k, line in enumerate(text.split("\n")[:-1] if text else [], 1)]


def inside(root: Path, rel, code="E_PILOT_PATH") -> Path:
    """A relative POSIX path that stays inside root."""
    if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel or ":" in rel \
            or any(part in ("", ".", "..") for part in rel.split("/")):
        raise EvaluationInputError([issue(str(rel), code, "not a plain relative path inside its folder")])
    p = root / rel
    if root.resolve() not in p.resolve().parents:
        raise EvaluationInputError([issue(str(rel), code, "the path leaves its folder")])
    return p


def objects_line_ids(document: str) -> list:
    for line in document.split("\n"):
        if line.startswith('{"objects":'):
            return [row[0] for row in json.loads(line)["objects"]]
    raise EvaluationInputError([issue("document", "E_PILOT_BUNDLE", "the document has no objects line")])


def read_bundle(bundle: Path) -> dict:
    issues = verify_bundle(bundle)
    if issues:
        raise EvaluationInputError([issue(str(bundle), "E_PILOT_BUNDLE", m) for m in issues[:20]])
    raw = {}
    for n in BUNDLE_FILES:
        try:
            raw[n] = inside(bundle, n, "E_PILOT_BUNDLE").read_bytes()
        except OSError as e:
            raise EvaluationInputError([issue(n, "E_PILOT_BUNDLE", f"cannot read: {e}")]) from None
    manifest = strict_json("bundle manifest.json", raw["manifest.json"], "E_PILOT_BUNDLE")
    summary = strict_json("bundle summary.json", raw["summary.json"], "E_PILOT_BUNDLE")
    index = rows_of("preparation-index.jsonl", raw["preparation-index.jsonl"], "E_PILOT_BUNDLE")
    meas = rows_of("measurements.jsonl", raw["measurements.jsonl"], "E_PILOT_BUNDLE")
    bad = []
    roles = (("index", index, "iref_model_input_index"), ("measurement", meas, "iref_model_input_measurement"))
    for label, rows, role in roles:
        for k, r in enumerate(rows, 1):
            if not isinstance(r, dict) or r.get("record_type") != role or type(r.get("format_version")) is not int \
                    or r["format_version"] != 1 or r.get("policy_id") != "iref_model_inputs.v1":
                bad.append(f"{label} row {k}: not a version-1 {role} of iref_model_inputs.v1")
    if not (isinstance(manifest, dict) and manifest.get("record_type") == "iref_model_input_manifest"
            and type(manifest.get("format_version")) is int and manifest["format_version"] == 1):
        bad.append("the bundle manifest is not a version-1 iref_model_input_manifest")
    if bad:
        raise EvaluationInputError([issue(str(bundle), "E_PILOT_BUNDLE", m) for m in bad[:20]])
    by_pv, by_pvf = {}, {}
    for r in index:
        key = (r["parent_command_id"], r["view_id"])
        if key in by_pv:
            bad.append(f"duplicate index entry {key}")
        by_pv[key] = r
    for m in meas:
        key = (m["parent_command_id"], m["view_id"], m["format"])
        if key in by_pvf:
            bad.append(f"duplicate measurement entry {key}")
        by_pvf[key] = m
    parents = sorted({p for p, _ in by_pv})
    views, formats = ("full_inventory", "source_known_nyu"), ("coordinates_v2", "coordinates_relations_v2")
    for p in parents:
        for v in views:
            if (p, v) not in by_pv:
                bad.append(f"{p} lacks its {v} index entry")
            for f in formats:
                if (p, v, f) not in by_pvf:
                    bad.append(f"{p} lacks its {v}/{f} measurement")
    pop = summary.get("population", {}) if isinstance(summary, dict) else {}
    if pop.get("pairs") != len(index) or pop.get("measurement_rows") != len(meas):
        bad.append("the bundle summary's population does not describe the complete index and measurements")
    if bad:
        raise EvaluationInputError([issue(str(bundle), "E_PILOT_BUNDLE", m) for m in bad[:20]])
    return {"raw": raw, "manifest": manifest, "summary": summary, "index": by_pv, "meas": by_pvf, "parents": parents}


def eligible_parents(src) -> list:
    out = []
    for p in src["parents"]:
        if all(src["index"][(p, v)]["preparation_status"] == "materialized" for v in ("full_inventory", "source_known_nyu")) \
                and all(src["meas"][(p, v, f)]["status"] == "rendered" for v in ("full_inventory", "source_known_nyu")
                        for f in ("coordinates_v2", "coordinates_relations_v2")):
            out.append(p)
    return out


def _code_hashes() -> dict:
    files = sorted((REPO / "grounding" / "inference").rglob("*.py")) + [SCHEMA_PATH, PROTOCOL_PATH]
    files += [REPO / "grounding/preparation/iref_vla/prepare.py", REPO / "grounding/preparation/iref_vla/tokens.py"]
    return {p.relative_to(REPO).as_posix(): sha256(p.read_bytes()) for p in files}


def prepare_requests(*, bundle, tokenizer_dir, model_description, out, protocol=None, tokenizer=None) -> dict:
    """The `prepare` command; returns the request manifest. `tokenizer` is injectable for fixture tests only."""
    out = output.refuse_existing(out)
    proto_path = Path(protocol) if protocol is not None else PROTOCOL_PATH
    bundle = Path(bundle)
    _check_overlap(out, (bundle, tokenizer_dir, model_description, proto_path))
    proto = load_protocol(proto_path)
    proto_bytes = proto_path.read_bytes()
    try:
        desc_bytes = Path(model_description).read_bytes()
    except OSError as e:
        raise EvaluationInputError([issue(str(model_description), "E_PREP_MODEL", f"cannot read: {e}")]) from None
    desc = strict_json("model description", desc_bytes, "E_PREP_MODEL")
    T.check_model_description(desc)
    if (desc["hf_repo"], desc["hf_revision"]) != (proto["model"]["hf_repo"], proto["model"]["hf_revision"]):
        raise EvaluationInputError([issue("model description", "E_PILOT_PROTOCOL", "the description and the protocol "
                                                                                   "name different models")])
    if not bundle.is_dir():
        raise EvaluationInputError([issue(str(bundle), "E_PILOT_BUNDLE", "the bundle is not a folder")])
    src = read_bundle(bundle)
    eligible = eligible_parents(src)
    sel = proto["selection"]
    chosen, digest = select_parents(eligible, sel["salt"], sel["count"])
    if len(eligible) != sel["expected_eligible"] or digest != sel["expected_selected_sha256"]:
        raise EvaluationInputError([issue("selection", "E_PILOT_SELECTION", f"{len(eligible)} eligible parents and selected-"
                                          f"list hash {digest}; the protocol expects {sel['expected_eligible']} and "
                                          f"{sel['expected_selected_sha256']}")])
    tok = tokenizer if tokenizer is not None else T.load_pinned_tokenizer(tokenizer_dir)
    limit, ceilings = proto["context_limit_tokens"], proto["input_only_ceilings"]
    rows, files, k = [], {}, 0
    for rank, parent in enumerate(chosen, 1):
        for view, fmt in proto["request_order"]:
            k += 1
            irow, mrow = src["index"][(parent, view)], src["meas"][(parent, view, fmt)]
            rid = f"q{k:03d}"
            doc_path = inside(bundle, mrow["paths"]["document"], "E_PILOT_BUNDLE")
            doc_bytes = doc_path.read_bytes()
            scene = strict_json("scene", inside(bundle, irow["scene_path"], "E_PILOT_BUNDLE").read_bytes(), "E_PILOT_BUNDLE")
            command = strict_json("command", inside(bundle, irow["command_path"], "E_PILOT_BUNDLE").read_bytes(), "E_PILOT_BUNDLE")
            document = doc_bytes.decode("utf-8")
            planned = irow["planned_object_ids"]
            if (sha256(doc_bytes) != mrow["tokens"]["document"]["sha256"] or scene.get("scene_id") != irow["derived_scene_id"]
                    or command.get("command_id") != irow["derived_command_id"] or command.get("scene_id") != scene.get("scene_id")
                    or mrow["derived_command_id"] != irow["derived_command_id"]
                    or [o["object_id"] for o in scene.get("objects", [])] != planned or objects_line_ids(document) != planned):
                raise EvaluationInputError([issue(rid, "E_PILOT_BUNDLE", f"{parent} {view} {fmt}: the document, scene, "
                                                                         "command and index disagree")])
            mapping = choice_mapping(planned, proto)
            prompt = build_prompt(proto, mapping, document)
            ids = tok.encode(prompt)
            offered = check_boundary(tok, prompt, mapping, proto)
            pbytes = prompt.encode("utf-8")
            tbytes = (json.dumps(ids, separators=(",", ":")) + "\n").encode("ascii")
            ppath, tpath = f"requests/{rid}/prompt.txt", f"requests/{rid}/token_ids.json"
            files[ppath], files[tpath] = pbytes, tbytes
            rows.append({"format_version": 1, "record_type": "iref_pilot_request", "protocol_id": proto["protocol_id"],
                         "request_index": k, "request_id": rid, "selection_rank": rank, "parent_command_id": parent,
                         "view_id": view, "format": fmt, "derived_scene_id": irow["derived_scene_id"],
                         "derived_command_id": irow["derived_command_id"], "object_ids": list(planned),
                         "mapping": [[c, t, i] for c, t, i in offered], "empty_candidate_set": not planned,
                         "source_document_path": mrow["paths"]["document"], "source_document_sha256": sha256(doc_bytes),
                         "prompt_path": ppath, "prompt_sha256": sha256(pbytes), "prompt_bytes": len(pbytes),
                         "token_ids_path": tpath, "token_ids_file_sha256": sha256(tbytes),
                         "token_ids_sha256": sha256(",".join(str(i) for i in ids).encode("ascii")), "input_tokens": len(ids),
                         "context_status": context_status(len(ids), limit), "pair_eligible": False,
                         "input_only_ceilings": ceiling_results(len(ids), ceilings)})
    for r in rows:
        pair = [x for x in rows if x["parent_command_id"] == r["parent_command_id"] and x["view_id"] == r["view_id"]]
        r["pair_eligible"] = all(x["context_status"] == "within_context_limit" for x in pair)
    files["protocol.json"] = proto_bytes
    files["request-index.jsonl"] = encode_jsonl(rows)
    pop = src["summary"]["population"]
    manifest = {"format_version": 1, "record_type": "iref_pilot_request_manifest", "protocol_id": proto["protocol_id"],
                "protocol_sha256": sha256(proto_bytes),
                "source_bundle": {"manifest_sha256": sha256(src["raw"]["manifest.json"]),
                                  "artifacts": {n: sha256(src["raw"][n]) for n in BUNDLE_FILES if n != "manifest.json"},
                                  "aggregates": src["manifest"]["outputs"]["aggregates"],
                                  "preparation_policy": "iref_model_inputs.v1", "verification": "verify_bundle: no issues",
                                  "population": pop},
                "selection": {"salt": sel["salt"], "count": sel["count"], "eligible": len(eligible),
                              "selected_parent_ids": chosen, "selected_sha256": digest,
                              "conditioned_on": "commands the accepted rules parser supports (A2.2b), whose "
                                                "category-complete selections fit ten objects in both views (A2.2c) and "
                                                "render in both formats (A2.2d); not a representative or held-out sample"},
                "model_description": {"sha256": sha256(desc_bytes), "hf_repo": desc["hf_repo"], "hf_revision": desc["hf_revision"]},
                "tokenizer": {"identity": tok.identity, "kind": tok.kind, "files": proto["tokenizer"]["files"],
                              "versions": getattr(tok, "versions", None), "class": getattr(tok, "class_name", None)},
                "requests": {"count": len(rows), "parents": len(chosen),
                             "within_context_limit": sum(r["context_status"] == "within_context_limit" for r in rows),
                             "context_budget_exceeded": sum(r["context_status"] == "context_budget_exceeded" for r in rows),
                             "empty_candidate_set": sum(r["empty_candidate_set"] for r in rows)},
                "context_limit_tokens": limit, "input_only_ceilings": ceilings,
                "files": {p: sha256(b) for p, b in sorted(files.items())}, "code": _code_hashes(),
                "runtime": dict(runtime(), **(getattr(tok, "versions", None) or {})),
                "notes": ["Exact prompt bytes and token IDs; the run verifies and re-tokenizes them and never rebuilds a prompt.",
                          "No annotation, statement, graph, prediction or score was read."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for rel, data in sorted(files.items()):
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            output._write_file(target, data)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        problems = verify_request_dir(staging)
        if problems:
            raise RuntimeError("the request folder failed readback: " + "; ".join(problems[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def verify_request_dir(folder) -> list:
    """Every file, hash, identity, mapping, prompt header and count of a frozen request folder."""
    f, bad = Path(folder), []
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_PILOT_REQUESTS")
        proto = load_protocol(f / "protocol.json")
        rows = rows_of("request-index.jsonl", (f / "request-index.jsonl").read_bytes(), "E_PILOT_REQUESTS")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable request folder: {e}"]
    bad += [f"manifest: {e.message}" for e in validator("request_manifest").iter_errors(manifest)]
    if bad:
        return bad
    on_disk = {p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file()} - {"manifest.json"}
    if on_disk != set(manifest["files"]):
        bad.append(f"files on disk differ from the manifest: {sorted(on_disk ^ set(manifest['files']))[:4]}")
    for rel, want in manifest["files"].items():
        try:
            if sha256(inside(f, rel).read_bytes()) != want:
                bad.append(f"{rel}: changed")
        except (OSError, EvaluationInputError):
            bad.append(f"{rel}: missing or outside the folder")
    if manifest["protocol_sha256"] != sha256((f / "protocol.json").read_bytes()):
        bad.append("protocol.json differs from the manifest's protocol hash")
    sel = manifest["selection"]
    if select_parents(sel["selected_parent_ids"], sel["salt"], len(sel["selected_parent_ids"]))[1] != sel["selected_sha256"] \
            or sel["selected_sha256"] != proto["selection"]["expected_selected_sha256"]:
        bad.append("the selected parents do not hash to the protocol's literal expectation")
    want_keys = [(p, v, fm) for p in sel["selected_parent_ids"] for v, fm in proto["request_order"]]
    got_keys = [(r.get("parent_command_id"), r.get("view_id"), r.get("format")) if isinstance(r, dict) else None for r in rows]
    if got_keys != want_keys or len(rows) != manifest["requests"]["count"]:
        bad.append("the index is not exactly four variants per selected parent, in order")
    for k, r in enumerate(rows, 1):
        errs = [e.message for e in validator("request_row").iter_errors(r)]
        if errs:
            bad.append(f"index row {k}: {errs[0]}")
            continue
        rid = f"q{k:03d}"
        if r["request_index"] != k or r["request_id"] != rid or r["prompt_path"] != f"requests/{rid}/prompt.txt" \
                or r["token_ids_path"] != f"requests/{rid}/token_ids.json":
            bad.append(f"index row {k}: identity or paths are not the canonical ones")
            continue
        try:
            pbytes, tbytes = (f / r["prompt_path"]).read_bytes(), (f / r["token_ids_path"]).read_bytes()
            ids = json.loads(tbytes)
        except (OSError, ValueError):
            bad.append(f"{rid}: prompt or token IDs unreadable")
            continue
        mapping = choice_mapping(r["object_ids"], proto)
        prompt = pbytes.decode("utf-8", errors="replace")
        w = proto["wrapper"]
        head = w["before_system"] + proto["system_message"] + w["between"] + choices_line(mapping)
        if (sha256(pbytes) != r["prompt_sha256"] or len(pbytes) != r["prompt_bytes"] or sha256(tbytes) != r["token_ids_file_sha256"]
                or not isinstance(ids, list) or any(type(i) is not int for i in ids) or len(ids) != r["input_tokens"]
                or sha256(",".join(str(i) for i in ids).encode("ascii")) != r["token_ids_sha256"]
                or r["mapping"] != [[c, t, proto["code_token_ids"][c]] for c, t in mapping]
                or not prompt.startswith(head) or not prompt.endswith(w["after_user"])
                or r["empty_candidate_set"] != (not r["object_ids"])
                or r["context_status"] != context_status(r["input_tokens"], manifest["context_limit_tokens"])
                or r["input_only_ceilings"] != ceiling_results(r["input_tokens"], manifest["input_only_ceilings"])):
            bad.append(f"{rid}: prompt, token IDs, mapping or classification inconsistent")
    return bad
