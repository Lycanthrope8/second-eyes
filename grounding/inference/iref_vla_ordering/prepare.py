"""A2.3c preparation (D94): the frozen bundle of 4,536 crossed-cyclic requests, on the laptop.

Starts from the accepted A2.3a request folder and its preparation bundle, both verified with the accepted checks.
For every base request it rebuilds the original prompt from the bundle's document and requires byte equality, then
writes each cell's prompt with only the choices line changed (the A2.3a system message, wrapper and document are
reused byte for byte), tokenizes it with the pinned tokenizer and checks every offered code's one-token boundary. No
annotation, score, pilot result or model is read here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import tempfile
from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl,
                                             issue, runtime, sha256, strict_json)
from ...preparation.iref_vla import tokens as T
from ..iref_vla.choices import build_prompt, check_boundary, choice_mapping, choices_line, context_status
from ..iref_vla.prepare import inside, objects_line_ids, read_bundle, rows_of, verify_request_dir
from ..iref_vla.protocol import load_protocol
from . import design as D

FILES = ("manifest.json", "policy.json", "protocol.json", "request-index.jsonl", "schedule.json")


def _check_overlap(out: Path, inputs) -> None:
    o = out.resolve()
    for p in inputs:
        q = Path(p).resolve()
        if o == q or q in o.parents or o in q.parents:
            raise EvaluationInputError([issue(str(out), "E_ORDER_PATH", f"the output overlaps the input {p}")])


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.py")) + [here / "ordering-policy.v1.json", D.SCHEMA_PATH]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def _fail(code, problems):
    raise EvaluationInputError([issue(where, code, msg) for where, msg in problems])


def prepare_ordering(*, base_requests, bundle, tokenizer_dir, model_description, out, policy=None, tokenizer=None) -> dict:
    """The `prepare` command; returns the manifest. `tokenizer` and `policy` are injectable for fixture tests only."""
    out = output.refuse_existing(out)
    base, bundle = Path(base_requests), Path(bundle)
    pol_path = Path(policy) if policy is not None else D.POLICY_PATH
    _check_overlap(out, (base, bundle, model_description, pol_path) + ((tokenizer_dir,) if tokenizer_dir else ()))
    pol, pol_bytes = D.load_policy(pol_path), pol_path.read_bytes()
    problems = verify_request_dir(base)
    if problems:
        _fail("E_ORDER_BASE", [(str(base), m) for m in problems[:20]])
    bman_bytes = (base / "manifest.json").read_bytes()
    bman = json.loads(bman_bytes)
    proto_bytes = (base / "protocol.json").read_bytes()
    proto = load_protocol(base / "protocol.json")
    brows = rows_of("request-index.jsonl", (base / "request-index.jsonl").read_bytes(), "E_ORDER_BASE")
    if bman["selection"]["selected_sha256"] != pol["pins"]["selected_parents_sha256"]:
        _fail("E_ORDER_PIN", [("base selection", f"selected-parent hash {bman['selection']['selected_sha256']} is not the "
                                                 f"pinned {pol['pins']['selected_parents_sha256']}")])
    if not bundle.is_dir():
        _fail("E_ORDER_BUNDLE", [(str(bundle), "the preparation bundle is not a folder")])
    src = read_bundle(bundle)
    if sha256(src["raw"]["manifest.json"]) != bman["source_bundle"]["manifest_sha256"]:
        _fail("E_ORDER_BUNDLE", [(str(bundle), "this bundle is not the one the base requests were prepared from")])
    try:
        desc_bytes = Path(model_description).read_bytes()
    except OSError as e:
        _fail("E_ORDER_MODEL", [(str(model_description), f"cannot read: {e}")])
    if sha256(desc_bytes) != bman["model_description"]["sha256"]:
        _fail("E_ORDER_MODEL", [(str(model_description), "not the model description the base requests were prepared with")])
    tok = tokenizer if tokenizer is not None else T.load_pinned_tokenizer(tokenizer_dir)
    if (getattr(tok, "identity", None), getattr(tok, "kind", None)) != (bman["tokenizer"]["identity"], bman["tokenizer"]["kind"]):
        _fail("E_ORDER_TOKENIZER", [("tokenizer", "not the tokenizer the base requests were prepared with")])
    limit = proto["context_limit_tokens"]
    if limit != pol["context_limit_tokens"]:
        _fail("E_ORDER_POLICY", [("context", "the policy and the base protocol give different context limits")])
    w = proto["wrapper"]
    rows, lines_by_base, k = [], {}, 0
    for b in brows:
        bid, objs = b["request_id"], list(b["object_ids"])
        where = f"base {bid}"
        if not objs:
            _fail("E_ORDER_BASE", [(where, "a zero-object base request is not part of the accepted pilot population")])
        if len(set(objs)) != len(objs):
            _fail("E_ORDER_BASE", [(where, "duplicate object IDs")])
        irow = src["index"].get((b["parent_command_id"], b["view_id"]))
        doc_bytes = inside(bundle, b["source_document_path"], "E_ORDER_BUNDLE").read_bytes()
        document = doc_bytes.decode("utf-8")
        if (irow is None or sha256(doc_bytes) != b["source_document_sha256"] or irow["derived_scene_id"] != b["derived_scene_id"]
                or irow["derived_command_id"] != b["derived_command_id"] or list(irow["planned_object_ids"]) != objs
                or objects_line_ids(document) != objs):
            _fail("E_ORDER_BUNDLE", [(where, "the bundle's document, index and the base request disagree")])
        base_prompt = (base / b["prompt_path"]).read_bytes()
        base_ids = json.loads((base / b["token_ids_path"]).read_bytes())
        if build_prompt(proto, choice_mapping(objs, proto), document).encode("utf-8") != base_prompt:
            _fail("E_ORDER_BASE", [(where, "the original prompt does not rebuild byte for byte from the bundle's document")])
        n, grid, lines = len(objs), {}, []
        for a, p in D.cells(n):
            k += 1
            mapping = D.full_mapping(objs, a, p, proto)
            grid[(a, p)] = mapping[:-1]
            prompt = build_prompt(proto, mapping, document)
            pbytes = prompt.encode("utf-8")
            ids = tok.encode(prompt)
            offered = [list(x) for x in check_boundary(tok, prompt, mapping, proto)]
            vid = D.variant_id(bid, a, p)
            if (a, p) == (0, 0) and (pbytes != base_prompt or ids != base_ids or offered != b["mapping"]):
                _fail("E_ORDER_BASE", [(vid, "the identity cell does not reproduce the original prompt, token IDs and mapping")])
            if not pbytes.startswith((w["before_system"] + proto["system_message"] + w["between"] + choices_line(mapping)).encode("utf-8")):
                _fail("E_ORDER_INTERNAL", [(vid, "the prompt does not begin with its own choices line")])
            rows.append({"format_version": 1, "record_type": "iref_order_request", "policy_id": pol["policy_id"],
                         "variant_index": k, "variant_id": vid, "base_request_index": b["request_index"], "base_request_id": bid,
                         "selection_rank": b["selection_rank"], "parent_command_id": b["parent_command_id"],
                         "view_id": b["view_id"], "format": b["format"], "derived_scene_id": b["derived_scene_id"],
                         "derived_command_id": b["derived_command_id"], "object_count": n, "assignment_shift": a,
                         "order_shift": p, "object_ids": objs, "mapping": offered,
                         "source_document_sha256": b["source_document_sha256"], "base_prompt_sha256": b["prompt_sha256"],
                         "prompt_sha256": sha256(pbytes), "prompt_bytes": len(pbytes),
                         "token_ids_sha256": sha256(",".join(str(i) for i in ids).encode("ascii")), "input_tokens": len(ids),
                         "context_status": context_status(len(ids), limit), "variants_file": f"variants/{bid}.jsonl",
                         "schedule_position": 1})
            lines.append({"variant_id": vid, "prompt": prompt, "token_ids": ids})
        bad = D.balance_problems(objs, grid)
        if bad:
            _fail("E_ORDER_INTERNAL", [(where, m) for m in bad[:5]])
        lines_by_base[bid] = lines
    order = D.schedule_order(rows, pol["schedule"]["salt"])
    pos = {v: i for i, v in enumerate(order, 1)}
    for r in rows:
        r["schedule_position"] = pos[r["variant_id"]]
    counts = population_counts(rows)
    exp = pol["expected"]
    want = {"bases": exp["bases"], "cells": exp["cells"], "identity_cells": exp["identity_cells"],
            "nonidentity_cells": exp["nonidentity_cells"], "bases_by_object_count": exp["bases_by_object_count"],
            "cells_by_view": exp["cells_by_view"]}
    got = {k2: counts[k2] for k2 in want}
    if got != want:
        _fail("E_ORDER_POPULATION", [("population", f"counts {got} differ from the policy's expected {want}")])
    sched = {"format_version": 1, "record_type": "iref_order_schedule", "salt": pol["schedule"]["salt"],
             "rule": pol["schedule"]["rule"], "identity_first": counts["identity_cells"], "order": order,
             "order_sha256": D.schedule_sha256(order)}
    files = {"policy.json": pol_bytes, "protocol.json": proto_bytes, "request-index.jsonl": encode_jsonl(rows),
             "schedule.json": encode_json(sched)}
    for bid, lines in lines_by_base.items():
        files[f"variants/{bid}.jsonl"] = encode_jsonl(lines)
    toks = [r["input_tokens"] for r in rows]
    base_tok = {b["request_id"]: b["input_tokens"] for b in brows}
    delta = [r["input_tokens"] - base_tok[r["base_request_id"]] for r in rows]
    manifest = {"format_version": 1, "record_type": "iref_order_request_manifest", "policy_id": pol["policy_id"],
                "policy_sha256": sha256(pol_bytes), "protocol_id": proto["protocol_id"], "protocol_sha256": sha256(proto_bytes),
                "base_requests": {"manifest_sha256": sha256(bman_bytes), "count": len(brows),
                                  "verification": "verify_request_dir: no issues"},
                "source_bundle": bman["source_bundle"], "selection": bman["selection"],
                "model_description": bman["model_description"],
                "tokenizer": {"identity": tok.identity, "kind": tok.kind, "versions": getattr(tok, "versions", None),
                              "class": getattr(tok, "class_name", None)},
                "counts": counts,
                "tokens": {"min": min(toks), "median": statistics.median(toks), "max": max(toks), "total": sum(toks),
                           "change_from_base": {"min": min(delta), "max": max(delta)}},
                "schedule_sha256": sched["order_sha256"], "context_limit_tokens": limit,
                "disk_bytes": sum(len(v) for v in files.values()),
                "files": {p: sha256(v) for p, v in sorted(files.items())}, "code": code_hashes(),
                "runtime": dict(runtime(), **(getattr(tok, "versions", None) or {})),
                "notes": ["Only the choices line differs between the cells of a base request; the document is byte identical.",
                          "No annotation, score, pilot result or model was read."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for rel, data in sorted(files.items()):
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            output._write_file(target, data)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        problems = verify_order_requests(staging)
        if problems:
            raise RuntimeError("the request bundle failed readback: " + "; ".join(problems[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def population_counts(rows) -> dict:
    bases = {}
    for r in rows:
        bases.setdefault(r["base_request_id"], r)
    by_n, cells_by_n = {}, {}
    for r in bases.values():
        by_n[str(r["object_count"])] = by_n.get(str(r["object_count"]), 0) + 1
    for r in rows:
        cells_by_n[str(r["object_count"])] = cells_by_n.get(str(r["object_count"]), 0) + 1
    by_view, by_vf = {}, {}
    for r in rows:
        by_view[r["view_id"]] = by_view.get(r["view_id"], 0) + 1
        key = f"{r['view_id']}/{r['format']}"
        by_vf[key] = by_vf.get(key, 0) + 1
    ident = sum(1 for r in rows if r["assignment_shift"] == 0 and r["order_shift"] == 0)
    return {"bases": len(bases), "cells": len(rows), "identity_cells": ident, "nonidentity_cells": len(rows) - ident,
            "bases_by_object_count": dict(sorted(by_n.items(), key=lambda x: int(x[0]))),
            "cells_by_object_count": dict(sorted(cells_by_n.items(), key=lambda x: int(x[0]))),
            "cells_by_view": dict(sorted(by_view.items())), "cells_by_view_format": dict(sorted(by_vf.items())),
            "context_budget_exceeded": sum(1 for r in rows if r["context_status"] != "within_context_limit")}


def read_variants(folder: Path, rows) -> dict:
    """{variant_id: (prompt, token IDs)} from the per-base variant files, checked line by line against the index."""
    out, by_file = {}, {}
    for r in rows:
        by_file.setdefault(r["variants_file"], []).append(r["variant_id"])
    for rel, vids in by_file.items():
        lines = rows_of(rel, inside(folder, rel, "E_ORDER_REQUESTS").read_bytes(), "E_ORDER_REQUESTS")
        if [x.get("variant_id") if isinstance(x, dict) else None for x in lines] != vids:
            raise EvaluationInputError([issue(rel, "E_ORDER_REQUESTS", "the variant lines are not the index's cells, in order")])
        for x in lines:
            # The same rules as the schema's variant_line, checked directly: validating thousands of token IDs per
            # line through jsonschema dominated every readback.
            ids = x.get("token_ids")
            if (set(x) != {"variant_id", "prompt", "token_ids"} or not isinstance(x["prompt"], str)
                    or not isinstance(ids, list) or not all(type(i) is int and i >= 0 for i in ids)):
                raise EvaluationInputError([issue(rel, "E_ORDER_REQUESTS", f"{x.get('variant_id')}: malformed variant line")])
            out[x["variant_id"]] = (x["prompt"], ids)
    return out


def verify_order_requests(folder) -> list:
    """Every file, hash, identity, cell, mapping, document byte, token hash, schedule position and count."""
    f, bad = Path(folder), []
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_ORDER_REQUESTS")
        pol = D.load_policy(f / "policy.json")
        proto = load_protocol(f / "protocol.json")
        rows = rows_of("request-index.jsonl", (f / "request-index.jsonl").read_bytes(), "E_ORDER_REQUESTS")
        sched = strict_json("schedule.json", (f / "schedule.json").read_bytes(), "E_ORDER_REQUESTS")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable request bundle: {e}"]
    bad += [f"manifest: {m}" for m in D.schema_errors("request_manifest", manifest)]
    bad += [f"schedule: {m}" for m in D.schema_errors("schedule", sched)]
    if bad:
        return bad
    on_disk = {p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file()} - {"manifest.json"}
    if on_disk != set(manifest["files"]):
        bad.append(f"files on disk differ from the manifest: {sorted(on_disk ^ set(manifest['files']))[:4]}")
    for rel, want in manifest["files"].items():
        try:
            if sha256(inside(f, rel, "E_ORDER_REQUESTS").read_bytes()) != want:
                bad.append(f"{rel}: changed")
        except (OSError, EvaluationInputError):
            bad.append(f"{rel}: missing or outside the folder")
    if manifest["policy_sha256"] != sha256((f / "policy.json").read_bytes()) \
            or manifest["protocol_sha256"] != sha256((f / "protocol.json").read_bytes()):
        bad.append("policy.json or protocol.json differs from the manifest's hashes")
    if bad:
        return bad
    for k, r in enumerate(rows, 1):
        errs = D.schema_errors("request_row", r) if isinstance(r, dict) else ["not a JSON object"]
        if errs:
            return bad + [f"index row {k}: {errs[0]}"]
    try:
        texts = read_variants(f, rows)
    except EvaluationInputError as e:
        return bad + [str(e)]
    want_key, k, w = [], 0, proto["wrapper"]
    bases = []
    for r in rows:
        if not bases or bases[-1][0] != r["base_request_id"]:
            bases.append((r["base_request_id"], []))
        bases[-1][1].append(r)
    for bid, group in bases:
        first = group[0]
        n, objs = first["object_count"], first["object_ids"]
        if len(objs) != n or any(x["object_ids"] != objs or x["object_count"] != n for x in group):
            bad.append(f"{bid}: its cells disagree on the objects")
            continue
        if [(x["assignment_shift"], x["order_shift"]) for x in group] != D.cells(n):
            bad.append(f"{bid}: the cells are not the complete {n} x {n} grid in canonical order (repeated, missing or extra)")
            continue
        grid, docs = {}, set()
        for x in group:
            k += 1
            vid = D.variant_id(bid, x["assignment_shift"], x["order_shift"])
            mapping = D.full_mapping(objs, x["assignment_shift"], x["order_shift"], proto)
            prompt, ids = texts.get(x["variant_id"], ("", []))
            pbytes = prompt.encode("utf-8")
            head = w["before_system"] + proto["system_message"] + w["between"] + choices_line(mapping)
            document = prompt[len(head):len(prompt) - len(w["after_user"])] if prompt.startswith(head) \
                and prompt.endswith(w["after_user"]) else None
            if (x["variant_index"] != k or x["variant_id"] != vid
                    or x["mapping"] != [[c, t, proto["code_token_ids"][c]] for c, t in mapping]
                    or x["prompt_sha256"] != sha256(pbytes) or x["prompt_bytes"] != len(pbytes)
                    or x["token_ids_sha256"] != sha256(",".join(str(i) for i in ids).encode("ascii"))
                    or x["input_tokens"] != len(ids) or document is None
                    or sha256(document.encode("utf-8")) != x["source_document_sha256"]
                    or x["context_status"] != context_status(len(ids), manifest["context_limit_tokens"])
                    or ((x["assignment_shift"], x["order_shift"]) == (0, 0) and x["prompt_sha256"] != x["base_prompt_sha256"])):
                bad.append(f"{x['variant_id']}: identity, mapping, prompt, document, token IDs or classification inconsistent")
            docs.add(document)
            grid[(x["assignment_shift"], x["order_shift"])] = mapping[:-1]
        if len(docs) != 1:
            bad.append(f"{bid}: the scene document is not byte identical across its cells")
        bad += [f"{bid}: {m}" for m in D.balance_problems(objs, grid)]
        if len(bad) > 20:
            return bad
    order = D.schedule_order(rows, sched["salt"])
    pos = {v: i for i, v in enumerate(order, 1)}
    if sched["order"] != order or sched["order_sha256"] != D.schedule_sha256(order) \
            or manifest["schedule_sha256"] != sched["order_sha256"] or sched["salt"] != pol["schedule"]["salt"] \
            or [r["schedule_position"] for r in rows] != [pos[r["variant_id"]] for r in rows]:
        bad.append("the execution schedule is not the policy's deterministic order")
    if population_counts(rows) != manifest["counts"]:
        bad.append("the manifest's counts differ from a recount")
    return bad
