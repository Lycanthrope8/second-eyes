"""A2.3d preparation (D95): the frozen request bundle for the three-system comparison, on the laptop.

Reads the accepted A2.2d bundle through A2.3a's verified reader; recomputes A2.3a's eligible population (1,004) and its
32 pilot parents from A2.3a's own salt and count (checked against the pinned list hash), and excludes them; selects 256
of the remaining 972 by the policy's salted hash. For every parent and view one letter assignment and one list order
come from two separate hash streams over the parent, the view and the object IDs only, so both formats and both models
see the same mapping. Prompts use A2.3a's unchanged builder (system message, wrapper, choices line, the bundle's document
bytes). Each model's own pinned tokenizer counts the tokens, checks every offered letter's one-token boundary and
classifies the request against the 8,192-token ceiling; nothing is truncated or dropped. No annotation, result or model
is read here.
"""
from __future__ import annotations

import hashlib
import importlib
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
from ..iref_vla.choices import build_prompt, check_boundary, choices_line, context_status, select_parents
from ..iref_vla.prepare import eligible_parents, inside, objects_line_ids, read_bundle, rows_of
from ..iref_vla.protocol import PROTOCOL_PATH, load_protocol
from . import design as D

MODEL_KEYS = ("qwen2.5-0.5b-instruct", "qwen2.5-7b-instruct")


def _fail(code, problems):
    raise EvaluationInputError([issue(w, code, m) for w, m in problems])


class FileTokenizer:
    """A tokenizer loaded from a model's own acquired folder after its pinned files are checked."""
    kind = "exact"

    def __init__(self, backend, identity, versions, class_name):
        self._backend, self.identity, self.versions, self.class_name = backend, identity, versions, class_name

    def encode(self, text: str) -> list:
        return list(self._backend(text, add_special_tokens=False)["input_ids"])


def load_file_tokenizer(directory, model_key, policy) -> FileTokenizer:
    spec, d = policy["models"][model_key], Path(directory)
    bad, data = [], {}
    for name, want in spec["tokenizer_files"].items():
        try:
            data[name] = (d / name).read_bytes()
        except OSError as e:
            bad.append((str(d / name), f"cannot read: {e}"))
            continue
        if hashlib.sha256(data[name]).hexdigest() != want:
            bad.append((str(d / name), f"SHA-256 {hashlib.sha256(data[name]).hexdigest()} is not the pinned {want}"))
    try:
        acq = json.loads((d / "acquisition.json").read_text(encoding="utf-8"))
        if acq.get("hf_revision") != spec["hf_revision"] or acq.get("hf_repo") != spec["hf_repo"]:
            bad.append((str(d), "acquisition.json names another repository or revision"))
    except (OSError, ValueError):
        bad.append((str(d), "no readable acquisition.json from grounding.models.acquire"))
    if bad:
        _fail("E_COMPARE_TOKENIZER", bad)
    transformers, tokenizers = importlib.import_module("transformers"), importlib.import_module("tokenizers")
    got = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    if got != policy["tokenizer_versions"]:
        _fail("E_COMPARE_TOKENIZER", [("environment", f"found {got}; pinned to {policy['tokenizer_versions']}")])
    stage = Path(tempfile.mkdtemp(prefix="iref-compare-tokenizer-"))
    try:
        for name, b in data.items():
            (stage / name).write_bytes(b)
        backend = transformers.AutoTokenizer.from_pretrained(str(stage), local_files_only=True, trust_remote_code=False,
                                                             use_fast=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if type(backend).__name__ != "Qwen2TokenizerFast":
        _fail("E_COMPARE_TOKENIZER", [("tokenizer", f"loaded {type(backend).__name__}, not Qwen2TokenizerFast")])
    return FileTokenizer(backend, f"{spec['hf_repo']}@{spec['hf_revision']}", got, type(backend).__name__)


def load_tokenizers(policy, dir_small, dir_large) -> dict:
    small = T.load_pinned_tokenizer(dir_small)
    if small.identity != f"{policy['models'][MODEL_KEYS[0]]['hf_repo']}@{policy['models'][MODEL_KEYS[0]]['hf_revision']}":
        _fail("E_COMPARE_TOKENIZER", [("tokenizer", "the 0.5B loader names another model")])
    return {MODEL_KEYS[0]: small, MODEL_KEYS[1]: load_file_tokenizer(dir_large, MODEL_KEYS[1], policy)}


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(here.glob("*.py")) + sorted(here.glob("*.json"))}


def _stats(v) -> dict:
    v = sorted(v)
    return {"n": len(v), "min": v[0], "median": statistics.median(v), "p95": v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
            "max": v[-1]} if v else {"n": 0}


def population(src, policy) -> tuple:
    """(eligible, excluded pilot parents, remaining, selected), each checked against the policy's expectations."""
    eligible = eligible_parents(src)
    ex = policy["population"]["excluded"]
    pilot, digest = select_parents(eligible, ex["salt"], ex["count"])
    bad = []
    if len(eligible) != policy["population"]["expected_eligible"]:
        bad.append(f"{len(eligible)} eligible parents, the policy expects {policy['population']['expected_eligible']}")
    if digest != ex["selected_sha256"]:
        bad.append(f"the recomputed pilot list hashes to {digest}, not the pinned {ex['selected_sha256']}")
    remaining = sorted(set(eligible) - set(pilot))
    if len(remaining) != policy["population"]["expected_remaining"]:
        bad.append(f"{len(remaining)} parents remain, the policy expects {policy['population']['expected_remaining']}")
    if bad:
        _fail("E_COMPARE_POPULATION", [("population", m) for m in bad])
    selected = D.select(remaining, policy["selection"]["salt"], policy["selection"]["count"])
    if len(selected) != policy["selection"]["count"]:
        _fail("E_COMPARE_POPULATION", [("selection", f"only {len(selected)} parents could be selected")])
    return eligible, pilot, remaining, selected


def prepare_compare(*, bundle, tokenizer_small, tokenizer_large, out, policy=None, tokenizers=None) -> dict:
    """The `prepare` command. `policy` and `tokenizers` are injectable for labelled fixtures only."""
    out = output.refuse_existing(out)
    bundle = Path(bundle)
    for p in (bundle,) + tuple(Path(x) for x in (tokenizer_small, tokenizer_large) if x):
        o, q = out.resolve(), p.resolve()
        if o == q or q in o.parents or o in q.parents:
            _fail("E_COMPARE_PATH", [(str(out), f"the output overlaps the input {p}")])
    pol_path = Path(policy) if policy is not None else D.POLICY_PATH
    pol, pol_bytes = D.load_policy(pol_path), pol_path.read_bytes()
    proto_bytes = PROTOCOL_PATH.read_bytes()
    proto = load_protocol()
    if proto["context_limit_tokens"] != pol["context_limit_tokens"]:
        _fail("E_COMPARE_POLICY", [("context", "the policy and A2.3a's protocol give different context limits")])
    if not bundle.is_dir():
        _fail("E_COMPARE_BUNDLE", [(str(bundle), "the preparation bundle is not a folder")])
    src = read_bundle(bundle)
    eligible, pilot, remaining, selected = population(src, pol)
    toks = tokenizers if tokenizers is not None else load_tokenizers(pol, tokenizer_small, tokenizer_large)
    limit, w = pol["context_limit_tokens"], proto["wrapper"]
    rows, prompts, token_lines, k = [], [], {m: [] for m in MODEL_KEYS}, 0
    same_ids = 0
    for rank, parent in enumerate(selected, 1):
        for view in D.VIEWS:
            irow = src["index"][(parent, view)]
            planned = list(irow["planned_object_ids"])
            mapping = D.mapping_for(pol, proto, parent, view, planned)
            codes = D.code_assignment(pol, parent, view, planned)
            order = D.list_order(pol, parent, view, planned)
            scene = strict_json("scene", inside(bundle, irow["scene_path"], "E_COMPARE_BUNDLE").read_bytes(), "E_COMPARE_BUNDLE")
            command = strict_json("command", inside(bundle, irow["command_path"], "E_COMPARE_BUNDLE").read_bytes(), "E_COMPARE_BUNDLE")
            for fmt in D.FORMATS:
                k += 1
                rid = D.request_id(k)
                mrow = src["meas"][(parent, view, fmt)]
                doc_bytes = inside(bundle, mrow["paths"]["document"], "E_COMPARE_BUNDLE").read_bytes()
                document = doc_bytes.decode("utf-8")
                if (sha256(doc_bytes) != mrow["tokens"]["document"]["sha256"] or scene.get("scene_id") != irow["derived_scene_id"]
                        or command.get("command_id") != irow["derived_command_id"] or mrow["derived_command_id"] != irow["derived_command_id"]
                        or [o["object_id"] for o in scene.get("objects", [])] != planned or objects_line_ids(document) != planned):
                    _fail("E_COMPARE_BUNDLE", [(rid, f"{parent} {view} {fmt}: the document, scene, command and index disagree")])
                prompt = build_prompt(proto, mapping, document)
                pbytes = prompt.encode("utf-8")
                if not prompt.startswith(w["before_system"] + proto["system_message"] + w["between"] + choices_line(mapping)):
                    _fail("E_COMPARE_INTERNAL", [(rid, "the prompt does not begin with its own choices line")])
                per_model, offered_by = {}, {}
                for key in MODEL_KEYS:
                    ids = toks[key].encode(prompt)
                    offered_by[key] = [list(x) for x in check_boundary(toks[key], prompt, mapping, proto)]
                    per_model[key] = {"input_tokens": len(ids), "context_status": context_status(len(ids), limit),
                                      "token_ids_sha256": sha256(",".join(str(i) for i in ids).encode("ascii"))}
                    token_lines[key].append({"request_id": rid, "token_ids": ids})
                if offered_by[MODEL_KEYS[0]] != offered_by[MODEL_KEYS[1]]:
                    _fail("E_COMPARE_TOKENIZER", [(rid, "the two tokenizers give different offered-letter token IDs")])
                same_ids += token_lines[MODEL_KEYS[0]][-1]["token_ids"] == token_lines[MODEL_KEYS[1]][-1]["token_ids"]
                rows.append({"format_version": 1, "record_type": "iref_compare_request", "policy_id": pol["policy_id"],
                             "request_index": k, "request_id": rid, "selection_rank": rank, "parent_command_id": parent,
                             "view_id": view, "format": fmt, "derived_scene_id": irow["derived_scene_id"],
                             "derived_command_id": irow["derived_command_id"], "object_ids": planned,
                             "object_count": len(planned), "codes": codes, "list_order": order,
                             "mapping": offered_by[MODEL_KEYS[0]], "source_document_path": mrow["paths"]["document"],
                             "source_document_sha256": sha256(doc_bytes), "prompt_sha256": sha256(pbytes),
                             "prompt_bytes": len(pbytes), "models": per_model})
                prompts.append({"request_id": rid, "prompt": prompt})
    counts = request_counts(rows)
    exp = pol["expected"]
    if (counts["parents"], counts["parent_views"], counts["requests"]) != (exp["parents"], exp["parent_views"], exp["requests"]):
        _fail("E_COMPARE_POPULATION", [("requests", f"{counts} differ from the policy's expected {exp}")])
    selection = {"format_version": 1, "record_type": "iref_compare_selection", "policy_id": pol["policy_id"],
                 "population_rule": pol["population"]["rule"], "eligible": len(eligible),
                 "excluded_pilot": {"salt": pol["population"]["excluded"]["salt"], "count": len(pilot),
                                    "parent_ids": pilot, "selected_sha256": D.list_sha256(pilot)},
                 "remaining": len(remaining), "salt": pol["selection"]["salt"], "rule": pol["selection"]["rule"],
                 "count": len(selected), "selected_parent_ids": selected, "selected_sha256": D.list_sha256(selected),
                 "conditioned_on": "commands the accepted rules parser supports (A2.2b), whose category-complete "
                                   "selections fit ten objects in both views (A2.2c) and render in both formats (A2.2d); "
                                   "a broader development screen in the same inspected room, not an unseen-room test"}
    files = {"policy.json": pol_bytes, "protocol.json": proto_bytes, "selection.json": encode_json(selection),
             "request-index.jsonl": encode_jsonl(rows), "prompts.jsonl": encode_jsonl(prompts)}
    for key in MODEL_KEYS:
        files[f"tokens/{key}.jsonl"] = encode_jsonl(token_lines[key])
    tok_stats = {key: {v: {f: _stats([r["models"][key]["input_tokens"] for r in rows if r["view_id"] == v and r["format"] == f])
                           for f in D.FORMATS} for v in D.VIEWS} for key in MODEL_KEYS}
    manifest = {"format_version": 1, "record_type": "iref_compare_request_manifest", "policy_id": pol["policy_id"],
                "policy_sha256": sha256(pol_bytes), "protocol_id": proto["protocol_id"], "protocol_sha256": sha256(proto_bytes),
                "source_bundle": {"manifest_sha256": sha256(src["raw"]["manifest.json"]),
                                  "preparation_index_sha256": sha256(src["raw"]["preparation-index.jsonl"])},
                "selection_sha256": selection["selected_sha256"],
                "tokenizers": {key: {"identity": toks[key].identity, "kind": toks[key].kind,
                                     "class": getattr(toks[key], "class_name", None),
                                     "versions": getattr(toks[key], "versions", None),
                                     "files": pol["models"][key]["tokenizer_files"]} for key in MODEL_KEYS},
                "counts": counts, "identical_token_ids_across_tokenizers": same_ids, "token_stats": tok_stats,
                "context_limit_tokens": limit, "files": {p: sha256(v) for p, v in sorted(files.items())},
                "code": code_hashes(), "runtime": runtime(),
                "notes": ["One letter assignment and one list order per parent and view, from two salted hash streams "
                          "over the parent, the view and the object IDs; identical for both formats and both models.",
                          "The scene documents are the A2.2d bundle's bytes; only the choices line is new.",
                          "No annotation, result or model was read."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for rel, data in sorted(files.items()):
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            output._write_file(staging / rel, data)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = verify_compare_requests(staging)
        if bad:
            raise RuntimeError("the request bundle failed readback: " + "; ".join(bad[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def request_counts(rows) -> dict:
    over = {key: sum(r["models"][key]["context_status"] != "within_context_limit" for r in rows) for key in MODEL_KEYS}
    n = {}
    for r in rows:
        n[str(r["object_count"])] = n.get(str(r["object_count"]), 0) + 1
    return {"parents": len({r["parent_command_id"] for r in rows}),
            "parent_views": len({(r["parent_command_id"], r["view_id"]) for r in rows}), "requests": len(rows),
            "requests_by_object_count": dict(sorted(n.items(), key=lambda x: int(x[0]))),
            "context_budget_exceeded": over}


def verify_compare_requests(folder) -> list:
    """Files and hashes, rows, the frozen mappings recomputed, prompts' heads and documents, token hashes, counts."""
    f, bad = Path(folder), []
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_COMPARE_REQUESTS")
        pol = D.load_policy(f / "policy.json")
        proto = load_protocol(f / "protocol.json")
        selection = strict_json("selection.json", (f / "selection.json").read_bytes(), "E_COMPARE_REQUESTS")
        rows = rows_of("request-index.jsonl", (f / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
        prompts = rows_of("prompts.jsonl", (f / "prompts.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
        toks = {k: rows_of(f"tokens/{k}.jsonl", (f / "tokens" / f"{k}.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
                for k in MODEL_KEYS}
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable request bundle: {e}"]
    on_disk = {p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file()} - {"manifest.json"}
    if on_disk != set(manifest.get("files", {})):
        return [f"files on disk differ from the manifest: {sorted(on_disk ^ set(manifest.get('files', {})))[:4]}"]
    for rel, want in manifest["files"].items():
        if sha256((f / rel).read_bytes()) != want:
            bad.append(f"{rel}: changed")
    if bad:
        return bad
    sel = selection["selected_parent_ids"]
    if D.list_sha256(sel) != selection["selected_sha256"] or manifest["selection_sha256"] != selection["selected_sha256"] \
            or len(sel) != selection["count"] or len(set(sel)) != len(sel) \
            or set(sel) & set(selection["excluded_pilot"]["parent_ids"]) \
            or D.list_sha256(selection["excluded_pilot"]["parent_ids"]) != pol["population"]["excluded"]["selected_sha256"]:
        bad.append("selection.json is inconsistent (hash, count, duplicates or overlap with the pilot)")
    expect = [(rank, p, v, fm) for rank, p in enumerate(sel, 1) for v in D.VIEWS for fm in D.FORMATS]
    if len(rows) != len(expect) or len(prompts) != len(rows) or any(len(toks[k]) != len(rows) for k in MODEL_KEYS):
        return bad + ["the index, prompts and token files do not hold one line per planned request"]
    w, limit = proto["wrapper"], manifest["context_limit_tokens"]
    docs = {}
    for k, (r, (rank, p, v, fm), pr) in enumerate(zip(rows, expect, prompts), 1):
        where = r.get("request_id") if isinstance(r, dict) else f"line {k}"
        ints = ("format_version", "request_index", "selection_rank", "object_count", "prompt_bytes")
        if not isinstance(r, dict) or any(not D.plain_int(r.get(x)) for x in ints):
            bad.append(f"{where}: malformed row or a counter that is not a plain integer")
            continue
        objs = r["object_ids"]
        mapping = D.mapping_for(pol, proto, p, v, objs)
        want_map = [[c, t, proto["code_token_ids"][c]] for c, t in mapping]
        prompt = pr.get("prompt", "")
        head = w["before_system"] + proto["system_message"] + w["between"] + choices_line(mapping)
        document = prompt[len(head):len(prompt) - len(w["after_user"])] if prompt.startswith(head) and prompt.endswith(w["after_user"]) else None
        if ((r["request_index"], r["request_id"], r["selection_rank"], r["parent_command_id"], r["view_id"], r["format"])
                != (k, D.request_id(k), rank, p, v, fm) or pr.get("request_id") != r["request_id"]
                or r["mapping"] != want_map or r["codes"] != D.code_assignment(pol, p, v, objs)
                or r["list_order"] != D.list_order(pol, p, v, objs) or r["object_count"] != len(objs)
                or sha256(prompt.encode("utf-8")) != r["prompt_sha256"] or len(prompt.encode("utf-8")) != r["prompt_bytes"]
                or document is None or sha256(document.encode("utf-8")) != r["source_document_sha256"]):
            bad.append(f"{where}: identity, mapping, prompt or document inconsistent")
            continue
        docs.setdefault((p, v), set()).add(r["mapping"].__repr__())
        for key in MODEL_KEYS:
            line, m = toks[key][k - 1], r["models"].get(key, {})
            ids = line.get("token_ids") if isinstance(line, dict) else None
            if (not isinstance(ids, list) or line.get("request_id") != r["request_id"]
                    or not all(type(i) is int and i >= 0 for i in ids)
                    or m.get("token_ids_sha256") != sha256(",".join(str(i) for i in ids).encode("ascii"))
                    or m.get("input_tokens") != len(ids) or m.get("context_status") != context_status(len(ids), limit)):
                bad.append(f"{where}: {key} token IDs, count or context status inconsistent")
        if len(bad) > 20:
            return bad
    if any(len(s) != 1 for s in docs.values()):
        bad.append("the two formats of a parent and view do not share one mapping")
    if not bad and request_counts(rows) != manifest["counts"]:
        bad.append("the manifest's counts differ from a recount")
    return bad
