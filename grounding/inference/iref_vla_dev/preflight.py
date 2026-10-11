"""A2.6c preflight: the frozen A2.6b requests verified, read-only, before any model loads (ChatGPT's A2.6c brief, item 2).

**What it checks.**

- **The frozen identities:** the request file's SHA-256 and the specification's SHA-256 (both pinned here), and every
  file hash in the preparation manifest.
- **The index:** it agrees with the request file, field by field.
- **The request set:** unique request IDs; IDs equal to the deterministic rule (parent, view, format); four-way coverage
  per parent (two views x two formats); the parents exactly the sampling's selected ones.
- **Each request against its scene's A2.2d bundle:**
  - the derived scene and command IDs equal the bundle's index;
  - the document equals the bundle's rendered document for that view and format;
  - the D104 mapping (stable letters over the derived scene's object IDs in sorted order, K last) is recomputed with the
    pinned protocol.
- **Prompts and tokens:**
  - the prompt is rebuilt from the document and the mapping with the pinned protocol (its hash and byte count);
  - the token array's hash and count;
  - the context limit;
  - both token boundaries;
  - each model's own pinned tokenizer gives the token IDs and offered-letter boundaries (A2.3d's accepted checks).
- **Answers:** no request carries a target field. `reference_only/` is never opened here.

Any mismatch refuses, with every problem written to the diagnostics. Nothing is written into the inputs.

**The context.** `context_for` builds A2.3d's runner context from a passed preflight. A2.3d's accepted `smoke` and
`run_compare` then run unchanged: float32, eager attention, batch one, one uncached final-position forward, its
canaries, its resume lock.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, issue, runtime
from ...preparation.iref_vla import tokens as T
from ...quest import replay_inputs as RI
from ...quest.replay_bundle import boundary_before_command_line, boundary_before_last_token
from ..iref_vla.choices import build_prompt, check_boundary, choice_mapping, context_status
from ..iref_vla.prepare import read_bundle
from ..iref_vla.protocol import load_protocol
from ..iref_vla_compare import design as CD
from ..iref_vla_compare.prepare import MODEL_KEYS, load_file_tokenizer

FROZEN = {"spec_sha256": "9fa980010eb46f25bb471ede64a37b366d812adb654e23e8370edae450c538b0",
          "requests_sha256": "dc725cfe7bfab1b80a468b748d12a8b21b5583f14176d43337a0835c1b6f03ea"}
RUN_ID = "a26c.devbaseline.v1"
VIEWS = ("full_inventory", "source_known_nyu")
FORMATS = ("coordinates_v2", "coordinates_relations_v2")
ANSWER_FIELDS = ("target_object_id", "target_offered", "target_indices", "annotation_ids")
MAX_LISTED = 50


class PreflightError(EvaluationInputError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jl(b: bytes) -> list:
    return [json.loads(x) for x in b.decode("utf-8").splitlines() if x.strip()]


def request_id(parent, view, fmt) -> str:   # A2.6b's rule, restated so the preflight checks it independently
    return "a26-" + _sha(f"{parent}\n{view}\n{fmt}".encode("utf-8"))[:24]


def load_tokenizers(dirs: dict, policy) -> dict:
    """Each model's own pinned tokenizer: the 0.5B through A2.2d's pin check, the 7B through A2.3d's."""
    out = {}
    for key in MODEL_KEYS:
        if key in dirs:
            out[key] = T.load_pinned_tokenizer(dirs[key]) if key == MODEL_KEYS[0] else load_file_tokenizer(dirs[key], key, policy)
    return out


def _code() -> dict:
    return {p.name: _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _verify_scenes(job):
    """Verifies every request of the given scenes against their bundles, the mapping, the prompt, the tokens and each
    model's tokenizer. A module-level function, so worker processes can run it. job: (work, requests, tokenizer_dirs,
    tokenizers); worker processes load the tokenizers from their directories."""
    work, reqs, tok_dirs, tokenizers = job
    work = Path(work)
    if tokenizers is None:
        from ..iref_vla_compare import design as CD
        tokenizers = load_tokenizers(tok_dirs, CD.load_policy())
    proto = load_protocol()
    small = tokenizers[MODEL_KEYS[0]]
    problems, bundles, prompts = [], {}, {}

    def bad(check, where, message):
        problems.append({"check": check, "where": str(where), "message": message})
    for r in reqs:
        rid, scene = r["request_id"], r["scene"]
        if scene not in bundles:
            try:
                bundles[scene] = (read_bundle(work / scene / "bundle"),
                                  _sha((work / scene / "bundle" / "manifest.json").read_bytes()))
            except Exception as e:   # noqa: BLE001
                bad("bundle", scene, f"the bundle does not read: {type(e).__name__}: {e}"[:300])
                bundles[scene] = (None, None)
        src = bundles[scene][0]
        if src is None:
            continue
        irow = src["index"].get((r["parent_command_id"], r["view"]))
        meas = src["meas"].get((r["parent_command_id"], r["view"], r["format"]))
        if not irow or (irow["derived_scene_id"], irow["derived_command_id"]) != (r["derived_scene_id"], r["derived_command_id"]):
            bad("bundle", rid, "the derived scene or command ID is not the bundle index's")
            continue
        if not meas or meas.get("status") != "rendered" or meas["tokens"]["document"]["sha256"] != r["document_sha256"]:
            bad("document", rid, "the document is not the bundle's rendered document for this view and format")
        if _sha(r["document"].encode("utf-8")) != r["document_sha256"]:
            bad("document", rid, "the document does not hash to its recorded SHA-256")
        dscene = json.loads((work / scene / "bundle" / irow["scene_path"]).read_text(encoding="utf-8"))
        mapping = choice_mapping(sorted(o["object_id"] for o in dscene["objects"]), proto)
        codes, targets = [m[0] for m in mapping], [m[1] for m in mapping]
        code_ids = [proto["code_token_ids"][c] for c in codes]
        if (codes, targets, code_ids) != (r["codes"], r["choice_object_ids"], r["code_token_ids"]) \
                or RI.mapping_sha256(codes, targets, code_ids) != r["mapping_sha256"]:
            bad("mapping", rid, "the D104 mapping recomputed from the derived scene differs")
        prompt = build_prompt(proto, mapping, r["document"])
        pb = prompt.encode("utf-8")
        if _sha(pb) != r["prompt_sha256"] or len(pb) != r["prompt_bytes"]:
            bad("prompt", rid, "the prompt rebuilt from the document and mapping does not hash to the recorded prompt")
        tids = r["token_ids"]
        if RI.token_ids_sha256(tids) != r["token_ids_sha256"] or len(tids) != r["input_tokens"]:
            bad("tokens", rid, "the token array does not hash to its recorded SHA-256 or count")
        if context_status(len(tids), proto["context_limit_tokens"], proto["continuation_tokens"]) != "within_context_limit":
            bad("context", rid, f"{len(tids)} tokens leave no room within the context limit")
        if [int(x) for x in small.encode(prompt)] != tids:
            bad("tokens", rid, "the 0.5B pinned tokenizer does not give the frozen token IDs")
        elif (boundary_before_last_token(small, prompt, tids)["keep_tokens"] != r["keep_before_last_token"]
              or boundary_before_command_line(small, prompt, tids, proto, mapping)["keep_tokens"] != r["keep_before_command_line"]):
            bad("boundaries", rid, "a token boundary differs from the recorded one")
        prompts[rid] = (prompt, mapping)
    model_tokens = {}
    for key, tok in tokenizers.items():
        rows = {}
        for r in reqs:
            if r["request_id"] not in prompts:
                continue
            prompt, mapping = prompts[r["request_id"]]
            tid = [int(x) for x in tok.encode(prompt)]
            want = [[c, t, i] for c, t, i in zip(r["codes"], r["choice_object_ids"], r["code_token_ids"])]
            if [list(x) for x in check_boundary(tok, prompt, mapping, proto)] != want:
                bad("boundaries", r["request_id"], f"{key}: an offered letter's one-token boundary differs")
            rows[r["request_id"]] = {"input_tokens": len(tid), "token_ids_sha256": RI.token_ids_sha256(tid),
                                     "context_status": context_status(len(tid), proto["context_limit_tokens"], proto["continuation_tokens"]),
                                     "same_as_prepared": tid == r["token_ids"]}
        model_tokens[key] = {"rows": rows, "identity": getattr(tok, "identity", None)}
    return {"problems": problems, "model_tokens": model_tokens,
            "bundles": {s: b[1] for s, b in bundles.items() if b[1]}}


def preflight(*, prep, work, tokenizers=None, tokenizer_dirs=None, workers=1, frozen=FROZEN, progress=print) -> dict:
    """The full read-only preflight: the receipt, and each model's per-request token hashes. tokenizers: model key ->
    tokenizer, in this process; or tokenizer_dirs with workers > 1, the scenes then verified in parallel worker processes
    that load the tokenizers themselves. Raises PreflightError with every problem (and the receipt) on any mismatch."""
    import concurrent.futures
    import multiprocessing
    t_start = time.perf_counter()
    prep, work = Path(prep), Path(work)
    problems = []

    def bad(check, where, message):
        problems.append({"check": check, "where": str(where), "message": message})

    man_bytes = (prep / "manifest.json").read_bytes()
    manifest = json.loads(man_bytes)
    req_bytes = (prep / "requests.jsonl").read_bytes()
    spec_bytes = (prep / "spec.json").read_bytes()
    if _sha(req_bytes) != frozen["requests_sha256"]:
        bad("frozen", "requests.jsonl", f"SHA-256 {_sha(req_bytes)} is not the frozen {frozen['requests_sha256']}")
    if _sha(spec_bytes) != frozen["spec_sha256"]:
        bad("frozen", "spec.json", f"SHA-256 {_sha(spec_bytes)} is not the frozen {frozen['spec_sha256']}")
    if manifest.get("hashes", {}).get("requests") != frozen["requests_sha256"] or manifest.get("hashes", {}).get("spec") != frozen["spec_sha256"]:
        bad("frozen", "manifest.json", "the preparation manifest names other request or specification hashes")
    for name, h in manifest.get("files", {}).items():
        if name.startswith("reference_only/"):   # answers: never opened here; scoring verifies them before use
            continue
        f = prep / name
        if not f.is_file() or _sha(f.read_bytes()) != h:
            bad("manifest", name, "missing or changed since the preparation")
    if problems:
        raise PreflightError([issue(p["where"], "E_A26C_PREFLIGHT", f"{p['check']}: {p['message']}") for p in problems[:MAX_LISTED]])
    reqs = _jl(req_bytes)
    index = _jl((prep / "requests-index.jsonl").read_bytes())
    sampling = json.loads((prep / "sampling.json").read_text(encoding="utf-8"))
    ids = [r["request_id"] for r in reqs]
    if len(set(ids)) != len(ids):
        bad("set", "requests.jsonl", "request IDs repeat")
    if [x["request_id"] for x in index] != ids:
        bad("index", "requests-index.jsonl", "the index does not list the request file's IDs in its order")
    else:
        for x, r in zip(index, reqs):
            diff = [k for k in x if x[k] != r.get(k)]
            if diff:
                bad("index", x["request_id"], f"index fields differ from the request: {diff[:4]}")
    per = {}
    for r in reqs:
        per.setdefault(r["parent_command_id"], set()).add((r["view"], r["format"]))
        if r["request_id"] != request_id(r["parent_command_id"], r["view"], r["format"]):
            bad("set", r["request_id"], "the ID is not the rule's ID for its parent, view and format")
        if any(k in r for k in ANSWER_FIELDS):
            bad("answers", r["request_id"], "a request carries an answer field")
    full = {(v, f) for v in VIEWS for f in FORMATS}
    for p_, s in per.items():
        if s != full:
            bad("coverage", p_, f"has {sorted(s)}, not both views x both formats")
    selected = {x["parent_command_id"] for x in sampling["selected"] if x["requests_built"]}
    if set(per) != selected:
        bad("coverage", "sampling.json", f"{len(set(per) ^ selected)} parents differ from the sampling's selected parents")
    by_scene = {}
    for r in reqs:
        by_scene.setdefault(r["scene"], []).append(r)
    scenes = sorted(by_scene)
    results = []
    if workers > 1 and tokenizers is None:
        chunks = [scenes[k::workers] for k in range(workers) if scenes[k::workers]]
        jobs = [(str(work), [r for s in ch for r in by_scene[s]], {k: str(v) for k, v in tokenizer_dirs.items()}, None) for ch in chunks]
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as ex:
            for k, res in enumerate(ex.map(_verify_scenes, jobs), 1):
                results.append(res)
                progress(f"  verified worker {k}/{len(jobs)}")
    else:
        toks = tokenizers if tokenizers is not None else load_tokenizers(tokenizer_dirs, __import__(
            "grounding.inference.iref_vla_compare.design", fromlist=["x"]).load_policy())
        results.append(_verify_scenes((str(work), reqs, None, toks)))
    model_tokens, bundles = {}, {}
    for res in results:
        problems += res["problems"]
        bundles.update(res["bundles"])
        for key, mt in res["model_tokens"].items():
            m = model_tokens.setdefault(key, {"rows": {}, "identity": mt["identity"]})
            if m["identity"] != mt["identity"]:
                bad("tokenizer", key, "the workers loaded different tokenizers")
            m["rows"].update(mt["rows"])
    token_files = {}
    for key, m in model_tokens.items():
        rows = [dict(m["rows"][rid], request_id=rid) for rid in ids if rid in m["rows"]]
        token_files[key] = rows
        m["list_sha256"] = _sha("\n".join(x["token_ids_sha256"] for x in rows).encode("utf-8"))
        m["same_as_prepared"] = sum(x["same_as_prepared"] for x in rows)
    receipt = {
        "format_version": 2, "record_type": "a26c_preflight", "run_id": RUN_ID, "frozen": dict(frozen),
        "passed": not problems, "problems": problems[:MAX_LISTED], "problem_count": len(problems),
        "counts": {"requests": len(reqs), "parents": len(per), "scenes": len(scenes),
                   "by_partition": {p_: sum(1 for r in reqs if r["partition"] == p_) for p_ in sorted({r["partition"] for r in reqs})}},
        "binding": {"requests_sha256": _sha(req_bytes), "spec_sha256": _sha(spec_bytes), "prep_manifest_sha256": _sha(man_bytes),
                    "bundles": dict(sorted(bundles.items())), "code": _code()},
        "models": {k: {"tokenizer_identity": v["identity"], "same_token_ids_as_prepared": v["same_as_prepared"],
                       "token_ids_sha256_list": v["list_sha256"], "requests": len(token_files[k])} for k, v in model_tokens.items()},
        "answers": "reference_only/ was not opened; scoring verifies those files against the manifest before use",
        "workers": workers, "runtime": runtime(), "duration_s": round(time.perf_counter() - t_start, 1)}
    if problems:
        err = PreflightError([issue(p_["where"], "E_A26C_PREFLIGHT", f"{p_['check']}: {p_['message']}") for p_ in problems[:MAX_LISTED]])
        err.receipt = receipt
        raise err
    return {"receipt": receipt, "tokens": token_files}


def write_preflight(result, out) -> Path:
    """The passed preflight's folder: receipt.json and tokens-<model>.jsonl (one row per request: its token count, hash
    and context status under that model's tokenizer)."""
    out = Path(out)
    if out.exists():
        raise PreflightError([issue(str(out), "E_A26C_OUTPUT_EXISTS", "the preflight folder exists")])
    from ...evaluation.iref_vla.protocol import encode_json, encode_jsonl
    out.mkdir(parents=True)
    for key, rows in result["tokens"].items():
        (out / f"tokens-{key}.jsonl").write_bytes(encode_jsonl(rows))
    (out / "receipt.json").write_bytes(encode_json(result["receipt"]))
    return out


def bound_context(*, preflight_dir, prep, work, model_key, model_dir, tokenizer, frozen=FROZEN, policy=None,
                  evidence_fn=None) -> tuple:
    """A smoke or run step's context, bound to a passed preflight. Re-verifies the binding (the frozen hashes, the
    preparation manifest, every bundle manifest, this code, this model's tokenizer identity), rebuilds every prompt and
    checks its hash, then tokenizes each prompt with this model's tokenizer and checks every token hash against the
    receipt. Any mismatch refuses: rerun the preflight. Returns (context, binding record)."""
    t0 = time.perf_counter()
    pd, prep, work = Path(preflight_dir), Path(prep), Path(work)
    rec = json.loads((pd / "receipt.json").read_text(encoding="utf-8"))
    rows = {x["request_id"]: x for x in _jl((pd / f"tokens-{model_key}.jsonl").read_bytes())}
    b, problems = rec.get("binding", {}), []
    req_bytes = (prep / "requests.jsonl").read_bytes()
    checks = {
        "the preflight passed": rec.get("passed") is True and rec.get("frozen") == dict(frozen),
        "the request file is the frozen one it verified": _sha(req_bytes) == b.get("requests_sha256") == frozen["requests_sha256"],
        "the specification is the frozen one it verified": _sha((prep / "spec.json").read_bytes()) == b.get("spec_sha256") == frozen["spec_sha256"],
        "the preparation manifest is unchanged": _sha((prep / "manifest.json").read_bytes()) == b.get("prep_manifest_sha256"),
        "every bundle manifest is unchanged": bool(b.get("bundles")) and all(
            _sha((work / s / "bundle" / "manifest.json").read_bytes()) == h for s, h in b["bundles"].items()),
        "this code is the code that verified": b.get("code") == _code(),
        "this model's tokenizer is the one it checked": rec.get("models", {}).get(model_key, {}).get("tokenizer_identity")
                                                        == getattr(tokenizer, "identity", None),
    }
    problems += [k for k, ok in checks.items() if not ok]
    if problems:
        raise PreflightError([issue(str(pd), "E_A26C_BINDING", f"{p_}: no; rerun the preflight") for p_ in problems])
    reqs = _jl(req_bytes)
    proto = load_protocol()
    toks, bad_tok = {}, []
    for r in reqs:
        mapping = list(zip(r["codes"], r["choice_object_ids"]))
        prompt = build_prompt(proto, mapping, r["document"])
        if _sha(prompt.encode("utf-8")) != r["prompt_sha256"]:
            bad_tok.append(r["request_id"])
            continue
        tid = [int(x) for x in tokenizer.encode(prompt)]
        want = rows.get(r["request_id"])
        if want is None or RI.token_ids_sha256(tid) != want["token_ids_sha256"]:
            bad_tok.append(r["request_id"])
            continue
        toks[r["request_id"]] = {"token_ids": tid, "input_tokens": len(tid), "token_ids_sha256": want["token_ids_sha256"],
                                 "context_status": want["context_status"]}
    if bad_tok or len(toks) != len(reqs) or _sha("\n".join(rows[r["request_id"]]["token_ids_sha256"] for r in reqs).encode("utf-8")) \
            != rec["models"][model_key]["token_ids_sha256_list"]:
        raise PreflightError([issue(str(pd), "E_A26C_BINDING", f"{len(bad_tok)} prompts or token arrays differ from the "
                                                               f"verified ones (first {bad_tok[:3]}); rerun the preflight")])
    pre = {"receipt": rec, "requests": reqs, "proto": proto, "prep": prep, "requests_sha256": b["requests_sha256"],
           "spec_sha256": b["spec_sha256"], "model_tokens": {model_key: {"rows": toks}}}
    ctx = context_for(pre, model_key, model_dir, policy=policy, evidence_fn=evidence_fn)
    binding = {"format_version": 1, "record_type": "a26c_binding", "model_key": model_key,
               "receipt_sha256": _sha((pd / "receipt.json").read_bytes()), "checks": checks,
               "requests_tokenized": len(toks), "duration_s": round(time.perf_counter() - t0, 1)}
    return ctx, binding


def context_for(pre, model_key, model_dir, policy=None, evidence_fn=None) -> dict:
    """A2.3d's runner context from a passed preflight: its rows, this model's token IDs, protocol and checkpoint
    evidence (the pinned revision; A2.3d's model_protocol and checkpoint_evidence, unchanged)."""
    from ..iref_vla.model import checkpoint_evidence
    from ..iref_vla_compare.run import _digest, model_protocol
    pol = dict(policy or CD.load_policy())
    pol["policy_id"] = RUN_ID
    proto = pre["proto"]
    toks = pre["model_tokens"][model_key]["rows"]
    rows = []
    for k, r in enumerate(pre["requests"], 1):
        mt = toks[r["request_id"]]
        rows.append({"policy_id": RUN_ID, "request_index": k, "request_id": r["request_id"], "selection_rank": k,
                     "parent_command_id": r["parent_command_id"], "view_id": r["view"], "format": r["format"],
                     "derived_scene_id": r["derived_scene_id"], "derived_command_id": r["derived_command_id"],
                     "object_count": len(r["choice_object_ids"]) - 1, "prompt_sha256": r["prompt_sha256"],
                     "mapping": [[c, t, i] for c, t, i in zip(r["codes"], r["choice_object_ids"], r["code_token_ids"])],
                     "models": {model_key: {kk: mt[kk] for kk in ("input_tokens", "token_ids_sha256", "context_status")}}})
    mp, desc = model_protocol(proto, pol, model_key)
    evidence = (evidence_fn or checkpoint_evidence)(model_dir, mp, desc, None)
    identity = {"run_id": RUN_ID, "requests_sha256": pre["requests_sha256"], "spec_sha256": pre["spec_sha256"],
                "model_key": model_key, "token_ids_sha256_list": pre["receipt"]["models"][model_key]["token_ids_sha256_list"],
                "code": _code()}
    return {"req": pre["prep"], "rman": identity, "rman_sha256": _sha(json.dumps(identity, sort_keys=True).encode("utf-8")),
            "policy": pol, "proto": proto, "model_proto": mp, "rows": rows,
            "ids": {rid: v["token_ids"] for rid, v in toks.items()}, "verify_ms": {rid: 0.0 for rid in toks},
            "evidence": evidence, "evidence_sha256": _digest(evidence), "model_key": model_key, "tokenizer": None}
