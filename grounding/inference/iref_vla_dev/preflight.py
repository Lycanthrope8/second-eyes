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


def preflight(*, prep, work, tokenizers: dict, frozen=FROZEN, progress=print) -> dict:
    """The verified requests and per-model tokens, or PreflightError with every problem. tokenizers: model key ->
    tokenizer (load_tokenizers)."""
    t_start = time.perf_counter()
    prep, work = Path(prep), Path(work)
    problems = []

    def bad(check, where, message):
        problems.append({"check": check, "where": str(where), "message": message})

    manifest = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
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
    proto = load_protocol()
    # ---- the request set
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
    for p, s in per.items():
        if s != full:
            bad("coverage", p, f"has {sorted(s)}, not both views x both formats")
    selected = {x["parent_command_id"] for x in sampling["selected"] if x["requests_built"]}
    if set(per) != selected:
        bad("coverage", "sampling.json", f"{len(set(per) ^ selected)} parents differ from the sampling's selected parents")
    # ---- against the bundles, the mapping, the prompt and the 0.5B tokens
    small = tokenizers[MODEL_KEYS[0]]
    bundles, prompts = {}, {}
    for k, r in enumerate(reqs, 1):
        rid, scene = r["request_id"], r["scene"]
        if scene not in bundles:
            try:
                bundles[scene] = read_bundle(work / scene / "bundle")
            except Exception as e:   # noqa: BLE001
                bad("bundle", scene, f"the bundle does not read: {type(e).__name__}: {e}"[:300])
                bundles[scene] = None
        src = bundles[scene]
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
        if k % 2000 == 0:
            progress(f"  verified {k}/{len(reqs)} requests")
    # ---- each model's own tokenizer
    model_tokens = {}
    for key, tok in tokenizers.items():
        out, same = {}, 0
        for r in reqs:
            if r["request_id"] not in prompts:
                continue
            prompt, mapping = prompts[r["request_id"]]
            tid = [int(x) for x in tok.encode(prompt)]
            want = [[c, t, i] for c, t, i in zip(r["codes"], r["choice_object_ids"], r["code_token_ids"])]
            if [list(x) for x in check_boundary(tok, prompt, mapping, proto)] != want:
                bad("boundaries", r["request_id"], f"{key}: an offered letter's one-token boundary differs")
            same += tid == r["token_ids"]
            out[r["request_id"]] = {"token_ids": tid, "input_tokens": len(tid), "token_ids_sha256": RI.token_ids_sha256(tid),
                                    "context_status": context_status(len(tid), proto["context_limit_tokens"], proto["continuation_tokens"])}
        model_tokens[key] = {"rows": out, "identity": getattr(tok, "identity", None), "same_as_prepared": same}
    receipt = {
        "format_version": 1, "record_type": "a26c_preflight", "run_id": RUN_ID, "frozen": dict(frozen),
        "passed": not problems, "problems": problems[:MAX_LISTED], "problem_count": len(problems),
        "counts": {"requests": len(reqs), "parents": len(per), "scenes": len(bundles),
                   "by_partition": {p: sum(1 for r in reqs if r["partition"] == p) for p in sorted({r["partition"] for r in reqs})}},
        "models": {k: {"tokenizer_identity": v["identity"], "same_token_ids_as_prepared": v["same_as_prepared"],
                       "token_ids_sha256_list": _sha("\n".join(x["token_ids_sha256"] for x in v["rows"].values()).encode())}
                   for k, v in model_tokens.items()},
        "answers": "reference_only/ was not opened; scoring verifies those files against the manifest before use",
        "protocol_sha256": _sha(json.dumps(proto, sort_keys=True).encode("utf-8")),
        "code": {p.name: _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "runtime": runtime(), "duration_s": round(time.perf_counter() - t_start, 1)}
    if problems:
        err = PreflightError([issue(p["where"], "E_A26C_PREFLIGHT", f"{p['check']}: {p['message']}") for p in problems[:MAX_LISTED]])
        err.receipt = receipt
        raise err
    return {"receipt": receipt, "requests": reqs, "prompts": prompts, "model_tokens": model_tokens, "proto": proto,
            "manifest": manifest, "requests_sha256": _sha(req_bytes), "spec_sha256": _sha(spec_bytes), "prep": prep}


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
                "model_key": model_key, "token_ids_sha256_list": pre["receipt"]["models"][model_key]["token_ids_sha256_list"]}
    return {"req": pre["prep"], "rman": identity, "rman_sha256": _sha(json.dumps(identity, sort_keys=True).encode("utf-8")),
            "policy": pol, "proto": proto, "model_proto": mp, "rows": rows,
            "ids": {rid: v["token_ids"] for rid, v in toks.items()}, "verify_ms": {rid: 0.0 for rid in toks},
            "evidence": evidence, "evidence_sha256": _digest(evidence), "model_key": model_key, "tokenizer": None}
