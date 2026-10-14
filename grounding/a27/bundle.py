"""A2.7 desktop bundle (D115): what the desktop run needs, built on the PC that holds A2.4's run and A2.6's benchmark.

    python -m grounding.a27 bundle --prep PREP --work WORK --evaluation EVAL --merged MERGED --out FOLDER

EVAL is A2.4's evaluation of the candidate (eval/parser-s1-parser), the float32 reference. The bundle checks that it
evaluated this merged model on the frozen A2.6 benchmark with the current language and parser prompt, then rebuilds every
parent's prompt exactly as A2.4 did (A2.4's own parent order, prompt function and pinned tokenizer; each prompt's length
must equal the one A2.4 recorded). It writes requests.jsonl (one row per parent: the prompt's token IDs, the reference
output and its two views' outcomes, and whether it belongs to the frozen headset subset or its mask sample),
grammar.gbnf and manifest.json.

The headset subset (D115) is frozen here, before any desktop or headset result: 300 of the development parents,
stratified by relation, at least 10 per relation and the rest in proportion (largest remainder, ties by name), each
relation's parents ranked by SHA-256 of the salt and the parent ID. The mask sample is the first 2 of each relation's
chosen parents in that ranking. The selection uses only parent IDs and relations, never an outcome.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..a24 import evaluate as E
from ..a24 import language as L
from ..a24 import prep as PR
from ..evaluation.iref_vla.protocol import encode_json, encode_jsonl

POLICY = "a27.bundle.v1"
SALT = "a27.headset-subset.v1"
SUBSET_SIZE = 300
MIN_PER_RELATION = 10
MASK_PER_RELATION = 2
END_TOKEN_TEXT = "<|im_end|>"


class BundleError(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rank_key(parent_id: str) -> str:
    return _sha(f"{SALT}|{parent_id}".encode("utf-8"))


def allocate(sizes: dict, total=SUBSET_SIZE, minimum=MIN_PER_RELATION) -> dict:
    """At least `minimum` per relation (or all of a smaller one), the rest in proportion to what is left of each, by
    largest remainder with ties broken by the relation's name."""
    if total > sum(sizes.values()):
        raise BundleError(f"a subset of {total} from {sum(sizes.values())} parents")
    base = {s: min(n, minimum) for s, n in sizes.items()}
    rest = total - sum(base.values())
    spare = {s: sizes[s] - base[s] for s in sizes}
    pool = sum(spare.values())
    quota = {s: rest * spare[s] / pool if pool else 0.0 for s in sizes}
    alloc = {s: base[s] + int(quota[s]) for s in sizes}
    left = total - sum(alloc.values())
    for s in sorted(sizes, key=lambda s: (-(quota[s] - int(quota[s])), s))[:left]:
        alloc[s] += 1
    if any(alloc[s] > sizes[s] for s in sizes):
        raise BundleError("the allocation exceeds a relation's parents")
    return dict(sorted(alloc.items()))


def select(parents) -> dict:
    """The frozen headset subset and its mask sample, from the development parents' IDs and relations only."""
    by = defaultdict(list)
    for p in parents:
        if p["partition"] == "development":
            by[p["stratum"]].append(p["parent_command_id"])
    alloc = allocate({s: len(v) for s, v in by.items()})
    chosen = {s: sorted(by[s], key=rank_key)[:alloc[s]] for s in sorted(by)}
    mask = {s: ids[:MASK_PER_RELATION] for s, ids in chosen.items()}
    subset = sorted(i for ids in chosen.values() for i in ids)
    sample = sorted(i for ids in mask.values() for i in ids)
    return {"salt": SALT, "size": len(subset), "allocation": alloc, "ids": subset, "mask_sample": sample,
            "ids_sha256": _sha("\n".join(subset).encode("utf-8")),
            "mask_sample_sha256": _sha("\n".join(sample).encode("utf-8"))}


def _views(row) -> dict:
    return {v: {k: x for k, x in o.items() if k != "resolve_ms"} for v, o in row["views"].items()}


def check_evaluation(evaluation: Path, merged: Path, prep: Path) -> tuple:
    man = json.loads((evaluation / "manifest.json").read_text(encoding="utf-8"))
    body = (evaluation / "results.jsonl").read_bytes()
    ident_sha = _sha((merged / "a24-identity.json").read_bytes())
    settings = PR.load_settings()
    problems = []
    if man.get("record_type") != "a24_parser_evaluation":
        problems.append("not an A2.4 parser evaluation")
    if man.get("identity_sha256") != ident_sha:
        problems.append("it evaluated another model than this merged folder")
    if man.get("outputs", {}).get("results.jsonl") != _sha(body):
        problems.append("results.jsonl changed since the evaluation")
    lock = man.get("lock", {})
    if lock.get("language_sha256") != L.spec_sha256():
        problems.append("the language changed since the evaluation")
    if lock.get("parser_prompt_sha256") != _sha(json.dumps(settings["parser_prompt"], sort_keys=True).encode()):
        problems.append("the parser prompt changed since the evaluation")
    if lock.get("prep_manifest_sha256") != _sha((prep / "manifest.json").read_bytes()):
        problems.append("another A2.6 preparation than the evaluation's")
    if problems:
        raise BundleError(f"{evaluation}: " + "; ".join(problems))
    rows = {r["parent_command_id"]: r for r in (json.loads(x) for x in body.decode("utf-8").splitlines() if x.strip())}
    return man, rows, settings


def bundle(*, prep, work, evaluation, merged, out, tokenizer=None) -> dict:
    from ..a24 import decoder as DC
    from ..preparation.iref_vla import tokens as T
    prep, work, evaluation, merged, out = Path(prep), Path(work), Path(evaluation), Path(merged), Path(out)
    if out.exists():
        raise BundleError(f"{out} already exists")
    man, rows, settings = check_evaluation(evaluation, merged, prep)
    parents = E._parents(prep, work)
    missing = [p["parent_command_id"] for p in parents if p["parent_command_id"] not in rows]
    if missing or len(rows) != len(parents):
        raise BundleError(f"the evaluation covers {len(rows)} parents, the benchmark {len(parents)}")
    tok = tokenizer or T.load_pinned_tokenizer(PR.DEV.TOKENIZER)
    spec = L.load_spec()
    table = DC.TokenTable(tok._backend)
    if table.eos_id != tok._backend.token_to_id(END_TOKEN_TEXT):
        raise BundleError("the decoder's end token is not <|im_end|>")
    sel = select(parents)
    subset, sample = set(sel["ids"]), set(sel["mask_sample"])
    out_rows, bad_len = [], []
    for p in parents:
        r = rows[p["parent_command_id"]]
        ids = tok.encode(PR.parser_prompt(p["text"], settings))
        if len(ids) != r["input_tokens"]:
            bad_len.append(p["parent_command_id"])
        out_rows.append({k: p[k] for k in ("partition", "group", "scene", "stratum", "parent_command_id")} | {
            "prompt_ids": list(ids),
            "reference": {"status": r["status"], "valid": r["valid"], "token_ids": r["token_ids"], "text": r["text"],
                          "views": _views(r)},
            "headset": p["parent_command_id"] in subset, "mask_sample": p["parent_command_id"] in sample})
    if bad_len:
        raise BundleError(f"{len(bad_len)} rebuilt prompts differ in length from A2.4's, e.g. {bad_len[0]}")
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True)
    req = encode_jsonl(out_rows)
    gbnf = L.gbnf(spec).encode("utf-8")
    (tmp / "requests.jsonl").write_bytes(req)
    (tmp / "grammar.gbnf").write_bytes(gbnf)
    manifest = {"format_version": 1, "record_type": "a27_bundle", "policy": POLICY,
                "candidate": {"model_name": man["model_name"], "identity_sha256": man["identity_sha256"]},
                "inputs": {"evaluation_manifest_sha256": _sha((evaluation / "manifest.json").read_bytes()),
                           "prep_manifest_sha256": _sha((prep / "manifest.json").read_bytes()),
                           "language_sha256": L.spec_sha256(),
                           "parser_prompt_sha256": man["lock"]["parser_prompt_sha256"]},
                "decoding": {"end_token": table.eos_id, "end_token_text": END_TOKEN_TEXT,
                             "max_new_tokens": spec["max_new_tokens"], "grammar_root": "root",
                             "greedy": "highest logit among the allowed tokens, ties to the lowest token ID",
                             "cache": "a fresh cache per command (keep 0), then within-request decoding only"},
                "parents": len(out_rows),
                "partitions": {k: sum(1 for r in out_rows if r["partition"] == k) for k in
                               sorted({r["partition"] for r in out_rows})},
                "subset": sel,
                "files": {"requests.jsonl": _sha(req), "grammar.gbnf": _sha(gbnf)}}
    (tmp / "manifest.json").write_bytes(encode_json(manifest))
    tmp.rename(out)
    return manifest
