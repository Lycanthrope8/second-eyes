"""A2.5 delivery 1 (D98, D99): the frozen-request replay bundle, built on the PC from existing artifacts only.

Inputs, each verified before use:
- the A2.3d request bundle, read back in full and pinned by its manifest's SHA-256;
- the 0.5B run `20261007_A2_r006` (`raw/results`), read back in full against the requests. Its manifest's full SHA-256 is
  the one the pinned A2.3d scores manifest records; only that `manifest.json` is opened in the scores folder, so no
  annotation, score or rules file is read;
- the pinned 0.5B tokenizer, which must be the one the requests were prepared with and must reproduce every frozen
  token list.

The replay list comes from the existing selection functions (A2.3e's `costs.select_cases` and `audit.audit_parents`) and
from the runs' canary rule applied within each format (`replay-policy.v1.json`). A float32 reference is reused only where
the prompt hash, token hash and mapping match exactly (D99(9)); its status must be completed, and its choice, shares and
tie handling are recomputed from its own offered logits. Both cache boundaries are verified as exact token prefixes. Five
written integration fixtures are frozen with the bundle, each stopping at one input check (D99(5)).

Output: a new folder, written beside its destination, read back in full, then renamed into place.
  manifest.json, policy.json, selection.json
  headset/replay-manifest.json, headset/requests.jsonl, headset/fixtures-written.jsonl   (what the headset reads)
  reference/references.jsonl                                                            (stays on the PC)
"""
from __future__ import annotations

import base64
import json
import math
import os
import shutil
import statistics
import tempfile
from pathlib import Path

from ..evaluation.iref_vla import output
from .publish import publish
from ..evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl, issue,
                                             runtime, sha256, strict_json)
from ..inference.iref_vla.choices import build_prompt, choices_line
from ..inference.iref_vla.prepare import rows_of
from ..inference.iref_vla.protocol import load_protocol
from ..inference.iref_vla_compare import design as D
from ..inference.iref_vla_compare.audit import audit_parents
from ..inference.iref_vla_compare.costs import select_cases
from ..inference.iref_vla_compare.prepare import verify_compare_requests
from ..inference.iref_vla_compare.run import verify_compare_results
from ..preparation.iref_vla import tokens as T
from . import replay_inputs as RI

REPO = Path(__file__).resolve().parents[2]
POLICY_PATH = Path(__file__).resolve().parent / "replay-policy.v1.json"
POLICY_ID = "a25.replay.frozen_requests.v1"
SOURCES = ("a23e_case", "a23e_audit", "longest_in_format")
SHARE_TOLERANCE = 1e-12   # recorded shares against their recomputation: exp() may differ in the last bit between platforms
MAX_CHARS_BACK = 64       # how far before the end the last token's text may start
FILES = ("policy.json", "selection.json", "headset/replay-manifest.json", "headset/requests.jsonl",
         "headset/fixtures-written.jsonl", "reference/references.jsonl")
CODE_DIRS = ("grounding/quest", "grounding/inference/iref_vla_compare", "grounding/inference/iref_vla")
CODE_FILES = ("grounding/preparation/iref_vla/tokens.py",)


def _fail(code, problems):
    raise EvaluationInputError([issue(w, code, m) for w, m in problems])


def code_hashes() -> dict:
    out = {}
    for d in CODE_DIRS:
        for p in sorted((REPO / d).glob("*")):
            if p.is_file() and p.suffix in (".py", ".json"):
                out[p.relative_to(REPO).as_posix()] = sha256(p.read_bytes())
    for rel in CODE_FILES:
        out[rel] = sha256((REPO / rel).read_bytes())
    return out


def _hex(x, n=64) -> bool:
    return isinstance(x, str) and len(x) == n and all(c in "0123456789abcdef" for c in x)


def load_policy(path=None) -> dict:
    p = Path(path) if path is not None else POLICY_PATH
    try:
        pol = strict_json("replay policy", p.read_bytes(), "E_REPLAY_POLICY")
    except OSError as e:
        raise EvaluationInputError([issue(str(p), "E_REPLAY_POLICY", f"cannot read: {e}")]) from None
    bad = []
    if not isinstance(pol, dict) or pol.get("record_type") != "a25_replay_policy" or pol.get("policy_id") != POLICY_ID:
        bad.append(f"not the {POLICY_ID} policy")
    else:
        inp, sel = pol.get("inputs") or {}, pol.get("selection") or {}
        wf, hs = pol.get("written_fixtures") or {}, pol.get("headset") or {}
        if not (_hex(inp.get("requests_manifest_sha256")) and _hex(inp.get("scores_manifest_sha256"))
                and isinstance(inp.get("small_run_id"), str)
                and isinstance(inp.get("small_run_manifest_sha256_prefix"), str)
                and len(inp["small_run_manifest_sha256_prefix"]) >= 8
                and _hex(inp["small_run_manifest_sha256_prefix"], len(inp["small_run_manifest_sha256_prefix"]))
                and inp.get("scores_files_read") == ["manifest.json"]):
            bad.append("inputs need two full SHA-256 pins, the reference run's ID and manifest prefix, and scores_files_read")
        if pol.get("model_key") not in ("qwen2.5-0.5b-instruct", "qwen2.5-7b-instruct"):
            bad.append("model_key must be one of A2.3d's model keys")
        exp = sel.get("expected") or {}
        if sel.get("sources") != list(SOURCES) or type(exp.get("a23e_case_requests")) is not int \
                or not isinstance(exp.get("a23e_audit_request_ids"), list) \
                or not all(isinstance(x, str) for x in exp["a23e_audit_request_ids"]):
            bad.append(f"selection needs the sources {list(SOURCES)}, the expected case count and the audit request IDs")
        if type(hs.get("vocab_size")) is not int or hs["vocab_size"] < 1 or hs.get("input_checks") != list(RI.CHECKS):
            bad.append(f"headset needs a positive vocab_size and input_checks {list(RI.CHECKS)}")
        fx = wf.get("fixtures")
        filler = wf.get("overflow_filler") or {}
        lines = wf.get("document_lines")
        try:
            last = json.loads(lines[-1]) if isinstance(lines, list) and lines and all(isinstance(x, str) for x in lines) else None
        except ValueError:
            last = None
        if not (isinstance(wf.get("mapping"), list) and all(isinstance(m, list) and len(m) == 2 for m in wf["mapping"])
                and isinstance(last, dict) and list(last) == ["command"] and isinstance(last["command"], str)
                and last["command"][:1].isalpha() and last["command"][:1].islower()
                and isinstance(filler.get("text_before_number"), str) and isinstance(filler.get("text_after_number"), str)
                and isinstance(fx, list) and [f.get("request_id") for f in fx] == list(FIXTURE_IDS)
                and all(f.get("expected_outcome") == RI.FAILS_AS[FIXTURE_CHECKS[f["request_id"]]]
                        and f.get("expected_reason") in RI.REASONS[FIXTURE_CHECKS[f["request_id"]]] for f in fx)):
            bad.append("written_fixtures needs a mapping, document lines ending with a command line whose text starts with "
                       f"a lower-case letter, the overflow filler, and the fixtures {list(FIXTURE_IDS)} with their outcomes")
    if bad:
        raise EvaluationInputError([issue(str(p), "E_REPLAY_POLICY", m) for m in bad])
    return pol


FIXTURE_IDS = ("written.context_overflow", "written.prompt_hash_mismatch", "written.mapping_ask_not_last",
               "written.code_token_mismatch", "written.token_out_of_vocabulary")
FIXTURE_CHECKS = {"written.context_overflow": "context", "written.prompt_hash_mismatch": "prompt_bytes",
                  "written.mapping_ask_not_last": "mapping", "written.code_token_mismatch": "mapping",
                  "written.token_out_of_vocabulary": "token_ids"}


# ---------------------------------------------------------------------------------------------------------- selection

def select_replay(rows, model_key) -> dict:
    """The replay list from A2.3d's request index (rows in request order). Pure; the expectations are checked apart."""
    index = {r["request_id"]: r["request_index"] for r in rows}
    cases = select_cases(rows)
    case_rows = [r for pv in cases for r in rows if (r["parent_command_id"], r["view_id"]) == pv]   # as cache.py
    parents = audit_parents(rows)
    audit_rows = [r for p in parents for r in rows if r["parent_command_id"] == p]                 # as audit.py
    longest = {}
    for fmt in D.FORMATS:
        in_format = [r for r in rows if r["format"] == fmt]
        if in_format:   # the runs' canary rule, within the format: ties go to the later request
            longest[fmt] = max(in_format, key=lambda r: (r["models"][model_key]["input_tokens"], r["request_index"]))
    sources = {}
    for tag, chosen in (("a23e_case", case_rows), ("a23e_audit", audit_rows), ("longest_in_format", list(longest.values()))):
        for r in chosen:
            tags = sources.setdefault(r["request_id"], [])
            if tag not in tags:
                tags.append(tag)
    ids = sorted(sources, key=lambda rid: index[rid])
    return {"format_version": 1, "record_type": "a25_replay_selection", "model_key": model_key,
            "a23e_cases": [list(pv) for pv in cases], "a23e_case_request_ids": [r["request_id"] for r in case_rows],
            "a23e_audit_parents": list(parents), "a23e_audit_request_ids": [r["request_id"] for r in audit_rows],
            "longest_in_format": {f: {"request_id": r["request_id"], "input_tokens": r["models"][model_key]["input_tokens"]}
                                  for f, r in longest.items()},
            "request_ids": ids, "count": len(ids), "list_sha256": D.list_sha256(ids),
            "sources": {rid: sources[rid] for rid in ids},
            "chosen_by_several": [rid for rid in ids if len(sources[rid]) > 1]}


def selection_problems(sel, policy) -> list:
    exp, bad = policy["selection"]["expected"], []
    cases = sel["a23e_case_request_ids"]
    if len(cases) != exp["a23e_case_requests"] or len(set(cases)) != len(cases):
        bad.append(f"{len(cases)} A2.3e case requests ({len(set(cases))} distinct); the policy expects "
                   f"{exp['a23e_case_requests']}")
    if sel["a23e_audit_request_ids"] != exp["a23e_audit_request_ids"]:
        bad.append(f"the audit requests are {sel['a23e_audit_request_ids']}; the policy expects {exp['a23e_audit_request_ids']}")
    if len(sel["longest_in_format"]) != len(D.FORMATS):
        bad.append(f"a longest request was found for {sorted(sel['longest_in_format'])}, not for every format")
    return bad


# ---------------------------------------------------------------------------------------------------------- boundaries

def prefix_token_count(tok, prompt: str, chars: int, ids):
    """The number of tokens of prompt[:chars] if they are exactly the first tokens of ids (a token boundary), else None."""
    pre = list(tok.encode(prompt[:chars]))
    return len(pre) if list(ids[:len(pre)]) == pre else None


def boundary_before_last_token(tok, prompt: str, ids) -> dict:
    """Keep n - 1 tokens: verified when the text before the last token re-tokenizes to exactly the first n - 1 tokens."""
    n = len(ids)
    if n < 2:
        raise ValueError("fewer than two tokens")
    want = list(ids[:n - 1])
    for chars in range(len(prompt) - 1, max(-1, len(prompt) - 1 - MAX_CHARS_BACK), -1):
        if list(tok.encode(prompt[:chars])) == want:
            return {"keep_tokens": n - 1, "prefix_chars": chars}
    raise ValueError(f"no text prefix within {MAX_CHARS_BACK} characters of the end re-tokenizes to the first n - 1 tokens")


def boundary_before_command_line(tok, prompt: str, ids, proto, mapping) -> dict:
    """Keep the tokens before the document's last line, which must be the command line; verified as a token prefix."""
    w = proto["wrapper"]
    head = w["before_system"] + proto["system_message"] + w["between"] + choices_line(mapping)
    tail = w["after_user"]
    if not (prompt.startswith(head) and prompt.endswith(tail) and len(prompt) > len(head) + len(tail)):
        raise ValueError("the prompt is not the protocol's head, a document and the open assistant turn")
    doc = prompt[len(head):len(prompt) - len(tail)]
    if not doc.endswith("\n"):
        raise ValueError("the document does not end with LF")
    start = doc.rfind("\n", 0, len(doc) - 1) + 1
    try:
        line = json.loads(doc[start:-1])
    except ValueError:
        line = None
    if not (isinstance(line, dict) and list(line) == ["command"] and isinstance(line["command"], str)):
        raise ValueError("the document's last line is not the command line")
    chars = len(head) + start
    keep = prefix_token_count(tok, prompt, chars, ids)
    if keep is None:
        raise ValueError("the command line does not start at a token boundary")
    if not 0 < keep < len(ids) - 1:
        raise ValueError(f"keeping {keep} of {len(ids)} tokens leaves no command tokens before the last token")
    return {"keep_tokens": keep, "prefix_chars": chars}


# ---------------------------------------------------------------------------------------------------------- references

def reference_problems(row, *, request_id, prompt_sha256, token_ids_sha256, mapping) -> list:
    """A float32 reference row is reusable only for exactly this request (D99(9)), completed, and self-consistent: its
    choice, shares and tie handling recomputed from its own offered logits."""
    rid = request_id
    if not isinstance(row, dict):
        return [f"{rid}: no reference row"]
    if row.get("technical_status") != "completed":
        return [f"{rid}: the reference run did not complete this request ({row.get('technical_status')!r})"]
    bad = []
    if (row.get("request_id"), row.get("prompt_sha256"), row.get("token_ids_sha256"), row.get("mapping")) \
            != (rid, prompt_sha256, token_ids_sha256, mapping):
        bad.append(f"{rid}: the reference's request ID, prompt hash, token hash or mapping differs from the request's")
    scores = row.get("scores") if isinstance(row.get("scores"), list) else []
    if [(s.get("code"), s.get("target"), s.get("token_id")) if isinstance(s, dict) else None for s in scores] \
            != [tuple(m) for m in mapping]:
        return bad + [f"{rid}: the reference's offered codes, targets or token IDs differ from the mapping"]
    offered = [s.get("logit") for s in scores]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in offered):
        return bad + [f"{rid}: a reference logit is missing or not finite"]
    top = max(offered)
    e = [math.exp(x - top) for x in offered]
    z = math.fsum(e)
    shares = [x / z for x in e]
    if any(type(s.get("restricted_share")) not in (int, float) or abs(a - s["restricted_share"]) > SHARE_TOLERANCE
           for a, s in zip(shares, scores)):
        bad.append(f"{rid}: a recorded restricted share differs from its recomputation by more than {SHARE_TOLERANCE}")
    tied = [m[0] for m, x in zip(mapping, offered) if x == top]
    ask = mapping[-1][0]
    code, reason = (ask, "exact_score_tie") if len(tied) > 1 else (tied[0], "max_offered_logit")
    target = None if code == ask else next(m[1] for m in mapping if m[0] == code)
    want = (code, reason, tied if len(tied) > 1 else [], target, "model_choice_ask" if code == ask else "model_choice_object")
    if (row.get("choice_code"), row.get("selection_reason"), row.get("tied_codes"), row.get("choice_object_id"),
            row.get("model_choice")) != want:
        bad.append(f"{rid}: the recorded choice differs from the one its offered logits give ({want[0]}, {want[1]})")
    ranked = sorted(offered, reverse=True)
    if row.get("logit_margin") != ranked[0] - ranked[1]:
        bad.append(f"{rid}: the recorded logit margin differs from its recomputation")
    t = row.get("top_restricted_share")
    if type(t) not in (int, float) or abs(max(shares) - t) > SHARE_TOLERANCE:
        bad.append(f"{rid}: the recorded top restricted share differs from its recomputation")
    return bad


# ---------------------------------------------------------------------------------------------------------- headset parts

def headset_constants(proto, requests_manifest, policy) -> dict:
    codes = list(proto["object_codes"])
    return {"context_limit_tokens": requests_manifest["context_limit_tokens"],
            "continuation_tokens": proto["continuation_tokens"], "vocab_size": policy["headset"]["vocab_size"],
            "object_codes": codes, "ask_code": proto["ask_code"], "ask_target": proto["ask_target"],
            "code_token_ids": {c: proto["code_token_ids"][c] for c in codes + [proto["ask_code"]]}}


def _constants_from_manifest(hc) -> dict:
    """The constants as the bundle records them (choice codes and token IDs as parallel lists)."""
    return {"context_limit_tokens": hc["context_limit_tokens"], "continuation_tokens": hc["continuation_tokens"],
            "vocab_size": hc["vocab_size"], "object_codes": list(hc["object_codes"]), "ask_code": hc["ask_code"],
            "ask_target": hc["ask_target"], "code_token_ids": dict(zip(hc["choice_codes"], hc["choice_token_ids"]))}


def constants_record(c) -> dict:
    """The constants in the flat form the headset reads."""
    choice = list(c["object_codes"]) + [c["ask_code"]]
    return {"context_limit_tokens": c["context_limit_tokens"], "continuation_tokens": c["continuation_tokens"],
            "vocab_size": c["vocab_size"], "object_codes": list(c["object_codes"]), "ask_code": c["ask_code"],
            "ask_target": c["ask_target"], "choice_codes": choice,
            "choice_token_ids": [c["code_token_ids"][x] for x in choice]}


def written_fixtures(policy, proto, tok, constants) -> tuple:
    """The five written integration fixtures, each failing exactly one input check, and the overflow's evidence."""
    wf = policy["written_fixtures"]
    mapping = [[c, t, constants["code_token_ids"][c]] for c, t in wf["mapping"]]
    lines, fill = wf["document_lines"], wf["overflow_filler"]

    def text(filler: int) -> str:
        doc = "".join(x + "\n" for x in lines[:-1]) \
            + "".join(fill["text_before_number"] + str(k) + fill["text_after_number"] + "\n" for k in range(1, filler + 1)) \
            + lines[-1] + "\n"
        return build_prompt(proto, mapping, doc)

    limit, cont = constants["context_limit_tokens"], constants["continuation_tokens"]

    def count(filler: int) -> int:
        return len(tok.encode(text(filler)))

    lo, hi = 0, 1   # the fewest filler lines that exceed the limit, by doubling then bisection over a monotone count
    if count(0) + cont > limit:
        hi = 0
    else:
        while count(hi) + cont <= limit:
            lo, hi = hi, hi * 2
            if hi > 1_000_000:
                raise RuntimeError("the overflow fixture cannot reach the context limit")
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if count(mid) + cont > limit else (mid, hi)
    over_text = text(hi)
    over_ids = list(tok.encode(over_text))
    evidence = {"filler_lines": hi, "tokens": len(over_ids), "tokens_one_line_fewer": count(hi - 1) if hi > 0 else None,
                "rule": "the fewest filler lines whose tokens plus the continuation exceed the context limit"}
    spec = {f["request_id"]: f for f in wf["fixtures"]}

    def record(rid, data: bytes, ids):
        return RI.make_record(request_id=rid, kind="written_integration_fixture", prompt=data, token_ids=ids,
                              mapping=mapping, keep_before_last_token=RI.NOT_EVALUATED,
                              keep_before_command_line=RI.NOT_EVALUATED, expected_outcome=spec[rid]["expected_outcome"],
                              expected_reason=spec[rid]["expected_reason"])

    base_text = text(0)
    base, base_ids = base_text.encode("utf-8"), list(tok.encode(base_text))
    out = [record("written.context_overflow", over_text.encode("utf-8"), over_ids)]
    r = record("written.prompt_hash_mismatch", base, base_ids)
    k = base_text.rindex('{"command":"') + len('{"command":"')
    changed = base_text[:k] + base_text[k].upper() + base_text[k + 1:]
    r["prompt_b64"] = base64.b64encode(changed.encode("utf-8")).decode("ascii")   # same length; prompt_sha256 stays
    out.append(r)
    r = record("written.mapping_ask_not_last", base, base_ids)
    for f in ("codes", "targets", "code_token_ids"):
        r[f] = r[f][-1:] + r[f][:-1]
    r["mapping_sha256"] = RI.mapping_sha256(r["codes"], r["targets"], r["code_token_ids"])
    out.append(r)
    r = record("written.code_token_mismatch", base, base_ids)
    a = r["codes"].index("A")
    r["code_token_ids"][a] = constants["code_token_ids"]["A"] + 1
    r["mapping_sha256"] = RI.mapping_sha256(r["codes"], r["targets"], r["code_token_ids"])
    out.append(r)
    r = record("written.token_out_of_vocabulary", base, base_ids)
    r["token_ids"][-1] = constants["vocab_size"]
    r["token_ids_sha256"] = RI.token_ids_sha256(r["token_ids"])
    out.append(r)
    return out, evidence


# ---------------------------------------------------------------------------------------------------------- build

def _stats(values) -> dict:
    v = sorted(values)
    return {"n": len(v), "min": v[0], "median": statistics.median(v), "max": v[-1], "total": sum(v)} if v else {"n": 0}


def _read_inputs(policy, requests, small_run, scores):
    """Every input verified and tied to its pin; returns what the build and the verifier need."""
    key, pins = policy["model_key"], policy["inputs"]
    req, run, sc = Path(requests), Path(small_run), Path(scores)
    bad = verify_compare_requests(req)
    if bad:
        _fail("E_REPLAY_INPUT", [(str(req), m) for m in bad[:20]])
    rman_bytes = (req / "manifest.json").read_bytes()
    if sha256(rman_bytes) != pins["requests_manifest_sha256"]:
        _fail("E_REPLAY_INPUT", [(str(req), f"manifest SHA-256 {sha256(rman_bytes)} is not the pinned "
                                            f"{pins['requests_manifest_sha256']}")])
    rman = strict_json("requests manifest", rman_bytes, "E_REPLAY_INPUT")
    proto = load_protocol(req / "protocol.json")
    if proto.get("context_limit_tokens") != rman.get("context_limit_tokens"):
        _fail("E_REPLAY_INPUT", [(str(req), "the protocol's and the requests' context limits differ")])
    try:
        sman_bytes = (sc / "manifest.json").read_bytes()
    except OSError as e:
        _fail("E_REPLAY_INPUT", [(str(sc), f"cannot read the scores manifest: {e}")])
    if sha256(sman_bytes) != pins["scores_manifest_sha256"]:
        _fail("E_REPLAY_INPUT", [(str(sc), f"manifest SHA-256 {sha256(sman_bytes)} is not the pinned "
                                           f"{pins['scores_manifest_sha256']}")])
    sman = strict_json("scores manifest", sman_bytes, "E_REPLAY_INPUT")
    links = sman.get("inputs") if isinstance(sman, dict) else None
    if not isinstance(links, dict) or sman.get("record_type") != "iref_compare_score_manifest" \
            or links.get("requests_manifest_sha256") != sha256(rman_bytes):
        _fail("E_REPLAY_INPUT", [(str(sc), "not an A2.3d scores manifest made from these requests")])
    try:
        run_bytes = (run / "manifest.json").read_bytes()
    except OSError as e:
        _fail("E_REPLAY_INPUT", [(str(run), f"cannot read the run's manifest: {e}")])
    run_sha = sha256(run_bytes)
    if run_sha != links.get("small_run_manifest_sha256") or not run_sha.startswith(pins["small_run_manifest_sha256_prefix"]):
        _fail("E_REPLAY_INPUT", [(str(run), f"manifest SHA-256 {run_sha} is not the run the scores manifest records "
                                            f"({links.get('small_run_manifest_sha256')}) with the recorded prefix "
                                            f"{pins['small_run_manifest_sha256_prefix']}")])
    run_man = strict_json("run manifest", run_bytes, "E_REPLAY_INPUT")
    if run_man.get("model_key") != key:
        _fail("E_REPLAY_INPUT", [(str(run), f"a run of {run_man.get('model_key')!r}; the replay needs {key}")])
    bad = verify_compare_results(run, req)
    if bad:
        _fail("E_REPLAY_INPUT", [(str(run), m) for m in bad[:20]])
    rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_REPLAY_INPUT")
    prompts = {x["request_id"]: x["prompt"] for x in rows_of("prompts.jsonl", (req / "prompts.jsonl").read_bytes(),
                                                              "E_REPLAY_INPUT")}
    frozen = {x["request_id"]: x["token_ids"] for x in rows_of(f"tokens/{key}.jsonl", (req / "tokens" / f"{key}.jsonl").read_bytes(),
                                                                 "E_REPLAY_INPUT")}
    results = {x["request_id"]: x for x in rows_of("results.jsonl", (run / "results.jsonl").read_bytes(), "E_REPLAY_INPUT")}
    return {"rman": rman, "rman_sha256": sha256(rman_bytes), "sman_sha256": sha256(sman_bytes), "run_sha256": run_sha,
            "proto": proto, "rows": rows, "prompts": prompts, "frozen": frozen, "results": results}


def _dataset_parts(policy, src, sel, tok, constants):
    """The headset record and the reference row of every selected request, or the list of problems."""
    key, proto, problems = policy["model_key"], src["proto"], []
    by_id = {r["request_id"]: r for r in src["rows"]}
    records, refs = [], []
    for rid in sel["request_ids"]:
        r, prompt, ids = by_id[rid], src["prompts"][rid], src["frozen"][rid]
        if list(tok.encode(prompt)) != list(ids):
            problems.append(("E_REPLAY_TOKENS", rid, "the pinned tokenizer does not reproduce the frozen token IDs"))
            continue
        try:
            last = boundary_before_last_token(tok, prompt, ids)
            cmd = boundary_before_command_line(tok, prompt, ids, proto, r["mapping"])
        except ValueError as e:
            problems.append(("E_REPLAY_BOUNDARY", rid, str(e)))
            continue
        rec = RI.make_record(request_id=rid, kind="dataset_command", prompt=prompt.encode("utf-8"), token_ids=ids,
                             mapping=r["mapping"], keep_before_last_token=last["keep_tokens"],
                             keep_before_command_line=cmd["keep_tokens"], expected_outcome="completed", expected_reason=None)
        if (rec["prompt_sha256"], rec["token_ids_sha256"]) != (r["prompt_sha256"], r["models"][key]["token_ids_sha256"]):
            problems.append(("E_REPLAY_INPUT", rid, "the prompt or token hash differs from the request index"))
        chk = RI.check_inputs(rec, constants, tokenize=tok.encode)
        if chk["outcome"] != RI.PASSED:
            problems.append(("E_REPLAY_INPUT", rid, f"the input checks stop it at {chk['check']}: {chk['reason']}"))
        problems += [("E_REPLAY_REFERENCE", rid, m) for m in reference_problems(
            src["results"].get(rid), request_id=rid, prompt_sha256=r["prompt_sha256"],
            token_ids_sha256=r["models"][key]["token_ids_sha256"], mapping=r["mapping"])]
        records.append(rec)
        refs.append({"format_version": 1, "record_type": "a25_replay_reference", "request_id": rid,
                     "request_index": r["request_index"], "sources": sel["sources"][rid],
                     "parent_command_id": r["parent_command_id"], "view_id": r["view_id"], "format": r["format"],
                     "object_count": r["object_count"], "prompt_sha256": rec["prompt_sha256"],
                     "token_ids_sha256": rec["token_ids_sha256"], "mapping_sha256": rec["mapping_sha256"],
                     "input_tokens": rec["input_tokens"],
                     "boundaries": {"keep_before_last_token": last, "keep_before_command_line": cmd},
                     "reference_run_id": policy["inputs"]["small_run_id"], "reference": src["results"].get(rid)})
    return records, refs, problems


def build_replay_bundle(*, requests, small_run, scores, tokenizer_dir, out, policy=None, tokenizer=None) -> dict:
    """Build, read back and publish the bundle; returns its manifest. Any mismatch stops it, every one listed."""
    out = output.refuse_existing(out)
    pol_path = Path(policy) if policy is not None else POLICY_PATH
    pol_bytes = pol_path.read_bytes()
    pol = load_policy(pol_path)
    key = pol["model_key"]
    src = _read_inputs(pol, requests, small_run, scores)
    tok = tokenizer if tokenizer is not None else T.load_pinned_tokenizer(tokenizer_dir)
    if tok.identity != src["rman"]["tokenizers"][key]["identity"]:
        _fail("E_REPLAY_TOKENIZER", [("tokenizer", f"{tok.identity} is not the tokenizer the requests were prepared with "
                                                   f"({src['rman']['tokenizers'][key]['identity']})")])
    sel = select_replay(src["rows"], key)
    bad = selection_problems(sel, pol)
    if bad:
        _fail("E_REPLAY_SELECTION", [("selection", m) for m in bad])
    constants = headset_constants(src["proto"], src["rman"], pol)
    records, refs, problems = _dataset_parts(pol, src, sel, tok, constants)
    if problems:
        raise EvaluationInputError([issue(rid, code, m) for code, rid, m in problems])
    fixtures, evidence = written_fixtures(pol, src["proto"], tok, constants)
    for f in fixtures:
        got = RI.check_inputs(f, constants, tokenize=tok.encode)
        if (got["outcome"], got["reason"]) != (f["expected_outcome"], f["expected_reason"]):
            raise RuntimeError(f"written fixture {f['request_id']} gives {got['outcome']}/{got['reason']}, not its "
                               f"expected {f['expected_outcome']}/{f['expected_reason']}")
    requests_bytes, fixtures_bytes = encode_jsonl(records), encode_jsonl(fixtures)
    hm = {"format_version": 1, "record_type": "a25_replay_headset_manifest", "policy_id": pol["policy_id"],
          "requests_file": "requests.jsonl", "requests_sha256": sha256(requests_bytes), "requests_count": len(records),
          "fixtures_file": "fixtures-written.jsonl", "fixtures_sha256": sha256(fixtures_bytes),
          "fixtures_count": len(fixtures), "input_checks": list(RI.CHECKS), **constants_record(constants)}
    files = {"policy.json": pol_bytes, "selection.json": encode_json(sel), "headset/replay-manifest.json": encode_json(hm),
             "headset/requests.jsonl": requests_bytes, "headset/fixtures-written.jsonl": fixtures_bytes,
             "reference/references.jsonl": encode_jsonl(refs)}
    by_source = {t: sum(t in sel["sources"][rid] for rid in sel["request_ids"]) for t in SOURCES}
    manifest = {"format_version": 1, "record_type": "a25_replay_bundle_manifest", "policy_id": pol["policy_id"],
                "policy_sha256": sha256(pol_bytes), "model_key": key,
                "inputs": {"requests_manifest_sha256": src["rman_sha256"], "scores_manifest_sha256": src["sman_sha256"],
                           "small_run_id": pol["inputs"]["small_run_id"], "small_run_manifest_sha256": src["run_sha256"],
                           "tokenizer_identity": tok.identity, "tokenizer_kind": getattr(tok, "kind", None),
                           "tokenizer_versions": getattr(tok, "versions", None)},
                "selection": {"count": sel["count"], "list_sha256": sel["list_sha256"], "by_source": by_source,
                              "chosen_by_several": len(sel["chosen_by_several"]),
                              "longest_in_format": sel["longest_in_format"]},
                "input_tokens": _stats([r["input_tokens"] for r in records]),
                "tokens_after_command_boundary": _stats([r["input_tokens"] - r["keep_before_command_line"] for r in records]),
                "fixtures": {"count": len(fixtures), "request_ids": [f["request_id"] for f in fixtures], "overflow": evidence},
                "headset_constants": constants_record(constants), "files": {k: sha256(v) for k, v in sorted(files.items())},
                "code": code_hashes(), "runtime": runtime(),
                "notes": ["Built on the PC from existing artifacts only; no model was run.",
                          "headset/ is what the headset reads; reference/ stays on the PC.",
                          "Dataset commands come from A2.3d's requests; written integration fixtures are implementation "
                          "checks, not dataset data or a benchmark (D99(5)).",
                          "A float32 reference is reused only where prompt bytes, token IDs and mapping match (D99(9))."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for rel, data in sorted(files.items()):
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            output._write_file(staging / rel, data)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = verify_replay_bundle(staging, requests=requests, small_run=small_run, scores=scores, tokenizer=tok)
        if bad:
            raise RuntimeError("the bundle failed readback: " + "; ".join(bad[:5]))
        publish(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


# ---------------------------------------------------------------------------------------------------------- readback

def verify_replay_bundle(folder, *, requests=None, small_run=None, scores=None, tokenizer=None) -> list:
    """Problems found, or []. On its own: files and hashes, every record's input checks, boundaries, references'
    consistency, selection and fixtures. With requests: the selection re-derived and every record equal to its source;
    with small_run and scores too: the reference chain and every reference row; with a tokenizer: re-tokenization,
    both boundaries and the fixtures rebuilt."""
    f, bad = Path(folder), []
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_REPLAY_BUNDLE")
        files = manifest["files"]
    except (OSError, EvaluationInputError, KeyError, TypeError) as e:
        return [f"unreadable manifest: {e}"]
    on_disk = {p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file()} - {"manifest.json"}
    if on_disk != set(FILES) or set(files) != set(FILES):
        return [f"files differ from the expected set: on disk {sorted(on_disk)}, in the manifest {sorted(files)}"]
    bad += [f"{rel}: changed" for rel in FILES if sha256((f / rel).read_bytes()) != files[rel]]
    if bad:
        return bad
    try:
        pol = load_policy(f / "policy.json")
        sel = strict_json("selection.json", (f / "selection.json").read_bytes(), "E_REPLAY_BUNDLE")
        hm = strict_json("replay-manifest.json", (f / "headset" / "replay-manifest.json").read_bytes(), "E_REPLAY_BUNDLE")
        recs = rows_of("requests.jsonl", (f / "headset" / "requests.jsonl").read_bytes(), "E_REPLAY_BUNDLE")
        fixes = rows_of("fixtures-written.jsonl", (f / "headset" / "fixtures-written.jsonl").read_bytes(), "E_REPLAY_BUNDLE")
        refs = rows_of("references.jsonl", (f / "reference" / "references.jsonl").read_bytes(), "E_REPLAY_BUNDLE")
        constants = _constants_from_manifest(manifest["headset_constants"])
    except (OSError, EvaluationInputError, KeyError, TypeError) as e:
        return [f"unreadable bundle: {e}"]
    if not (isinstance(sel, dict) and isinstance(hm, dict) and isinstance(manifest.get("selection"), dict)
            and isinstance(manifest.get("inputs"), dict) and isinstance(manifest.get("fixtures"), dict)
            and isinstance(sel.get("sources"), dict)):
        return ["selection.json, headset/replay-manifest.json or the manifest is not the expected record"]
    if manifest.get("policy_sha256") != files["policy.json"] or manifest.get("policy_id") != pol["policy_id"]:
        bad.append("the manifest names another policy")
    key = pol["model_key"]
    if any(hm.get(k) != v for k, v in manifest["headset_constants"].items()) or hm.get("input_checks") != list(RI.CHECKS) \
            or (hm.get("requests_sha256"), hm.get("fixtures_sha256"), hm.get("requests_count"), hm.get("fixtures_count")) \
            != (files["headset/requests.jsonl"], files["headset/fixtures-written.jsonl"], len(recs), len(fixes)) \
            or (hm.get("requests_file"), hm.get("fixtures_file")) != ("requests.jsonl", "fixtures-written.jsonl"):
        bad.append("headset/replay-manifest.json disagrees with the bundle (files, counts, constants or check order)")
    # selection
    ids = [r.get("request_id") for r in recs]
    if sel.get("request_ids") != ids or sel.get("count") != len(ids) or sel.get("list_sha256") != D.list_sha256(ids) \
            or manifest["selection"].get("list_sha256") != sel.get("list_sha256") or len(set(ids)) != len(ids) \
            or sorted(sel.get("sources", {})) != sorted(ids) \
            or any(not t or any(x not in SOURCES for x in t) for t in sel.get("sources", {}).values()):
        bad.append("selection.json disagrees with the requests (order, count, list hash or sources)")
    if not bad:
        try:
            bad += [f"selection: {m}" for m in selection_problems(sel, pol)]
        except (KeyError, TypeError) as e:
            bad.append(f"selection.json lacks a field: {e}")
    # dataset records
    tokenize = tokenizer.encode if tokenizer is not None else None
    for rec in recs:
        rid = rec.get("request_id")
        if sorted(rec) != sorted(RI.FIELDS) or rec.get("kind") != "dataset_command" or rec.get("record_type") != RI.RECORD_TYPE \
                or (rec.get("expected_outcome"), rec.get("expected_reason")) != ("completed", None):
            bad.append(f"{rid}: not a dataset replay record")
            continue
        chk = RI.check_inputs(rec, constants, tokenize=tokenize)
        if chk["outcome"] != RI.PASSED:
            bad.append(f"{rid}: the input checks stop it at {chk['check']} ({chk['reason']}: {chk['detail']})")
        n, kl, kc = rec["input_tokens"], rec["keep_before_last_token"], rec["keep_before_command_line"]
        if not (type(kl) is int and type(kc) is int and kl == n - 1 and 0 < kc < kl):
            bad.append(f"{rid}: boundaries {kc} and {kl} are not 0 < command < last = {n - 1}")
    # references
    if [r.get("request_id") for r in refs] != ids:
        bad.append("reference/references.jsonl does not hold one row per request, in order")
    else:
        for rec, ref in zip(recs, refs):
            rid = rec["request_id"]
            mapping = [[c, t, i] for c, t, i in zip(rec["codes"], rec["targets"], rec["code_token_ids"])]
            b = ref.get("boundaries") or {}
            if (ref.get("prompt_sha256"), ref.get("token_ids_sha256"), ref.get("mapping_sha256"), ref.get("input_tokens"),
                    (b.get("keep_before_last_token") or {}).get("keep_tokens"),
                    (b.get("keep_before_command_line") or {}).get("keep_tokens"), ref.get("sources"),
                    ref.get("reference_run_id")) != \
                    (rec["prompt_sha256"], rec["token_ids_sha256"], rec["mapping_sha256"], rec["input_tokens"],
                     rec["keep_before_last_token"], rec["keep_before_command_line"], (sel.get("sources") or {}).get(rid),
                     pol["inputs"]["small_run_id"]):
                bad.append(f"{rid}: the reference row disagrees with its headset record")
            bad += reference_problems(ref.get("reference"), request_id=rid, prompt_sha256=rec["prompt_sha256"],
                                      token_ids_sha256=rec["token_ids_sha256"], mapping=mapping)
    # written fixtures
    spec = {x["request_id"]: x for x in pol["written_fixtures"]["fixtures"]}
    if [x.get("request_id") for x in fixes] != list(FIXTURE_IDS):
        bad.append(f"the written fixtures are {[x.get('request_id') for x in fixes]}, not {list(FIXTURE_IDS)}")
    else:
        for x in fixes:
            rid = x["request_id"]
            if sorted(x) != sorted(RI.FIELDS) or x.get("kind") != "written_integration_fixture" \
                    or (x.get("keep_before_last_token"), x.get("keep_before_command_line")) != (RI.NOT_EVALUATED,) * 2 \
                    or (x.get("expected_outcome"), x.get("expected_reason")) \
                    != (spec[rid]["expected_outcome"], spec[rid]["expected_reason"]):
                bad.append(f"{rid}: not the policy's written fixture")
                continue
            got = RI.check_inputs(x, constants, tokenize=tokenize)
            if (got["outcome"], got["check"], got["reason"]) != (x["expected_outcome"], FIXTURE_CHECKS[rid], x["expected_reason"]):
                bad.append(f"{rid}: gives {got['outcome']} at {got['check']} ({got['reason']}), not its expected "
                           f"{x['expected_outcome']} at {FIXTURE_CHECKS[rid]} ({x['expected_reason']})")
    if bad or requests is None:
        return bad
    # against the sources
    try:
        src = _read_inputs(pol, requests, small_run, scores) if small_run is not None and scores is not None else None
    except EvaluationInputError as e:
        return [f"sources: {i['code']} {i['message']}" for i in e.issues]
    if src is None:
        req = Path(requests)
        rb = verify_compare_requests(req)
        if rb:
            return [f"requests: {m}" for m in rb[:20]]
        rman_bytes = (req / "manifest.json").read_bytes()
        src = {"rman": json.loads(rman_bytes), "rman_sha256": sha256(rman_bytes), "proto": load_protocol(req / "protocol.json"),
               "rows": rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_REPLAY_INPUT"),
               "prompts": {x["request_id"]: x["prompt"] for x in rows_of("prompts.jsonl", (req / "prompts.jsonl").read_bytes(),
                                                                          "E_REPLAY_INPUT")},
               "frozen": {x["request_id"]: x["token_ids"] for x in rows_of(f"tokens/{key}.jsonl",
                                                                             (req / "tokens" / f"{key}.jsonl").read_bytes(),
                                                                             "E_REPLAY_INPUT")},
               "results": None}
    if src["rman_sha256"] != manifest["inputs"].get("requests_manifest_sha256") \
            or src["rman_sha256"] != pol["inputs"]["requests_manifest_sha256"]:
        bad.append("the request bundle is not the one the bundle was built from")
    if select_replay(src["rows"], key) != sel:
        bad.append("selection.json differs from the selection re-derived from the request index")
    if headset_constants(src["proto"], src["rman"], pol) != constants:
        bad.append("the bundle's constants differ from the requests' protocol")
    by_id = {r["request_id"]: r for r in src["rows"]}
    for rec in recs:
        rid, r = rec["request_id"], by_id.get(rec["request_id"])
        if r is None or RI.decode_prompt(rec) != src["prompts"][rid].encode("utf-8") or rec["token_ids"] != src["frozen"][rid] \
                or [[c, t, i] for c, t, i in zip(rec["codes"], rec["targets"], rec["code_token_ids"])] != r["mapping"] \
                or rec["prompt_sha256"] != r["prompt_sha256"] or rec["token_ids_sha256"] != r["models"][key]["token_ids_sha256"]:
            bad.append(f"{rid}: the headset record differs from its A2.3d request")
    if src["results"] is not None:
        if (src["sman_sha256"], src["run_sha256"]) != (manifest["inputs"].get("scores_manifest_sha256"),
                                                      manifest["inputs"].get("small_run_manifest_sha256")):
            bad.append("the scores manifest or the reference run is not the one the bundle was built from")
        for ref in refs:
            if ref.get("reference") != src["results"].get(ref["request_id"]):
                bad.append(f"{ref['request_id']}: the reference row differs from the run's")
    if tokenizer is not None:
        if tokenizer.identity != src["rman"]["tokenizers"][key]["identity"] \
                or tokenizer.identity != manifest["inputs"].get("tokenizer_identity"):
            bad.append("the tokenizer is not the one the requests and the bundle were made with")
        for rec, ref in zip(recs, refs):
            rid, prompt = rec["request_id"], src["prompts"][rec["request_id"]]
            try:
                last = boundary_before_last_token(tokenizer, prompt, rec["token_ids"])
                cmd = boundary_before_command_line(tokenizer, prompt, rec["token_ids"], src["proto"], by_id[rid]["mapping"])
            except ValueError as e:
                bad.append(f"{rid}: {e}")
                continue
            if ref["boundaries"] != {"keep_before_last_token": last, "keep_before_command_line": cmd}:
                bad.append(f"{rid}: the boundaries differ from their recomputation")
        rebuilt, evidence = written_fixtures(pol, src["proto"], tokenizer, constants)
        if rebuilt != fixes or evidence != manifest["fixtures"].get("overflow"):
            bad.append("the written fixtures differ from a rebuild")
    return bad
