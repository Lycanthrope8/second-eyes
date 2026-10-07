"""A2.3e input audit: the exact inputs of the canary parents' requests, checked part by part, on the laptop.

The canaries showed that the two logit paths agree; they do not show that the prompt is right. This export takes the
parents of A2.3d's canary requests (the first request, and each model's longest request), all four view/format variants
of each, and writes for every request the exact prompt bytes, both models' token IDs, the role boundaries as character
and token offsets, the choices mapping, the scene-document hash and the command text. Then it checks:

- roles: the prompt is exactly <|im_start|>system, the protocol's system message, <|im_end|>, the user turn and an open
  assistant turn, and every role boundary falls on a token boundary for both tokenizers;
- content: the user turn is the choices line followed by the bundle's rendered document byte for byte; the document's
  object list is the subscene's; its command line carries the derived command's exact text;
- choices: letters A..J for the objects, each object once, K last and K always ASK;
- no answers: the prompt is fully rebuilt from the protocol, the mapping and the document alone, and contains no
  annotation ID, target field or rules outcome (annotations, rules results and scores are never opened here);
- tokens: each tokenizer re-creates the frozen token IDs; every offered letter is one token continuing the prompt; the
  scored position is the last token, which closes the open assistant turn.

An input-integrity check, not an accuracy experiment. Exit 0 if every check passes; 1 if any fails (published as a finding).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, issue, runtime,
                                             sha256, strict_json)
from ..iref_vla.choices import check_boundary, choices_line
from ..iref_vla.prepare import inside, objects_line_ids, read_bundle, rows_of
from ..iref_vla.protocol import load_protocol
from . import design as D
from .prepare import MODEL_KEYS, code_hashes, load_tokenizers, verify_compare_requests

FORBIDDEN = ("source_target", "target_id", "annotation", "insufficient_information", "\"resolved\"", "no_match",
             "agrees_with_source")


def _fail(code, problems):
    raise EvaluationInputError([issue(w, code, m) for w, m in problems])


def audit_parents(rows) -> list:
    """The parents of the canary requests, chosen exactly as run.py chose them: the first request, then each model's
    longest request (ties to the later request index, as `max` over (tokens, index) gives)."""
    picks = [rows[0]["parent_command_id"]]
    for key in MODEL_KEYS:
        longest = max(rows, key=lambda r: (r["models"][key]["input_tokens"], r["request_index"]))
        if longest["parent_command_id"] not in picks:
            picks.append(longest["parent_command_id"])
    return picks


def _token_offset(tok, prompt: str, ids, k: int):
    """The token index at character offset k, if k is a token boundary (the prefix's tokens are a prefix of ids)."""
    pre = tok.encode(prompt[:k])
    return len(pre) if ids[:len(pre)] == pre else None


def audit_request(r, prompt, frozen, toks, proto, doc_bytes, command, policy) -> dict:
    w = proto["wrapper"]
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    mapping = [m[:2] for m in r["mapping"]]
    document = doc_bytes.decode("utf-8")
    user = choices_line(mapping) + document
    rebuilt = w["before_system"] + proto["system_message"] + w["between"] + user + w["after_user"]
    check("the prompt is exactly the protocol's system turn, the user turn and an open assistant turn", prompt == rebuilt)
    check("the system turn is the protocol's message, verbatim",
          prompt.startswith(w["before_system"] + proto["system_message"] + w["between"]))
    spans = {"system_header": [0, len(w["before_system"])],
             "system_message": [len(w["before_system"]), len(w["before_system"]) + len(proto["system_message"])]}
    u0 = spans["system_message"][1] + len(w["between"])
    spans["user_content"] = [u0, u0 + len(user)]
    spans["choices_line"] = [u0, u0 + len(choices_line(mapping))]
    spans["document"] = [spans["choices_line"][1], spans["user_content"][1]]
    spans["assistant_open"] = [spans["user_content"][1], len(prompt)]
    check("the document is the bundle's rendered document, byte for byte", sha256(doc_bytes) == r["source_document_sha256"]
          and prompt[spans["document"][0]:spans["document"][1]] == document)
    check("the document lists exactly the subscene's objects, in scene order", objects_line_ids(document) == r["object_ids"])
    cmd_line = json.dumps({"command": command["text"]}, ensure_ascii=False, separators=(",", ":"))
    check("the document carries the derived command's exact text once, on its own line",
          document.count(cmd_line + "\n") == 1, command["text"][:120])
    letters = [m[0] for m in mapping]
    check("letters A..J for the objects, each object once, and K last meaning ASK",
          mapping[-1] == [proto["ask_code"], proto["ask_target"]] and proto["ask_code"] not in letters[:-1]
          and sorted(m[1] for m in mapping[:-1]) == sorted(r["object_ids"])
          and sorted(letters[:-1]) == list(D.LETTERS[:len(r["object_ids"])])
          and [x for x in letters if x == proto["ask_code"]] == [proto["ask_code"]])
    check("the choices are the frozen policy's letters and list order for this parent and view",
          D.mapping_for(policy, proto, r["parent_command_id"], r["view_id"], r["object_ids"]) == mapping)
    leaks = [s for s in FORBIDDEN if s in prompt] + ([f"{r['parent_command_id']}.a"] if f"{r['parent_command_id']}.a" in prompt else [])
    check("no annotation ID, target field or rules outcome appears in the prompt", not leaks, ", ".join(leaks))
    per_model = {}
    for key in MODEL_KEYS:
        tok, ids = toks[key], frozen[key]
        again = tok.encode(prompt)
        offsets = {name: [_token_offset(tok, prompt, ids, a), _token_offset(tok, prompt, ids, b)] for name, (a, b) in spans.items()}
        offered = [list(x) for x in check_boundary(tok, prompt, mapping, proto)]
        per_model[key] = {"tokens": len(ids), "token_ids_sha256": sha256(",".join(str(i) for i in ids).encode("ascii")),
                          "token_spans": offsets, "scored_position": len(ids) - 1, "offered": offered}
        check(f"{key}: re-tokenizing gives the frozen token IDs", again == ids)
        check(f"{key}: every role boundary is a token boundary",
              all(v is not None for pair in offsets.values() for v in pair) and offsets["assistant_open"][1] == len(ids))
        check(f"{key}: every offered letter is one token continuing the prompt, with the pinned IDs", offered == r["mapping"])
        check(f"{key}: the scored position is the last token, which closes the open assistant turn",
              offsets["assistant_open"][0] is not None and offsets["assistant_open"][0] < len(ids)
              and per_model[key]["scored_position"] == len(ids) - 1 == offsets["assistant_open"][1] - 1)
    return {"request_id": r["request_id"], "parent_command_id": r["parent_command_id"], "view_id": r["view_id"],
            "format": r["format"], "object_count": r["object_count"], "mapping": r["mapping"], "command_text": command["text"],
            "document_sha256": sha256(doc_bytes), "prompt_sha256": sha256(prompt.encode("utf-8")),
            "char_spans": spans, "models": per_model, "checks": checks, "passed": all(c["passed"] for c in checks)}


def render(records) -> str:
    L = ["# A2.3e input audit: the canary parents' requests", "",
         "An input-integrity check, not an accuracy experiment. Every check is listed; exact bytes are in `requests/`.", ""]
    for a in records:
        L += [f"## {a['request_id']} · {a['view_id']} · {a['format']} · {a['object_count']} objects", "",
              f"- Command: `{a['command_text']}`", f"- Document SHA-256 `{a['document_sha256']}`; prompt `{a['prompt_sha256']}`",
              "- Choices: " + ", ".join(f"{c} = {t}" for c, t, _ in a["mapping"])]
        for key, m in a["models"].items():
            sp = m["token_spans"]
            L.append(f"- {key}: {m['tokens']} tokens; system message tokens {sp['system_message']}, choices "
                     f"{sp['choices_line']}, document {sp['document']}, open assistant turn {sp['assistant_open']}; scored "
                     f"position {m['scored_position']}")
        L += ["", "| Check | Passed | Detail |", "|---|---|---|"]
        for c in a["checks"]:
            L.append(f"| {c['check']} | {'yes' if c['passed'] else '**no**'} | {c['detail'].replace('|', '/')} |")
        L.append("")
    return "\n".join(L) + "\n"


def run_audit(*, requests, bundle, tokenizer_small, tokenizer_large, out, tokenizers=None) -> dict:
    out = output.refuse_existing(out)
    req, bundle = Path(requests), Path(bundle)
    bad = verify_compare_requests(req)
    if bad:
        _fail("E_COMPARE_REQUESTS", [(str(req), m) for m in bad[:20]])
    rman = json.loads((req / "manifest.json").read_bytes())
    pol, proto = D.load_policy(req / "policy.json"), load_protocol(req / "protocol.json")
    rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
    prompts = {x["request_id"]: x["prompt"] for x in rows_of("prompts.jsonl", (req / "prompts.jsonl").read_bytes(), "E_COMPARE_REQUESTS")}
    frozen = {k: {x["request_id"]: x["token_ids"] for x in rows_of(f"tokens/{k}.jsonl", (req / "tokens" / f"{k}.jsonl").read_bytes(),
                                                                       "E_COMPARE_REQUESTS")} for k in MODEL_KEYS}
    src = read_bundle(bundle)
    if sha256(src["raw"]["manifest.json"]) != rman["source_bundle"]["manifest_sha256"]:
        _fail("E_COMPARE_BUNDLE", [(str(bundle), "not the bundle the requests were prepared from")])
    toks = tokenizers if tokenizers is not None else load_tokenizers(pol, tokenizer_small, tokenizer_large)
    parents = audit_parents(rows)
    chosen = [r for p in parents for r in rows if r["parent_command_id"] == p]
    records, files = [], {}
    for r in chosen:
        irow = src["index"][(r["parent_command_id"], r["view_id"])]
        command = strict_json("command", inside(bundle, irow["command_path"], "E_COMPARE_BUNDLE").read_bytes(), "E_COMPARE_BUNDLE")
        doc = inside(bundle, r["source_document_path"], "E_COMPARE_BUNDLE").read_bytes()
        a = audit_request(r, prompts[r["request_id"]], {k: frozen[k][r["request_id"]] for k in MODEL_KEYS}, toks, proto, doc,
                          command, pol)
        records.append(a)
        files[f"requests/{r['request_id']}/prompt.txt"] = prompts[r["request_id"]].encode("utf-8")
        for k in MODEL_KEYS:
            files[f"requests/{r['request_id']}/tokens-{k}.json"] = (json.dumps(frozen[k][r["request_id"]]) + "\n").encode("ascii")
        files[f"requests/{r['request_id']}/audit.json"] = encode_json(a)
    summary = {"format_version": 1, "record_type": "iref_compare_input_audit", "policy_id": pol["policy_id"],
               "parents": parents, "requests": [a["request_id"] for a in records], "checks": sum(len(a["checks"]) for a in records),
               "failed": [(a["request_id"], c["check"]) for a in records for c in a["checks"] if not c["passed"]],
               "passed": all(a["passed"] for a in records)}
    files["audit.json"] = encode_json(summary)
    files["report.md"] = render(records).encode("utf-8")
    manifest = {"format_version": 1, "record_type": "iref_compare_input_audit_manifest", "policy_id": pol["policy_id"],
                "requests_manifest_sha256": sha256((req / "manifest.json").read_bytes()),
                "bundle_manifest_sha256": sha256(src["raw"]["manifest.json"]),
                "tokenizers": {k: toks[k].identity for k in MODEL_KEYS}, "files": {k: sha256(v) for k, v in sorted(files.items())},
                "code": code_hashes(), "runtime": runtime(),
                "notes": ["Annotations, rules results, model results and scores were not opened."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for rel, data in files.items():
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            output._write_file(staging / rel, data)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary
