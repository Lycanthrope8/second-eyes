"""A2.5 replay inputs (D98, D99): the record the headset reads, its hash rules and the ordered input checks.

Pure functions only: no files, no model. One replay request is one JSON line in the bundle's `headset/` folder. Its fields
are flat (strings, integers, lists of strings or integers), so the headset's JSON reader parses it without dictionaries or
nested lists:

- `prompt_b64`, `prompt_bytes`, `prompt_sha256`: the prompt's exact UTF-8 bytes in base64, their count and SHA-256. The
  prompt reaches the tokenizer byte for byte, never through a string conversion.
- `token_ids`, `input_tokens`, `token_ids_sha256`: the frozen token IDs; the hash is SHA-256 of the ASCII decimal IDs
  joined by commas, as in A2.3d.
- `codes`, `targets`, `code_token_ids`, `mapping_sha256`: the choices in prompt order as three parallel lists, K and ASK
  last; the hash is SHA-256 of the UTF-8 lines "code TAB target TAB token ID LF", one per choice, in order.
- `keep_before_last_token`, `keep_before_command_line`: the two cache boundaries as the number of leading tokens to keep;
  -1 in written fixtures, which never reach evaluation.
- `kind` (`dataset_command` or `written_integration_fixture`, D99(5)), `expected_outcome`, `expected_reason`.

`check_inputs` applies the input checks in a fixed order and reports the first failure, so each written fixture fails at
exactly one intended check. The headset's replay mode must apply the same checks in the same order.
"""
from __future__ import annotations

import base64
import binascii
import hashlib

KINDS = ("dataset_command", "written_integration_fixture")
CHECKS = ("prompt_bytes", "mapping", "token_ids", "tokenization", "context")
FAILS_AS = {"prompt_bytes": "invalid_input", "mapping": "invalid_input", "token_ids": "invalid_input",
            "tokenization": "tokenization_mismatch", "context": "context_budget_exceeded"}
REASONS = {"prompt_bytes": ("prompt_bytes_invalid", "prompt_hash_mismatch"),
           "mapping": ("mapping_invalid", "code_token_mismatch"),
           "token_ids": ("token_ids_invalid", "token_out_of_vocabulary"),
           "tokenization": ("tokenization_mismatch",),
           "context": ("context_budget_exceeded",)}
PASSED = "passed"
NOT_EVALUATED = -1   # the boundaries of written fixtures, which stop before evaluation
FIELDS = ("code_token_ids", "codes", "expected_outcome", "expected_reason", "format_version", "input_tokens",
          "keep_before_command_line", "keep_before_last_token", "kind", "mapping_sha256", "prompt_b64", "prompt_bytes",
          "prompt_sha256", "record_type", "request_id", "targets", "token_ids", "token_ids_sha256")
RECORD_TYPE = "a25_replay_request"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_ids_sha256(ids) -> str:
    """A2.3d's rule: SHA-256 of the decimal IDs joined by commas, in ASCII."""
    return sha256_hex(",".join(str(i) for i in ids).encode("ascii"))


def mapping_sha256(codes, targets, token_ids) -> str:
    """SHA-256 of the UTF-8 lines "code TAB target TAB token ID LF", one per choice, in prompt order."""
    return sha256_hex("".join(f"{c}\t{t}\t{i}\n" for c, t, i in zip(codes, targets, token_ids)).encode("utf-8"))


def make_record(*, request_id: str, kind: str, prompt: bytes, token_ids, mapping, keep_before_last_token: int,
                keep_before_command_line: int, expected_outcome: str, expected_reason=None) -> dict:
    """A consistent record; mapping is [[code, target, token ID], ...] in prompt order."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    codes, targets, ids = [m[0] for m in mapping], [m[1] for m in mapping], [m[2] for m in mapping]
    return {"format_version": 1, "record_type": RECORD_TYPE, "request_id": request_id, "kind": kind,
            "prompt_b64": base64.b64encode(prompt).decode("ascii"), "prompt_bytes": len(prompt),
            "prompt_sha256": sha256_hex(prompt), "token_ids": list(token_ids), "input_tokens": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids), "codes": codes, "targets": targets, "code_token_ids": ids,
            "mapping_sha256": mapping_sha256(codes, targets, ids), "keep_before_last_token": keep_before_last_token,
            "keep_before_command_line": keep_before_command_line, "expected_outcome": expected_outcome,
            "expected_reason": expected_reason}


def decode_prompt(record) -> bytes:
    """The prompt's bytes; ValueError if prompt_b64 is not strict base64 of UTF-8 text."""
    text = record.get("prompt_b64")
    if not isinstance(text, str):
        raise ValueError("prompt_b64 is not a string")
    try:
        data = base64.b64decode(text.encode("ascii"), validate=True)
        data.decode("utf-8")
    except (binascii.Error, UnicodeError) as e:
        raise ValueError(f"prompt_b64 does not decode to UTF-8 bytes: {e}") from None
    return data


def _plain_int(x) -> bool:
    return type(x) is int


def _result(check=None, reason=None, detail="", skipped=()):
    return {"outcome": PASSED if check is None else FAILS_AS[check], "check": check, "reason": reason, "detail": detail,
            "skipped": list(skipped)}


def check_inputs(record: dict, constants: dict, tokenize=None) -> dict:
    """The input checks in order; the first failure decides the outcome. constants: context_limit_tokens,
    continuation_tokens, vocab_size, object_codes, ask_code, ask_target and code_token_ids ({code: token ID}).
    tokenize: text -> token IDs, the runtime's own tokenizer; None skips the tokenization check (reported in
    `skipped`). Returns {"outcome", "check", "reason", "detail", "skipped"}; outcome "passed" means evaluation may run."""
    # 1. prompt bytes
    try:
        data = decode_prompt(record)
    except ValueError as e:
        return _result("prompt_bytes", "prompt_bytes_invalid", str(e))
    if not _plain_int(record.get("prompt_bytes")) or len(data) != record["prompt_bytes"]:
        return _result("prompt_bytes", "prompt_bytes_invalid", f"{len(data)} bytes, recorded {record.get('prompt_bytes')!r}")
    if sha256_hex(data) != record.get("prompt_sha256"):
        return _result("prompt_bytes", "prompt_hash_mismatch", "the bytes' SHA-256 differs from prompt_sha256")
    # 2. mapping: structure, K and ASK last, hash, then each code's token ID
    codes, targets, ids = record.get("codes"), record.get("targets"), record.get("code_token_ids")
    objects, ask, ask_target = constants["object_codes"], constants["ask_code"], constants["ask_target"]
    if not (isinstance(codes, list) and isinstance(targets, list) and isinstance(ids, list)
            and len(codes) == len(targets) == len(ids) and 2 <= len(codes) <= len(objects) + 1):
        return _result("mapping", "mapping_invalid", "codes, targets and code_token_ids must be parallel lists of 2 to "
                                                     f"{len(objects) + 1} entries")
    if codes[-1] != ask or targets[-1] != ask_target:
        return _result("mapping", "mapping_invalid", f"the last choice must be {ask} for {ask_target}")
    head_codes, head_targets = codes[:-1], targets[:-1]
    if any(not isinstance(c, str) or c not in objects for c in head_codes) or len(set(head_codes)) != len(head_codes):
        return _result("mapping", "mapping_invalid", "object codes must be distinct letters of the protocol's object codes")
    if any(not isinstance(t, str) or not t or t == ask_target for t in head_targets) \
            or len(set(head_targets)) != len(head_targets):
        return _result("mapping", "mapping_invalid", "object targets must be distinct, non-empty and not the ASK target")
    if any(not _plain_int(i) for i in ids):
        return _result("mapping", "mapping_invalid", "code token IDs must be plain integers")
    if mapping_sha256(codes, targets, ids) != record.get("mapping_sha256"):
        return _result("mapping", "mapping_invalid", "the mapping's SHA-256 differs from mapping_sha256")
    wrong = [f"{c}: {i} (protocol {constants['code_token_ids'][c]})" for c, i in zip(codes, ids)
             if i != constants["code_token_ids"][c]]
    if wrong:
        return _result("mapping", "code_token_mismatch", "; ".join(wrong))
    # 3. token IDs: plain integers, count and hash, then the vocabulary
    toks = record.get("token_ids")
    if not isinstance(toks, list) or not toks or any(not _plain_int(i) for i in toks):
        return _result("token_ids", "token_ids_invalid", "token_ids must be a non-empty list of plain integers")
    if record.get("input_tokens") != len(toks) or not _plain_int(record.get("input_tokens")):
        return _result("token_ids", "token_ids_invalid", f"{len(toks)} token IDs, recorded {record.get('input_tokens')!r}")
    if token_ids_sha256(toks) != record.get("token_ids_sha256"):
        return _result("token_ids", "token_ids_invalid", "the IDs' SHA-256 differs from token_ids_sha256")
    outside = [(k, i) for k, i in enumerate(toks) if i < 0 or i >= constants["vocab_size"]]
    if outside:
        return _result("token_ids", "token_out_of_vocabulary",
                       f"{len(outside)} ID(s) outside 0..{constants['vocab_size'] - 1}, first at position {outside[0][0]}: "
                       f"{outside[0][1]}")
    # 4. the runtime's own tokenization equals the frozen IDs
    skipped = []
    if tokenize is None:
        skipped.append("tokenization")
    else:
        got = list(tokenize(data.decode("utf-8")))
        if got != toks:
            first = next((k for k, (a, b) in enumerate(zip(got, toks)) if a != b), min(len(got), len(toks)))
            return _result("tokenization", "tokenization_mismatch",
                           f"{len(got)} tokens against {len(toks)} frozen; first difference at position {first}")
    # 5. the context ceiling, never truncated: the prompt plus its one scored continuation must fit
    need = len(toks) + constants["continuation_tokens"]
    if need > constants["context_limit_tokens"]:
        return _result("context", "context_budget_exceeded",
                       f"{len(toks)} tokens + {constants['continuation_tokens']} > {constants['context_limit_tokens']}",
                       skipped)
    return _result(skipped=skipped)
