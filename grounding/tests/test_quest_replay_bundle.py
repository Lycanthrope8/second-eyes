"""A2.5 delivery 1, step 1 (D98, D99): the replay bundle, its records, input checks, boundaries, selection and readback.

Run directly (python grounding/tests/test_quest_replay_bundle.py) or as a module. Fixtures are A2.3b's labelled fixture
chain with labelled tokenizer and model doubles; expectations are literal or recomputed independently with hashlib and
string arithmetic, never read from the code under test. With --tokenizer-dir DIR the pinned tokenizer also checks both
boundaries on prompts built from the serializer's stored goldens, and the written fixtures under the real context limit:
not acceptance, since no A2.3d artifact is read.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
KEYS = ("qwen2.5-0.5b-instruct", "qwen2.5-7b-instruct")
CASE_SALT = "second-eyes/a23e/cases/v1"


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def outcome(fn):
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
    try:
        return ("ok", fn())
    except EI as e:
        return ("input_error", sorted({i["code"] for i in e.issues}))
    except Exception as e:  # noqa: BLE001
        return ("crashed", f"{type(e).__name__}: {e}")


def refusal(fn):
    """(codes, all messages joined) of an input error; None if fn did not raise one."""
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
    try:
        fn()
    except EI as e:
        return sorted({i["code"] for i in e.issues}), " | ".join(i["message"] for i in e.issues)
    except Exception as e:  # noqa: BLE001
        return ["crashed"], f"{type(e).__name__}: {e}"
    return None


def attempt(fn):
    """fn's result, or the exception it raised, so a check reports it instead of stopping the suite."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"raised {type(e).__name__}: {e}"


def hs(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hb(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines()]


def write_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")


# ------------------------------------------------------------------------------------------------ 1. input checks

CONST = {"context_limit_tokens": 10, "continuation_tokens": 1, "vocab_size": 1000, "object_codes": list("ABCDEFGHIJ"),
         "ask_code": "K", "ask_target": "ASK", "code_token_ids": {c: 32 + k for k, c in enumerate("ABCDEFGHIJK")}}


def hand_record(**changes):
    """Every value written out by hand: 'hand prompt' is 11 bytes; the hashes are of literal strings."""
    prompt = b"hand prompt"
    rec = {"format_version": 1, "record_type": "a25_replay_request", "request_id": "hand", "kind": "dataset_command",
           "prompt_b64": base64.b64encode(prompt).decode("ascii"), "prompt_bytes": 11, "prompt_sha256": hb(prompt),
           "token_ids": [5, 6, 7], "input_tokens": 3, "token_ids_sha256": hb(b"5,6,7"),
           "codes": ["B", "A", "K"], "targets": ["o2", "o1", "ASK"], "code_token_ids": [33, 32, 42],
           "mapping_sha256": hb(b"B\to2\t33\nA\to1\t32\nK\tASK\t42\n"), "keep_before_last_token": 2,
           "keep_before_command_line": 1, "expected_outcome": "completed", "expected_reason": None}
    rec.update(changes)
    return rec


def remap(codes, targets, ids):
    return {"codes": codes, "targets": targets, "code_token_ids": ids,
            "mapping_sha256": hb("".join(f"{c}\t{t}\t{i}\n" for c, t, i in zip(codes, targets, ids)).encode())}


def retoken(ids):
    return {"token_ids": ids, "input_tokens": len(ids), "token_ids_sha256": hb(",".join(map(str, ids)).encode())}


def check_inputs_unit(RI):
    print("-- the record and its ordered input checks (hand-written records)")
    made = RI.make_record(request_id="hand", kind="dataset_command", prompt=b"hand prompt", token_ids=[5, 6, 7],
                          mapping=[["B", "o2", 33], ["A", "o1", 32], ["K", "ASK", 42]], keep_before_last_token=2,
                          keep_before_command_line=1, expected_outcome="completed")
    check("make_record gives the hand-written record: base64 bytes, A2.3d's token hash, the TAB/LF mapping hash",
          made == hand_record(), str(set(made.items()) ^ set((k, v if not isinstance(v, list) else tuple(v))
                                                              for k, v in hand_record().items()))[:200] if made != hand_record() else "")
    check("the record has exactly the documented fields", sorted(made) == sorted(RI.FIELDS))
    r = RI.check_inputs(hand_record(), CONST)
    check("a valid record passes; without a tokenizer the tokenization check is reported as skipped",
          (r["outcome"], r["check"], r["skipped"]) == ("passed", None, ["tokenization"]), str(r))
    r = RI.check_inputs(hand_record(), CONST, tokenize=lambda t: [5, 6, 7] if t == "hand prompt" else [])
    check("with the runtime's tokenizer reproducing the IDs it passes, nothing skipped",
          (r["outcome"], r["skipped"]) == ("passed", []), str(r))
    utf = base64.b64encode(b"\xff\xfeprompt___").decode()
    cases = [
        ("prompt_b64 that is not base64", dict(prompt_b64="@@@"), ("invalid_input", "prompt_bytes", "prompt_bytes_invalid")),
        ("bytes that are not UTF-8", dict(prompt_b64=utf, prompt_sha256=hb(b"\xff\xfeprompt___")),
         ("invalid_input", "prompt_bytes", "prompt_bytes_invalid")),
        ("a byte count that differs", dict(prompt_bytes=12), ("invalid_input", "prompt_bytes", "prompt_bytes_invalid")),
        ("same length, one byte changed, hash kept", dict(prompt_b64=base64.b64encode(b"Hand prompt").decode()),
         ("invalid_input", "prompt_bytes", "prompt_hash_mismatch")),
        ("K/ASK first (hash recomputed)", remap(["K", "B", "A"], ["ASK", "o2", "o1"], [42, 33, 32]),
         ("invalid_input", "mapping", "mapping_invalid")),
        ("K last but not targeting ASK", remap(["B", "A", "K"], ["o2", "o1", "o3"], [33, 32, 42]),
         ("invalid_input", "mapping", "mapping_invalid")),
        ("a repeated letter", remap(["A", "A", "K"], ["o2", "o1", "ASK"], [32, 32, 42]),
         ("invalid_input", "mapping", "mapping_invalid")),
        ("an object targeted as ASK", remap(["B", "A", "K"], ["ASK", "o1", "ASK"], [33, 32, 42]),
         ("invalid_input", "mapping", "mapping_invalid")),
        ("a repeated target", remap(["B", "A", "K"], ["o1", "o1", "ASK"], [33, 32, 42]),
         ("invalid_input", "mapping", "mapping_invalid")),
        ("a letter outside A..J", remap(["Z", "A", "K"], ["o2", "o1", "ASK"], [33, 32, 42]),
         ("invalid_input", "mapping", "mapping_invalid")),
        ("K alone", remap(["K"], ["ASK"], [42]), ("invalid_input", "mapping", "mapping_invalid")),
        ("twelve choices", remap(list("ABCDEFGHIJ") + ["A", "K"], [f"o{i}" for i in range(11)] + ["ASK"], [32] * 12),
         ("invalid_input", "mapping", "mapping_invalid")),
        ("a wrong mapping hash", dict(mapping_sha256="0" * 64), ("invalid_input", "mapping", "mapping_invalid")),
        ("A carrying B's token ID (hash recomputed)", remap(["B", "A", "K"], ["o2", "o1", "ASK"], [33, 33, 42]),
         ("invalid_input", "mapping", "code_token_mismatch")),
        ("K carrying J's token ID (hash recomputed)", remap(["B", "A", "K"], ["o2", "o1", "ASK"], [33, 32, 41]),
         ("invalid_input", "mapping", "code_token_mismatch")),
        ("a wrong token hash", dict(token_ids_sha256="0" * 64), ("invalid_input", "token_ids", "token_ids_invalid")),
        ("a token count that differs", dict(input_tokens=4), ("invalid_input", "token_ids", "token_ids_invalid")),
        ("a whole-valued float token ID", dict(token_ids=[5, 6.0, 7]), ("invalid_input", "token_ids", "token_ids_invalid")),
        ("a boolean token ID", dict(token_ids=[5, True, 7]), ("invalid_input", "token_ids", "token_ids_invalid")),
        ("no token IDs", retoken([]), ("invalid_input", "token_ids", "token_ids_invalid")),
        ("an ID equal to the vocabulary size (hash recomputed)", retoken([5, 1000, 7]),
         ("invalid_input", "token_ids", "token_out_of_vocabulary")),
        ("a negative ID (hash recomputed)", retoken([5, -1, 7]), ("invalid_input", "token_ids", "token_out_of_vocabulary")),
        ("10 tokens + 1 > 10", retoken(list(range(1, 11))), ("context_budget_exceeded", "context", "context_budget_exceeded")),
        ("two defects: the earlier check decides (prompt before tokens)",
         dict(prompt_b64=base64.b64encode(b"Hand prompt").decode(), token_ids_sha256="0" * 64),
         ("invalid_input", "prompt_bytes", "prompt_hash_mismatch")),
        ("two defects: mapping before an out-of-vocabulary ID",
         {**remap(["K", "B", "A"], ["ASK", "o2", "o1"], [42, 33, 32]), **retoken([5, 1000, 7])},
         ("invalid_input", "mapping", "mapping_invalid")),
    ]
    for label, change, want in cases:
        r = RI.check_inputs(hand_record(**change), CONST)
        check(f"{label}: {want[0]} at {want[1]} ({want[2]})", (r["outcome"], r["check"], r["reason"]) == want, str(r))
    r = RI.check_inputs(hand_record(**retoken(list(range(1, 10)))), CONST)
    check("9 tokens + 1 = 10 fits a limit of 10 (equality passes)", r["outcome"] == "passed", str(r))
    r = RI.check_inputs(hand_record(), CONST, tokenize=lambda t: [5, 6, 8])
    check("a runtime tokenization that differs: tokenization_mismatch, with the first differing position",
          (r["outcome"], r["check"], r["reason"]) == ("tokenization_mismatch", "tokenization", "tokenization_mismatch")
          and "position 2" in r["detail"], str(r))
    r = RI.check_inputs(hand_record(**retoken(list(range(1, 11)))), CONST, tokenize=lambda t: [9])
    check("the tokenization check comes before the context check", r["check"] == "tokenization", str(r))
    check("the check order is the policy's", RI.CHECKS == ("prompt_bytes", "mapping", "token_ids", "tokenization", "context"))


# ------------------------------------------------------------------------------------------------ 2. boundaries

class TailMergingTokenizer:
    """Labelled double: one token per character, except a final 't' + LF, which is one token."""
    identity, kind = "test_double.tail_merging", "test_double"

    def encode(self, text):
        ids = [ord(c) - 33 if 33 <= ord(c) < 127 else 200000 + ord(c) for c in text]
        return ids[:-2] + [777777] if text.endswith("t\n") else ids


class LfBraceMergingTokenizer:
    """Labelled double: one token per character, except LF followed by '{', which is one token."""
    identity, kind = "test_double.lf_brace", "test_double"

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            if text[i] == "\n" and i + 1 < len(text) and text[i + 1] == "{":
                out.append(888888)
                i += 2
            else:
                c = text[i]
                out.append(ord(c) - 33 if 33 <= ord(c) < 127 else 200000 + ord(c))
                i += 1
        return out


def hand_prompt(proto, doc):
    """The prompt written out from the protocol's fields, with the choices line typed by hand."""
    w = proto["wrapper"]
    head = w["before_system"] + proto["system_message"] + w["between"] + '{"choices":[["B","o2"],["A","o1"],["K","ASK"]]}\n'
    return head, head + doc + w["after_user"]


def check_boundaries(RB):
    print("-- cache boundaries (labelled tokenizer doubles)")
    A = importlib.import_module("grounding.tests.test_iref_vla_pilot")
    proto = importlib.import_module("grounding.inference.iref_vla.protocol").load_protocol()
    mapping = [["B", "o2", 33], ["A", "o1", 32], ["K", "ASK", 42]]
    lines = '{"objects":[]}\n{"pose":{"pose_kind":"none"}}\n'
    head, prompt = hand_prompt(proto, lines + '{"command":"Inspect it."}\n')
    tok = A.OffsetCharTokenizer()
    ids = tok.encode(prompt)
    last = attempt(lambda: RB.boundary_before_last_token(tok, prompt, ids))
    check("one token per character: keep n - 1 tokens, the text before the last character",
          last == {"keep_tokens": len(prompt) - 1, "prefix_chars": len(prompt) - 1}, str(last))
    cmd = attempt(lambda: RB.boundary_before_command_line(tok, prompt, ids, proto, mapping))
    want = len(head) + len(lines)
    check("the command line starts after the head and the two earlier document lines", cmd == {"keep_tokens": want,
                                                                                               "prefix_chars": want}, str(cmd))
    tt = TailMergingTokenizer()
    tids = tt.encode(prompt)
    last = attempt(lambda: RB.boundary_before_last_token(tt, prompt, tids))
    check("a last token of two characters: keep n - 1 tokens, the text before both characters",
          last == {"keep_tokens": len(tids) - 1, "prefix_chars": len(prompt) - 2} and len(tids) == len(prompt) - 1, str(last))
    lb = LfBraceMergingTokenizer()
    try:
        RB.boundary_before_command_line(lb, prompt, lb.encode(prompt), proto, mapping)
        check("a command line that starts inside a token is refused", False)
    except ValueError as e:
        check("a command line that starts inside a token is refused", "token boundary" in str(e), str(e))
    _, no_cmd = hand_prompt(proto, lines)
    try:
        RB.boundary_before_command_line(tok, no_cmd, tok.encode(no_cmd), proto, mapping)
        check("a document whose last line is not the command line is refused", False)
    except ValueError as e:
        check("a document whose last line is not the command line is refused", "command line" in str(e), str(e))
    try:
        RB.boundary_before_command_line(tok, prompt[:-1], tok.encode(prompt[:-1]), proto, mapping)
        check("a prompt without the open assistant turn is refused", False)
    except ValueError:
        check("a prompt without the open assistant turn is refused", True)
    try:
        RB.boundary_before_last_token(tok, "x", [1])
        check("a one-token prompt has no last-token boundary", False)
    except ValueError:
        check("a one-token prompt has no last-token boundary", True)


# ------------------------------------------------------------------------------------------------ 3. selection

def synthetic_rows():
    """30 parents x 2 views x 2 formats in A2.3d's order. Tokens are hash-derived below 4,000, except: r0005 and r0057
    (both full inventory, coordinates) tie at 9,999, and r0094 (parent p23, full inventory, augmented) is the longest of
    all at 12,000 for both models."""
    rows, k = [], 0
    for p in [f"p{i:02d}" for i in range(30)]:
        for v in ("full_inventory", "source_known_nyu"):
            for f in ("coordinates_v2", "coordinates_relations_v2"):
                k += 1
                t = 1000 + int(hs(f"tok{k}")[:4], 16) % 3000
                t = {5: 9999, 57: 9999, 94: 12000}.get(k, t)
                rows.append({"request_index": k, "request_id": f"r{k:04d}", "parent_command_id": p, "view_id": v,
                             "format": f, "models": {m: {"input_tokens": t} for m in KEYS}})
    return rows


def check_selection(RB):
    print("-- selection (synthetic request index; expectations recomputed with hashlib)")
    rows = synthetic_rows()
    pairs = list(dict.fromkeys((r["parent_command_id"], r["view_id"]) for r in rows))
    cases = sorted(pairs, key=lambda pv: (hs(CASE_SALT + "\n" + pv[0] + "\n" + pv[1]), pv))[:16]
    case_ids = [r["request_id"] for pv in cases for r in rows if (r["parent_command_id"], r["view_id"]) == pv]
    audit_ids = [f"r{k:04d}" for k in (1, 2, 3, 4, 93, 94, 95, 96)]
    longest = {"coordinates_v2": "r0057", "coordinates_relations_v2": "r0094"}
    union = sorted(set(case_ids) | set(audit_ids) | set(longest.values()))
    sel = RB.select_replay(rows, KEYS[0])
    check("the 16 cases are the first 16 pairs by SHA-256(salt + LF + parent + LF + view), each in both formats",
          sel["a23e_cases"] == [list(pv) for pv in cases] and sel["a23e_case_request_ids"] == case_ids and len(case_ids) == 32)
    check("the audit requests are the first request's parent and the longest request's parent, all four each",
          sel["a23e_audit_parents"] == ["p00", "p23"] and sel["a23e_audit_request_ids"] == audit_ids, str(sel["a23e_audit_request_ids"]))
    check("longest per format: a tie at the top goes to the later request (r0057 over r0005); r0094 for augmented",
          {f: x["request_id"] for f, x in sel["longest_in_format"].items()} == longest, str(sel["longest_in_format"]))
    check("the list is the union in request order, each ID once, with its list hash",
          sel["request_ids"] == union and sel["count"] == len(union) and sel["list_sha256"] == hs("".join(i + "\n" for i in union)))
    want_src = {i: [t for t, s in (("a23e_case", case_ids), ("a23e_audit", audit_ids), ("longest_in_format", longest.values()))
                    if i in s] for i in union}
    check("every request lists each source that chose it, in the policy's source order", sel["sources"] == want_src)
    check("requests chosen more than once are listed", sel["chosen_by_several"] == [i for i in union if len(want_src[i]) > 1])
    pol = {"selection": {"expected": {"a23e_case_requests": 32, "a23e_audit_request_ids": audit_ids}}}
    check("the expectations hold for this index", RB.selection_problems(sel, pol) == [])
    rows3 = synthetic_rows()
    rows3[41 - 1]["models"][KEYS[1]]["input_tokens"] = 13000   # r0041: parent p10, now the 7B's longest only
    sel3 = RB.select_replay(rows3, KEYS[0])
    want3 = [f"r{k:04d}" for k in (1, 2, 3, 4, 93, 94, 95, 96, 41, 42, 43, 44)]
    check("three audit parents keep the audit's order (first request, 0.5B's longest, 7B's longest), not request order",
          sel3["a23e_audit_parents"] == ["p00", "p23", "p10"] and sel3["a23e_audit_request_ids"] == want3,
          str(sel3["a23e_audit_request_ids"]))
    pol2 = copy.deepcopy(pol)
    pol2["selection"]["expected"]["a23e_audit_request_ids"] = audit_ids[:4] + ["r0329", "r0330", "r0331", "r0332"]
    check("other expected audit requests are reported", len(RB.selection_problems(sel, pol2)) == 1)


# ------------------------------------------------------------------------------------------------ 4. references

def reference_row(mapping, logits):
    """A run row written by hand from logits, as A2.3a's score() defines it (exact ties go to K)."""
    import math
    top = max(logits)
    e = [math.exp(x - top) for x in logits]
    z = math.fsum(e)
    shares = [x / z for x in e]
    tied = [m[0] for m, x in zip(mapping, logits) if x == top]
    code = "K" if len(tied) > 1 else tied[0]
    r = sorted(logits, reverse=True)
    return {"request_id": "r0001", "prompt_sha256": "p" * 64, "token_ids_sha256": "t" * 64, "mapping": mapping,
            "technical_status": "completed", "choice_code": code,
            "selection_reason": "exact_score_tie" if len(tied) > 1 else "max_offered_logit",
            "tied_codes": tied if len(tied) > 1 else [], "choice_object_id": None if code == "K" else
            next(m[1] for m in mapping if m[0] == code), "model_choice": "model_choice_ask" if code == "K" else "model_choice_object",
            "scores": [{"code": m[0], "target": m[1], "token_id": m[2], "logit": x, "log_prob": x - 20.0,
                        "restricted_share": s} for m, x, s in zip(mapping, logits, shares)],
            "top_restricted_share": max(shares), "logit_margin": r[0] - r[1]}


def check_references(RB):
    print("-- reusable float32 references (hand-written rows)")
    m = [["B", "o2", 33], ["A", "o1", 32], ["K", "ASK", 42]]
    ok = dict(request_id="r0001", prompt_sha256="p" * 64, token_ids_sha256="t" * 64, mapping=m)
    row = reference_row(m, [2.5, 1.0, -0.5])
    check("a consistent completed row is reusable", RB.reference_problems(row, **ok) == [])
    tie = reference_row(m, [2.0, 2.0, 1.0])
    check("an exact tie at the top chooses K with the tied codes listed", tie["choice_code"] == "K"
          and RB.reference_problems(tie, **ok) == [])
    bad_rows = [("a share off by 1e-9", lambda r: r["scores"][1].__setitem__("restricted_share", r["scores"][1]["restricted_share"] + 1e-9)),
                ("another choice than the logits give", lambda r: r.update(choice_code="A", choice_object_id="o1")),
                ("a logit changed so the choice differs", lambda r: r["scores"][1].__setitem__("logit", 9.0)),
                ("a tie resolved to an object", lambda r: None),
                ("a status other than completed", lambda r: r.update(technical_status="execution_failed")),
                ("another prompt hash", lambda r: r.update(prompt_sha256="q" * 64)),
                ("a mapping in another order", lambda r: r.update(mapping=[m[1], m[0], m[2]])),
                ("a wrong margin", lambda r: r.update(logit_margin=1.0)),
                ("a non-finite logit", lambda r: r["scores"][0].__setitem__("logit", float("inf")))]
    for label, edit in bad_rows:
        r = copy.deepcopy(tie if label == "a tie resolved to an object" else row)
        if label == "a tie resolved to an object":
            r.update(choice_code="B", selection_reason="max_offered_logit", tied_codes=[], choice_object_id="o2",
                     model_choice="model_choice_object")
        else:
            edit(r)
        check(f"not reusable: {label}", RB.reference_problems(r, **ok) != [])


# ------------------------------------------------------------------------------------------------ 5. end to end

class ShiftedTokenizer:
    """Labelled double: the offset-character tokenizer's identity, but the last ID shifted by one."""
    identity, kind = "test_double.offset_char", "test_double"

    def encode(self, text):
        ids = [ord(c) - 33 if 33 <= ord(c) < 127 else 200000 + ord(c) for c in text]
        return ids[:-1] + [ids[-1] + 1] if ids else ids


def fixture_chain(tmp, D, P, RUN):
    """A2.3b's fixture chain: A2.3d requests (tokenizer doubles) and fake 0.5B and 7B runs, as test_iref_vla_compare."""
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    A = SB.pilot_helpers()
    PI = importlib.import_module("grounding.inference.iref_vla.prepare")
    ev = lambda *a: {"revision": "fixture", "tie": "labelled fixture", "files": {}}  # noqa: E731
    bundle = SB.scoring_bundle(tmp, A)
    ann = SB.annotation_bundle(tmp / "annotations.json", SB.TARGETS)
    eligible = PI.eligible_parents(PI.read_bundle(bundle))
    pilot = sorted(eligible, key=lambda p: (hs("fixture-pilot\n" + p), p))[:1]
    pol = json.loads(D.POLICY_PATH.read_text(encoding="utf-8"))
    pol["population"] = {"rule": "fixture", "expected_eligible": len(eligible), "expected_remaining": len(eligible) - 1,
                         "excluded": {"what": "fixture", "salt": "fixture-pilot\n", "count": 1,
                                      "selected_sha256": hashlib.sha256((pilot[0] + "\n").encode()).hexdigest()}}
    pol["selection"]["count"] = len(eligible) - 1
    pol["expected"] = {"parents": len(eligible) - 1, "parent_views": 2 * (len(eligible) - 1), "requests": 4 * (len(eligible) - 1)}
    pp = tmp / "compare-policy.json"
    pp.write_text(json.dumps(pol), encoding="utf-8")
    P.prepare_compare(bundle=bundle, tokenizer_small=None, tokenizer_large=None, out=tmp / "req", policy=pp,
                      tokenizers={k: A.OffsetCharTokenizer() for k in KEYS})
    common = dict(requests=tmp / "req", model_dir=tmp, tokenizer_dir=None, device="cpu", evidence_fn=ev)
    for key in KEYS:
        RUN.smoke(**common, model_key=key, out=tmp / f"smoke-{key}.json", model_loader=A.loader_for(A.FakeModel()),
                  tokenizer=A.OffsetCharTokenizer())
        RUN.run_compare(**common, model_key=key, smoke_record=tmp / f"smoke-{key}.json", out=tmp / f"run-{key}",
                        model_loader=A.loader_for(A.FakeModel()), tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None)
    return A, SB, ann


def scores_folder(path, req, small, large):
    """A labelled stand-in for the A2.3d scores folder: its manifest links the requests and both runs; decoy files
    stand for the scores, which the builder must never open."""
    path.mkdir()
    man = {"format_version": 1, "record_type": "iref_compare_score_manifest",
           "inputs": {"requests_manifest_sha256": hb((req / "manifest.json").read_bytes()),
                      "small_run_manifest_sha256": hb((small / "manifest.json").read_bytes()),
                      "large_run_manifest_sha256": hb((large / "manifest.json").read_bytes())}}
    (path / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    for n in ("scores.jsonl", "summary.json", "report.md"):
        (path / n).write_text("decoy: never read\n", encoding="utf-8")
    return hb((path / "manifest.json").read_bytes())


def replay_policy(RB, tmp, name, req, run, scores_sha, rows, **edits):
    """The tracked policy with the fixture chain's pins, expectations recomputed independently, a vocabulary large enough
    for the offset-character double, and any edits."""
    pol = json.loads(RB.POLICY_PATH.read_text(encoding="utf-8"))
    pol["inputs"]["requests_manifest_sha256"] = hb((req / "manifest.json").read_bytes())
    pol["inputs"]["scores_manifest_sha256"] = scores_sha
    pol["inputs"]["small_run_manifest_sha256_prefix"] = hb((run / "manifest.json").read_bytes())[:8]
    pairs = list(dict.fromkeys((r["parent_command_id"], r["view_id"]) for r in rows))
    pol["selection"]["expected"]["a23e_case_requests"] = 2 * min(16, len(pairs))
    pol["selection"]["expected"]["a23e_audit_request_ids"] = expected_audit(rows)
    pol["headset"]["vocab_size"] = 2_000_000
    for k, v in edits.items():
        node = pol
        *path, last = k.split(".")
        for x in path:
            node = node[x]
        node[last] = v
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(pol, indent=2), encoding="utf-8")
    return p


def expected_audit(rows):
    parents = [rows[0]["parent_command_id"]]
    for key in KEYS:
        best = max(rows, key=lambda r: (r["models"][key]["input_tokens"], r["request_index"]))["parent_command_id"]
        if best not in parents:
            parents.append(best)
    return [r["request_id"] for p in parents for r in rows if r["parent_command_id"] == p]


def rehash(folder, rels):
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for rel in rels:
        m["files"][rel] = hb((folder / rel).read_bytes())
    (folder / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")


def check_end_to_end(RB, RI, D, P, RUN):
    print("-- build and readback on A2.3b's fixture chain (labelled tokenizer and model doubles)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        A, SB, ann = fixture_chain(tmp, D, P, RUN)
        req, run, run7 = tmp / "req", tmp / f"run-{KEYS[0]}", tmp / f"run-{KEYS[1]}"
        rows = jsonl(req / "request-index.jsonl")
        prompts = {x["request_id"]: x["prompt"] for x in jsonl(req / "prompts.jsonl")}
        frozen = {x["request_id"]: x["token_ids"] for x in jsonl(req / "tokens" / f"{KEYS[0]}.jsonl")}
        results = {x["request_id"]: x for x in jsonl(run / "results.jsonl")}
        sc_sha = scores_folder(tmp / "scores", req, run, run7)
        pol = replay_policy(RB, tmp, "pol", req, run, sc_sha, rows)
        decoys = [tmp / "scores" / n for n in ("scores.jsonl", "summary.json", "report.md")]
        args = dict(requests=req, small_run=run, scores=tmp / "scores", tokenizer_dir=None, policy=pol)
        with SB.Watch([ann] + decoys) as w:
            r = outcome(lambda: RB.build_replay_bundle(**args, out=tmp / "replay", tokenizer=A.OffsetCharTokenizer()))
        check("the build completes with the annotations and the scores' other files never opened",
              r[0] == "ok" and not w.denied, str(r)[:300])
        if r[0] != "ok":
            return
        b = tmp / "replay"
        check("the published bundle reads back on its own", RB.verify_replay_bundle(b) == [], str(RB.verify_replay_bundle(b))[:300])
        full = RB.verify_replay_bundle(b, requests=req, small_run=run, scores=tmp / "scores", tokenizer=A.OffsetCharTokenizer())
        check("and against its sources and the tokenizer", full == [], str(full)[:300])
        check("no staging folder is left beside it", [p.name for p in tmp.iterdir() if p.name.startswith(".replay")] == [])
        recs, fixes = jsonl(b / "headset" / "requests.jsonl"), jsonl(b / "headset" / "fixtures-written.jsonl")
        refs, sel = jsonl(b / "reference" / "references.jsonl"), json.loads((b / "selection.json").read_text(encoding="utf-8"))
        pairs = list(dict.fromkeys((x["parent_command_id"], x["view_id"]) for x in rows))
        cases = sorted(pairs, key=lambda pv: (hs(CASE_SALT + "\n" + pv[0] + "\n" + pv[1]), pv))[:16]
        case_ids = [x["request_id"] for pv in cases for x in rows if (x["parent_command_id"], x["view_id"]) == pv]
        longest = [max((x for x in rows if x["format"] == f), key=lambda x: (x["models"][KEYS[0]]["input_tokens"],
                                                                            x["request_index"]))["request_id"] for f in D.FORMATS]
        union = sorted(set(case_ids) | set(expected_audit(rows)) | set(longest))
        check(f"the list is the independently recomputed union ({len(union)} requests) in request order",
              [x["request_id"] for x in recs] == union and sel["request_ids"] == union)
        check("each headset record carries its prompt byte for byte, its frozen IDs and its mapping",
              all(base64.b64decode(x["prompt_b64"]) == prompts[x["request_id"]].encode("utf-8")
                  and x["token_ids"] == frozen[x["request_id"]]
                  and [[c, t, i] for c, t, i in zip(x["codes"], x["targets"], x["code_token_ids"])] ==
                  next(y["mapping"] for y in rows if y["request_id"] == x["request_id"]) for x in recs))
        check("before the last token: n - 1 tokens kept",
              all(x["keep_before_last_token"] == len(frozen[x["request_id"]]) - 1 for x in recs))
        check("before the command line: the characters up to the last LF-then-command (one token per character)",
              all(x["keep_before_command_line"] == prompts[x["request_id"]].rindex('\n{"command":') + 1 for x in recs))
        check("every reference row is the run's row, unchanged", all(y["reference"] == results[y["request_id"]] for y in refs))
        hm_keys = set(json.loads((b / "headset" / "replay-manifest.json").read_text(encoding="utf-8")))
        check("the headset reads no reference: its records hold only the documented fields, and no file there has a "
              "logit, a score or a choice", all(sorted(x) == sorted(RI.FIELDS) for x in recs + fixes)
              and not hm_keys & {"scores", "logit", "choice_code", "reference"}
              and all('"logit"' not in (b / "headset" / n).read_text(encoding="utf-8") and '"choice_code"' not in (b / "headset" / n).read_text(encoding="utf-8")
                      for n in ("requests.jsonl", "fixtures-written.jsonl", "replay-manifest.json")))
        tok = A.OffsetCharTokenizer()
        hm = json.loads((b / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))
        cons = {"context_limit_tokens": hm["context_limit_tokens"], "continuation_tokens": hm["continuation_tokens"],
                "vocab_size": hm["vocab_size"], "object_codes": hm["object_codes"], "ask_code": hm["ask_code"],
                "ask_target": hm["ask_target"], "code_token_ids": dict(zip(hm["choice_codes"], hm["choice_token_ids"]))}
        check("the headset constants are the protocol's: A..J and K as 32..42, 8,192 tokens, one continuation",
              hm["choice_token_ids"] == list(range(32, 43)) and hm["choice_codes"] == list("ABCDEFGHIJK")
              and hm["context_limit_tokens"] == 8192 and hm["continuation_tokens"] == 1 and hm["ask_target"] == "ASK")
        want = [("written.context_overflow", "context_budget_exceeded", "context", "context_budget_exceeded"),
                ("written.prompt_hash_mismatch", "invalid_input", "prompt_bytes", "prompt_hash_mismatch"),
                ("written.mapping_ask_not_last", "invalid_input", "mapping", "mapping_invalid"),
                ("written.code_token_mismatch", "invalid_input", "mapping", "code_token_mismatch"),
                ("written.token_out_of_vocabulary", "invalid_input", "token_ids", "token_out_of_vocabulary")]
        got = [(x["request_id"], *(lambda r: (r["outcome"], r["check"], r["reason"]))(RI.check_inputs(x, cons, tokenize=tok.encode)))
               for x in fixes]
        check("each written fixture stops at exactly its intended check", got == want, str(got))
        check("written fixtures are labelled apart and never reach evaluation (boundaries -1)",
              all(x["kind"] == "written_integration_fixture" and x["keep_before_last_token"] == -1
                  and x["keep_before_command_line"] == -1 for x in fixes) and all(x["kind"] == "dataset_command" for x in recs))
        over = fixes[0]
        man = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
        ev = man["fixtures"]["overflow"]
        check("the overflow fixture is just over: n + 1 > 8,192, and one filler line fewer fits",
              over["input_tokens"] + 1 > 8192 and ev["tokens"] == over["input_tokens"] and ev["tokens_one_line_fewer"] + 1 <= 8192,
              str(ev))
        base, changed = base64.b64decode(fixes[2]["prompt_b64"]), base64.b64decode(fixes[1]["prompt_b64"])
        diff = [k for k, (x, y) in enumerate(zip(base, changed)) if x != y]
        check("the hash fixture differs from the base prompt in exactly one byte, w -> W, with the base's hash",
              len(base) == len(changed) and len(diff) == 1 and (base[diff[0]], changed[diff[0]]) == (ord("w"), ord("W"))
              and fixes[1]["prompt_sha256"] == hb(base))
        r = outcome(lambda: RB.build_replay_bundle(**args, out=b, tokenizer=A.OffsetCharTokenizer()))
        check("an existing destination is refused", r == ("input_error", ["E_EVAL_OUTPUT_EXISTS"]), str(r))
        refusals = [
            ("a scores manifest that is not the pinned one",
             dict(policy=replay_policy(RB, tmp, "p1", req, run, "0" * 64, rows)), "E_REPLAY_INPUT", "is not the pinned"),
            ("expected audit requests that differ",
             dict(policy=replay_policy(RB, tmp, "p2", req, run, sc_sha, rows,
                                       **{"selection.expected.a23e_audit_request_ids": ["r0001"]})), "E_REPLAY_SELECTION",
             "the policy expects"),
            ("a reference-run prefix that differs",
             dict(policy=replay_policy(RB, tmp, "p3", req, run, sc_sha, rows,
                                       **{"inputs.small_run_manifest_sha256_prefix": "00000000"})), "E_REPLAY_INPUT",
             "with the recorded prefix"),
        ]
        sc7 = tmp / "scores7"
        sc7_sha = scores_folder(sc7, req, run7, run)
        refusals.append(("a run of the 7B model, even when a scores manifest links it",
                         dict(small_run=run7, scores=sc7, policy=replay_policy(RB, tmp, "p4", req, run7, sc7_sha, rows)),
                         "E_REPLAY_INPUT", "a run of 'qwen2.5-7b-instruct'"))
        refusals.append(("a scores manifest that links another run as the 0.5B's",
                         dict(scores=sc7, policy=replay_policy(RB, tmp, "p5", req, run, sc7_sha, rows)),
                         "E_REPLAY_INPUT", "is not the run the scores manifest records"))
        for k, (label, change, code, why) in enumerate(refusals):
            r = refusal(lambda: RB.build_replay_bundle(**{**args, **change}, out=tmp / f"x{k}", tokenizer=A.OffsetCharTokenizer()))
            check(f"refused: {label}", r is not None and r[0] == [code] and why in r[1] and not (tmp / f"x{k}").exists(), str(r)[:300])
        r = refusal(lambda: RB.build_replay_bundle(**args, out=tmp / "y1", tokenizer=A.MergingTokenizer()))
        check("refused: a tokenizer of another identity", r is not None and r[0] == ["E_REPLAY_TOKENIZER"], str(r)[:300])
        r = refusal(lambda: RB.build_replay_bundle(**args, out=tmp / "y2", tokenizer=ShiftedTokenizer()))
        check("refused: the same identity but other token IDs, every request listed",
              r is not None and r[0] == ["E_REPLAY_TOKENS"] and r[1].count("does not reproduce") == len(union), str(r)[:300])
        CLI = importlib.import_module("grounding.quest.__main__")
        check("the command line reads the bundle back on its own (exit 0)", CLI.main(["verify-replay-bundle", "--bundle", str(b)]) == 0)
        check("the command line refuses --small-run without --scores (exit 2)",
              CLI.main(["verify-replay-bundle", "--bundle", str(b), "--requests", str(req), "--small-run", str(run)]) == 2)
        check("the command line reads it back against the requests, the run and the scores manifest (exit 0)",
              CLI.main(["verify-replay-bundle", "--bundle", str(b), "--requests", str(req), "--small-run", str(run),
                        "--scores", str(tmp / "scores")]) == 0)
        # tampering with the published bundle
        tampers = []
        d = tmp / "t1"
        shutil.copytree(b, d)
        with open(d / "headset" / "requests.jsonl", "ab") as fh:
            fh.write(b" ")
        tampers.append(("a changed byte, not rehashed", RB.verify_replay_bundle(d), True))
        d = tmp / "t2"
        shutil.copytree(b, d)
        rr = jsonl(d / "reference" / "references.jsonl")
        rr[0]["reference"]["scores"][0]["logit"] += 0.5
        write_jsonl(d / "reference" / "references.jsonl", rr)
        rehash(d, ["reference/references.jsonl"])
        tampers.append(("a reference logit changed and rehashed", RB.verify_replay_bundle(d), True))
        d = tmp / "t3"
        shutil.copytree(b, d)
        s = json.loads((d / "selection.json").read_text(encoding="utf-8"))
        s["request_ids"].reverse()
        (d / "selection.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        rehash(d, ["selection.json"])
        tampers.append(("a reordered list, rehashed", RB.verify_replay_bundle(d), True))
        d = tmp / "t4"
        shutil.copytree(b, d)
        fx = jsonl(d / "headset" / "fixtures-written.jsonl")
        fx[1]["prompt_b64"] = fx[2]["prompt_b64"]
        write_jsonl(d / "headset" / "fixtures-written.jsonl", fx)
        h = json.loads((d / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))
        h["fixtures_sha256"] = hb((d / "headset" / "fixtures-written.jsonl").read_bytes())
        (d / "headset" / "replay-manifest.json").write_text(json.dumps(h, indent=2) + "\n", encoding="utf-8")
        rehash(d, ["headset/fixtures-written.jsonl", "headset/replay-manifest.json"])
        tampers.append(("a fixture whose defect is undone, rehashed", RB.verify_replay_bundle(d), True))
        d = tmp / "t5"
        shutil.copytree(b, d)
        rc = jsonl(d / "headset" / "requests.jsonl")
        rr = jsonl(d / "reference" / "references.jsonl")
        rc[0]["keep_before_command_line"] += 1
        rr[0]["boundaries"]["keep_before_command_line"]["keep_tokens"] += 1
        write_jsonl(d / "headset" / "requests.jsonl", rc)
        write_jsonl(d / "reference" / "references.jsonl", rr)
        h = json.loads((d / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))
        h["requests_sha256"] = hb((d / "headset" / "requests.jsonl").read_bytes())
        (d / "headset" / "replay-manifest.json").write_text(json.dumps(h, indent=2) + "\n", encoding="utf-8")
        rehash(d, ["headset/requests.jsonl", "reference/references.jsonl", "headset/replay-manifest.json"])
        alone = RB.verify_replay_bundle(d)
        tampers.append(("a command boundary moved by one token: consistent on its own, found with the tokenizer",
                        RB.verify_replay_bundle(d, requests=req, tokenizer=A.OffsetCharTokenizer()), alone == []))
        d = tmp / "t6"
        shutil.copytree(b, d)
        fx = jsonl(d / "headset" / "fixtures-written.jsonl")
        moved = bytearray(base64.b64decode(fx[1]["prompt_b64"]))
        moved[0] = ord(">")   # another byte changed instead: the fixture still fails where it should
        fx[1]["prompt_b64"] = base64.b64encode(bytes(moved)).decode("ascii")
        write_jsonl(d / "headset" / "fixtures-written.jsonl", fx)
        h = json.loads((d / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))
        h["fixtures_sha256"] = hb((d / "headset" / "fixtures-written.jsonl").read_bytes())
        (d / "headset" / "replay-manifest.json").write_text(json.dumps(h, indent=2) + "\n", encoding="utf-8")
        rehash(d, ["headset/fixtures-written.jsonl", "headset/replay-manifest.json"])
        alone = RB.verify_replay_bundle(d)
        tampers.append(("a fixture changed but still failing as expected: consistent on its own, found by the rebuild",
                        RB.verify_replay_bundle(d, requests=req, tokenizer=A.OffsetCharTokenizer()), alone == []))
        for label, probs, pre in tampers:
            check(f"the readback reports {label}", bool(probs) and pre, str(probs)[:200])
        check("the command line exits 1 on a tampered bundle", CLI.main(["verify-replay-bundle", "--bundle", str(tmp / "t2")]) == 1)


# ------------------------------------------------------------------------------------------------ 6. tracked policy

def check_tracked_policy(RB):
    print("-- the tracked policy")
    pol = RB.load_policy()
    check("it pins A2.3d's requests 0e49a9f6..., its scores manifest 3c92ac4e... and r006's prefix f06cb6c6",
          pol["inputs"]["requests_manifest_sha256"] == "0e49a9f6ed1bd07569ed0a24ec5753cbc4c26a55d1389f0e2240695f8bfe3db0"
          and pol["inputs"]["scores_manifest_sha256"] == "3c92ac4e843704a821f16718635f7815a2bfc78a1ac38c0e2ff517df43e33f57"
          and pol["inputs"]["small_run_manifest_sha256_prefix"] == "f06cb6c6"
          and pol["inputs"]["small_run_id"] == "20261007_A2_r006" and pol["inputs"]["scores_files_read"] == ["manifest.json"])
    check("it expects 32 case requests and the audit's r0001-r0004 and r0329-r0332",
          pol["selection"]["expected"] == {"a23e_case_requests": 32, "a23e_audit_request_ids":
                                           ["r0001", "r0002", "r0003", "r0004", "r0329", "r0330", "r0331", "r0332"]})
    desc = json.loads((REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json").read_text(encoding="utf-8"))
    check("its vocabulary size is the model description's (151,936)",
          pol["headset"]["vocab_size"] == desc["architecture"]["vocab_size"] == 151936)
    check("its model is the 0.5B and its five fixtures are the written ones",
          pol["model_key"] == KEYS[0] and [f["request_id"] for f in pol["written_fixtures"]["fixtures"]] == list(RB.FIXTURE_IDS))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "p.json"
        bad = json.loads(RB.POLICY_PATH.read_text(encoding="utf-8"))
        bad["inputs"]["small_run_manifest_sha256_prefix"] = ""
        p.write_text(json.dumps(bad), encoding="utf-8")
        check("a policy with an empty run prefix is refused", outcome(lambda: RB.load_policy(p)) == ("input_error", ["E_REPLAY_POLICY"]))


# ------------------------------------------------------------------------------------------------ 7. the pinned tokenizer

def check_pinned_tokenizer(RB, RI, directory):
    print("-- the pinned tokenizer on prompts built from the serializer's stored goldens (not acceptance)")
    T = importlib.import_module("grounding.preparation.iref_vla.tokens")
    proto = importlib.import_module("grounding.inference.iref_vla.protocol").load_protocol()
    tok = T.load_pinned_tokenizer(directory)
    gold = REPO / "grounding" / "tests" / "fixtures" / "serialization" / "golden"
    mapping = [["B", "o2", 33], ["A", "o1", 32], ["K", "ASK", 42]]
    names = sorted(p.name for p in gold.glob("*.jsonl") if not p.name.startswith("one_object"))
    for name in names:
        _, prompt = hand_prompt(proto, (gold / name).read_bytes().decode("utf-8"))
        ids = tok.encode(prompt)
        last = attempt(lambda: RB.boundary_before_last_token(tok, prompt, ids))
        chars = prompt.rindex('\n{"command":') + 1
        cmd = attempt(lambda: RB.boundary_before_command_line(tok, prompt, ids, proto, mapping))
        check(f"{name}: the prompt ends <|im_start|> assistant LF (151644, 77091, 198); keep n - 1, text before the LF",
              ids[-3:] == [151644, 77091, 198] and last == {"keep_tokens": len(ids) - 1, "prefix_chars": len(prompt) - 1}, str(last))
        check(f"{name}: the command line starts at a token boundary, and the rest tokenizes alone to the same tokens",
              isinstance(cmd, dict) and cmd["prefix_chars"] == chars and ids[cmd["keep_tokens"]:] == tok.encode(prompt[chars:]),
              str(cmd))
    RBm = importlib.import_module("grounding.quest.replay_bundle")
    cons = {"context_limit_tokens": 8192, "continuation_tokens": 1, "vocab_size": 151936, "object_codes": list("ABCDEFGHIJ"),
            "ask_code": "K", "ask_target": "ASK", "code_token_ids": {c: 32 + k for k, c in enumerate("ABCDEFGHIJK")}}
    fixes, ev = RBm.written_fixtures(RB.load_policy(), proto, tok, cons)
    got = [(x["request_id"], RI.check_inputs(x, cons, tokenize=tok.encode)["reason"]) for x in fixes]
    check("the written fixtures stop at their intended checks under the real tokenizer",
          got == [(f, r) for f, r in zip(RB.FIXTURE_IDS, ("context_budget_exceeded", "prompt_hash_mismatch", "mapping_invalid",
                                                            "code_token_mismatch", "token_out_of_vocabulary"))], str(got))
    check(f"the overflow fixture is just over 8,192 ({ev['filler_lines']} filler lines, {ev['tokens']} tokens; one fewer "
          f"{ev['tokens_one_line_fewer']})", ev["tokens"] + 1 > 8192 >= ev["tokens_one_line_fewer"] + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", help="the pinned 0.5B tokenizer files, for the real-tokenizer checks")
    a = ap.parse_args(argv)
    RI = importlib.import_module("grounding.quest.replay_inputs")
    RB = importlib.import_module("grounding.quest.replay_bundle")
    D = importlib.import_module("grounding.inference.iref_vla_compare.design")
    P = importlib.import_module("grounding.inference.iref_vla_compare.prepare")
    RUN = importlib.import_module("grounding.inference.iref_vla_compare.run")
    check_inputs_unit(RI)
    check_boundaries(RB)
    check_selection(RB)
    check_references(RB)
    check_end_to_end(RB, RI, D, P, RUN)
    check_tracked_policy(RB)
    if a.tokenizer_dir:
        check_pinned_tokenizer(RB, RI, a.tokenizer_dir)
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
