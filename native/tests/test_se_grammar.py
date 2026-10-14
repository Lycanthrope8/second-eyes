"""Tests of libse_llama's grammar-constrained decoding through ctypes (A2.7), against A2.4's Python decoder.

    python native/tests/test_se_grammar.py LIB MODEL.gguf

LIB is the built libse_llama (.so, .dll or .dylib); MODEL a GGUF with Qwen's tokenizer: the real model, or
native/tests/make_tiny_model.py's (random weights). Run from the repository root. Checks:
- the allowed tokens at the 21 frozen prefixes equal the Python decoder's, and its frozen expectations;
- along every valid corpus string, each token is allowed in turn, and the end token exactly where the string is complete;
- only <|im_end|> ends an output: every other end-of-generation token is never allowed;
- the greedy pick takes the lowest ID among tied maxima, and never a forbidden token;
- the native decode equals the Python decoder fed the same model's logits, token for token, completed and length
  exhausted alike; an unsatisfiable grammar gives an empty allowed set; a technical failure is kept apart;
- a refused token leaves the grammar unchanged.
Prints one line per check and exits 1 if any fails.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def bind(path: str):
    lib = C.CDLL(path)
    sig = {
        "se_load": (C.c_void_p, [C.c_char_p, C.c_int32, C.c_int32]),
        "se_free": (None, [C.c_void_p]),
        "se_last_error": (C.c_char_p, []),
        "se_n_vocab": (C.c_int32, [C.c_void_p]),
        "se_n_ctx": (C.c_int32, [C.c_void_p]),
        "se_n_cached": (C.c_int32, [C.c_void_p]),
        "se_tokenize": (C.c_int32, [C.c_void_p, C.c_char_p, C.POINTER(C.c_int32), C.c_int32]),
        "se_is_eog": (C.c_int32, [C.c_void_p, C.c_int32]),
        "se_eval": (C.c_int32, [C.c_void_p, C.POINTER(C.c_int32), C.c_int32, C.c_int32]),
        "se_logits": (C.c_int32, [C.c_void_p, C.POINTER(C.c_float), C.c_int32]),
        "se_grammar_new": (C.c_void_p, [C.c_void_p, C.c_char_p, C.c_char_p, C.c_int32]),
        "se_grammar_free": (None, [C.c_void_p]),
        "se_grammar_reset": (None, [C.c_void_p]),
        "se_grammar_allowed": (C.c_int32, [C.c_void_p, C.POINTER(C.c_uint8), C.c_int32]),
        "se_grammar_pick": (C.c_int32, [C.c_void_p, C.POINTER(C.c_float), C.c_int32]),
        "se_grammar_accept": (C.c_int32, [C.c_void_p, C.c_int32]),
        "se_decode_constrained": (C.c_int32, [C.c_void_p, C.c_void_p, C.POINTER(C.c_int32), C.c_int32, C.c_int32,
                                              C.POINTER(C.c_int32), C.POINTER(C.c_int32)]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype, f.argtypes = res, args
    return lib


class Native:
    def __init__(self, lib, model):
        self.lib, self.m = lib, model
        self.n_vocab = lib.se_n_vocab(model)

    def tokenize(self, text: str) -> list:
        buf = (C.c_int32 * 4096)()
        n = self.lib.se_tokenize(self.m, text.encode("utf-8"), buf, 4096)
        assert n >= 0, self.lib.se_last_error()
        return list(buf[:n])

    def allowed(self, g) -> set:
        mask = (C.c_uint8 * self.n_vocab)()
        n = self.lib.se_grammar_allowed(g, mask, self.n_vocab)
        assert n >= 0, self.lib.se_last_error()
        ids = {i for i in range(self.n_vocab) if mask[i]}
        assert len(ids) == n
        return ids

    def logits(self):
        buf = (C.c_float * self.n_vocab)()
        assert self.lib.se_logits(self.m, buf, self.n_vocab) == self.n_vocab
        return buf

    def decode(self, g, prompt, max_new):
        p = (C.c_int32 * len(prompt))(*prompt)
        out = (C.c_int32 * max_new)()
        status = C.c_int32(-9)
        n = self.lib.se_decode_constrained(self.m, g, p, len(prompt), max_new, out, C.byref(status))
        return n, list(out[:max(n, 0)]), status.value


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    import numpy as np
    from grounding.a24 import decoder as D
    from grounding.a24 import language as L
    from grounding.preparation.iref_vla import tokens as T
    TL = __import__("grounding.tests.test_a24_language", fromlist=["PREFIXES"])
    spec = L.load_spec()
    table = D.TokenTable(T.load_pinned_tokenizer(TL.TOKENIZER)._backend)
    con = D.Constraint(spec, table)
    end = table.eos_id
    lib = bind(argv[0])
    model = lib.se_load(argv[1].encode("utf-8"), 2048, 2)
    if not model:
        print("FAIL  load:", lib.se_last_error())
        return 1
    nat = Native(lib, model)
    gbnf = L.gbnf(spec).encode("utf-8")
    g = lib.se_grammar_new(model, gbnf, b"root", end)
    check("the v1 grammar loads into llama.cpp's grammar sampler, ending with <|im_end|>", bool(g),
          (lib.se_last_error() or b"").decode())
    bad_end = lib.se_grammar_new(model, gbnf, b"root", 100)
    bad_gbnf = lib.se_grammar_new(model, b"root ::= undefined_rule", b"root", end)
    check("a non-end token as the end, or an invalid grammar, is refused with a message", not bad_end and not bad_gbnf
          and bool(lib.se_last_error()))
    eog = [i for i in range(nat.n_vocab) if lib.se_is_eog(model, i)]

    # 1. the allowed tokens at the frozen prefixes
    frozen = json.loads(TL.EXPECTATIONS.read_text(encoding="utf-8"))["masks"]
    diffs, frozen_bad = [], []
    for p in TL.PREFIXES:
        lib.se_grammar_reset(g)
        for t in nat.tokenize(p):
            assert lib.se_grammar_accept(g, t) == 0, (p, t, lib.se_last_error())
        mine = nat.allowed(g)
        theirs = set(con.allowed(con.A.walk(p)))
        if mine != theirs:
            diffs.append((p, len(mine), len(theirs), sorted(mine ^ theirs)[:5]))
        ids = sorted(mine)
        f = frozen[p]
        if not (f["count"] == len(ids) and f["eos"] == (end in mine)
                and f["ids_sha256"] == hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()):
            frozen_bad.append(p)
    check(f"at the {len(TL.PREFIXES)} frozen prefixes, the native allowed tokens equal the Python decoder's", not diffs,
          str(diffs[:3]))
    check("and they equal the frozen expectations (count, end token, ID-list hash)", not frozen_bad, str(frozen_bad))

    # 2. every valid corpus string, step by step
    from grounding.a24 import gbnf_check as G
    valid = [x[0] for x in G.corpus(spec) if x[1]]
    walk_bad = []
    for s in valid:
        lib.se_grammar_reset(g)
        state = con.A.start
        for k, t in enumerate(nat.tokenize(s)):
            al = nat.allowed(g)
            if t not in al or ((end in al) != con.A.accepting(state)) or set(con.allowed(state)) != al:
                walk_bad.append((s, k))
                break
            state = con.allowed(state)[t]
            lib.se_grammar_accept(g, t)
        else:
            if end not in nat.allowed(g) or not con.A.accepting(state):
                walk_bad.append((s, "end"))
    check(f"along every one of the {len(valid)} valid corpus strings, the allowed tokens equal the Python decoder's at "
          "every step, and the end token is allowed exactly where the string is complete", valid and not walk_bad,
          str(walk_bad[:3]))
    others = [i for i in eog if i != end]
    lib.se_grammar_reset(g)
    for t in nat.tokenize("chair | near | table"):
        lib.se_grammar_accept(g, t)
    al = nat.allowed(g)
    check("only <|im_end|> ends an output: every other end-of-generation token is never allowed, even where the string "
          "is complete", end in al and others and not (set(others) & al), str(others))

    # 3. the greedy pick
    lib.se_grammar_reset(g)
    start_allowed = sorted(nat.allowed(g))
    zeros = (C.c_float * nat.n_vocab)()
    tie = (C.c_float * nat.n_vocab)()
    a, b = start_allowed[3], start_allowed[7]
    tie[a] = tie[b] = 5.0
    forbid = (C.c_float * nat.n_vocab)()
    forbidden = next(i for i in range(nat.n_vocab) if i not in set(start_allowed))
    forbid[forbidden] = 100.0
    forbid[b] = 1.0
    check("the greedy pick takes the lowest ID among tied maxima, and never a forbidden token, however high its score",
          lib.se_grammar_pick(g, zeros, nat.n_vocab) == start_allowed[0] and lib.se_grammar_pick(g, tie, nat.n_vocab) == a
          and lib.se_grammar_pick(g, forbid, nat.n_vocab) == b, f"{start_allowed[:8]} forbidden {forbidden}")
    rng = np.random.default_rng(7)
    rand_bad = []
    for p in ("", "chair | ", "chair | between | door", "orange juicer [", "chair | near | table"):
        lib.se_grammar_reset(g)
        for t in nat.tokenize(p):
            lib.se_grammar_accept(g, t)
        al = sorted(con.allowed(con.A.walk(p)))
        for _ in range(3):
            lg = rng.normal(size=nat.n_vocab).astype(np.float32)
            lg[rng.integers(0, nat.n_vocab, 50)] = 9.0   # ties among high scores, mostly on forbidden tokens
            want = max(al, key=lambda i: (lg[i], -i))
            got = lib.se_grammar_pick(g, lg.ctypes.data_as(C.POINTER(C.c_float)), nat.n_vocab)
            if got != want:
                rand_bad.append((p, got, want))
    check("on random logits with planted ties, the native pick equals the full-vocabulary rule (the highest allowed "
          "logit, ties to the lowest ID) at five grammar states", not rand_bad, str(rand_bad[:3]))
    bad_scores = []
    for label, fill in (("one NaN", "nan1"), ("all NaN", "nanall"), ("+infinity", "pinf"), ("-infinity", "ninf")):
        lg = np.zeros(nat.n_vocab, dtype=np.float32)
        allowed_now = sorted(con.allowed(con.A.walk("")))
        if fill == "nan1":
            lg[allowed_now[5]] = np.nan
        elif fill == "nanall":
            lg[:] = np.nan
        elif fill == "pinf":
            lg[allowed_now[2]] = np.inf
        else:
            lg[allowed_now[0]] = -np.inf
        lib.se_grammar_reset(g)
        got = lib.se_grammar_pick(g, lg.ctypes.data_as(C.POINTER(C.c_float)), nat.n_vocab)
        err = (lib.se_last_error() or b"").decode()
        if got != -2 or "not a finite number" not in err:
            bad_scores.append((label, got, err))
    check("a NaN or infinite score is a technical failure (-2 with a message), never a token, an empty allowed set or "
          "an answer: one NaN, all NaN, +infinity, -infinity", not bad_scores, str(bad_scores))
    lib.se_grammar_reset(g)
    before = nat.allowed(g)
    refused = lib.se_grammar_accept(g, forbidden)
    check("a refused token returns an error and leaves the grammar unchanged", refused < 0 and nat.allowed(g) == before)

    # 4. the native decode against the Python decoder on the same model's logits
    prompt = nat.tokenize("Command: the chair near the table\nAnswer: ")

    def step_logits(prev):
        if prev is None:
            arr = (C.c_int32 * len(prompt))(*prompt)
            assert lib.se_eval(model, arr, len(prompt), 0) == 0
        else:
            one = (C.c_int32 * 1)(prev)
            assert lib.se_eval(model, one, 1, lib.se_n_cached(model)) == 0
        return np.frombuffer(nat.logits(), dtype=np.float32).copy()
    names = {0: "completed", 1: "length_exhausted", 2: "empty_allowed_set"}
    same = []
    for max_new in (64, 2):
        n, toks, status = nat.decode(g, prompt, max_new)
        ref = D.decode(step_logits, con, max_new)
        same.append((max_new, toks == ref["token_ids"] and names.get(status) == ref["status"], names.get(status),
                     ref["status"], toks[:6], ref["token_ids"][:6]))
    full = same[0]
    check("the native decode equals the Python decoder fed the same model's logits, token for token (max 64)",
          full[1] and full[2] == "completed", str(full))
    check("and stops at the limit as length exhausted, as Python does (max 2)", same[1][1] and same[1][2] == "length_exhausted",
          str(same[1]))
    n, toks, status = nat.decode(g, prompt, 64)
    text = "".join(table.bytes_of[t].decode("ascii") for t in toks[:-1])
    try:
        L.parse_output(text, spec)
        parsed = True
    except Exception:   # noqa: BLE001
        parsed = False
    check("a completed native decode is a string of the language (read back by the independent reader)",
          status == 0 and toks[-1] == end and parsed, repr(text))
    dead = lib.se_grammar_new(model, b'root ::= "q" "\\x00"', b"root", end)   # llama.cpp matches no token to NUL
    if dead:
        n, toks, status = nat.decode(dead, prompt, 8)
        check("an unsatisfiable continuation gives the empty-allowed-set outcome: the forced 'q', then no token",
              status == 2 and toks == nat.tokenize("q"), f"{status} {toks}")
        lib.se_grammar_free(dead)
    else:
        check("an unsatisfiable continuation gives the empty-allowed-set outcome", False,
              "llama.cpp refused the grammar: " + (lib.se_last_error() or b"").decode())
    too_long = list(range(10, 10 + lib.se_n_ctx(model) + 5))
    n, toks, status = nat.decode(g, too_long, 4)
    check("a technical failure (a prompt longer than the context) is a negative return and status -1, never an outcome",
          n < 0 and status == -1 and bool(lib.se_last_error()))
    lib.se_grammar_free(g)
    lib.se_free(model)
    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
