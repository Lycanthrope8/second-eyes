"""Tests of libse_llama through ctypes, the way Unity will call it (A1.8b).

    python native/tests/test_se_llama.py LIB MODEL.gguf REFERENCE_DIR

LIB is the built libse_llama (.so, .dll or .dylib), MODEL a GGUF with Qwen's tokenizer (the real model, or
native/tests/make_tiny_model.py's), REFERENCE_DIR a folder with grounding/reference.py's reference_*.json. Checks:
the references' prompts tokenize to exactly their token IDs; the end-of-turn token ends generation; a cached scene plus
the rest scores like the whole prompt; se_score equals scoring token by token and restores the cache; errors are
reported. Prints one line per check and exits 1 if any fails.
"""
from __future__ import annotations

import ctypes as C
import json
import math
import sys
from pathlib import Path

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def bind(path: str):
    lib = C.CDLL(path)
    sig = {
        "se_system_info": ([], C.c_char_p), "se_llama_version": ([], C.c_char_p), "se_last_error": ([], C.c_char_p),
        "se_load": ([C.c_char_p, C.c_int32, C.c_int32], C.c_void_p), "se_free": ([C.c_void_p], None),
        "se_n_vocab": ([C.c_void_p], C.c_int32), "se_n_ctx": ([C.c_void_p], C.c_int32),
        "se_tokenize": ([C.c_void_p, C.c_char_p, C.POINTER(C.c_int32), C.c_int32], C.c_int32),
        "se_piece": ([C.c_void_p, C.c_int32, C.c_char_p, C.c_int32], C.c_int32),
        "se_is_eog": ([C.c_void_p, C.c_int32], C.c_int32), "se_n_cached": ([C.c_void_p], C.c_int32),
        "se_eval": ([C.c_void_p, C.POINTER(C.c_int32), C.c_int32, C.c_int32], C.c_int32),
        "se_argmax": ([C.c_void_p], C.c_int32), "se_logprob": ([C.c_void_p, C.c_int32], C.c_double),
        "se_logits": ([C.c_void_p, C.POINTER(C.c_float), C.c_int32], C.c_int32),
        "se_score": ([C.c_void_p, C.POINTER(C.c_int32), C.c_int32], C.c_double),
        "se_memory_kb": ([C.c_int32], C.c_int64), "se_set_threads": ([C.c_void_p, C.c_int32], None),
    }
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    return lib


def log_softmax(row: list) -> list:
    top = max(row)
    lse = top + math.log(sum(math.exp(v - top) for v in row))
    return [v - lse for v in row]


def ints(values):
    return (C.c_int32 * len(values))(*values)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(__doc__)
        return 2
    lib, model_path, ref_dir = bind(argv[0]), argv[1], Path(argv[2])
    print("llama.cpp", lib.se_llama_version().decode(), "|", lib.se_system_info().decode().strip())
    check("a missing model fails cleanly", lib.se_load(b"/no/such/model.gguf", 512, 1) is None,
          lib.se_last_error().decode()[:90])
    s = lib.se_load(model_path.encode(), 1024, 2)
    if not s:
        print("FAIL  load:", lib.se_last_error().decode())
        return 1
    n_vocab = lib.se_n_vocab(s)

    def tokenize(text: str) -> list:
        buf = (C.c_int32 * (len(text) + 16))()
        n = lib.se_tokenize(s, text.encode("utf-8"), buf, len(buf))
        return list(buf[:n])

    def last_logits() -> list:
        buf = (C.c_float * n_vocab)()
        lib.se_logits(s, buf, n_vocab)
        return list(buf)

    # 1. Tokenization against the PC references (Hugging Face's tokenizer).
    refs = sorted(ref_dir.glob("reference_*_fp32.json"))
    check("reference files found", bool(refs), str(ref_dir))
    for path in refs:
        ref = json.loads(path.read_text(encoding="utf-8"))
        ids = tokenize(ref["prompt"]["formatted"])
        want = ref["prompt"]["token_ids"]
        first = next((i for i, (a, b) in enumerate(zip(ids, want)) if a != b), None)
        check(f"tokens of {ref['prompt']['id']} equal the reference's", ids == want,
              f"{len(ids)} tokens" + ("" if ids == want else f", reference {len(want)}, first difference at {first}"))
        answer = tokenize(ref["answer"]["text"])
        check(f"the answer of {ref['prompt']['id']} tokenizes as the reference's", answer == ref["answer"]["token_ids"],
              f"{len(answer)} tokens")
    check("<|im_end|> (151645) ends generation", lib.se_is_eog(s, 151645) == 1)
    check("an ordinary token doesn't", lib.se_is_eog(s, 2011) == 0)
    piece = C.create_string_buffer(64)
    n = lib.se_piece(s, 151644, piece, 64)
    check("special tokens render as text", piece.raw[:n] == b"<|im_start|>", piece.raw[:n].decode())

    # 2. A cached scene plus the rest scores like the whole prompt.
    ref = json.loads(refs[0].read_text(encoding="utf-8"))
    formatted = ref["prompt"]["formatted"]
    split = formatted.index("\nUser at ") + 1
    ids, prefix = tokenize(formatted), tokenize(formatted[:split])
    check("the scene's tokens are a prefix of the prompt's", ids[:len(prefix)] == prefix,
          f"{len(prefix)} of {len(ids)} tokens")
    check("a whole prompt evaluates", lib.se_eval(s, ints(ids), len(ids), 0) == 0 and lib.se_n_cached(s) == len(ids))
    whole = last_logits()
    lib.se_eval(s, ints(prefix), len(prefix), 0)
    rest = ids[len(prefix):]
    check("the rest evaluates after the kept scene", lib.se_eval(s, ints(rest), len(rest), len(prefix)) == 0 and
          lib.se_n_cached(s) == len(ids))
    cached = last_logits()
    diff = max(abs(a - b) for a, b in zip(log_softmax(whole), log_softmax(cached)))
    # One batch against two gives slightly different float rounding; a misplaced position would differ by far more.
    check("cached scene + rest = whole prompt (log-probabilities)", diff < 1e-2, f"largest difference {diff:.2e}")

    # 3. se_score = token-by-token scoring, and the cache and scores are restored.
    answer_prefix = tokenize('{"action": "INSPECT", "target": "')
    lib.se_eval(s, ints(answer_prefix), len(answer_prefix), len(ids))
    base, before = lib.se_n_cached(s), last_logits()
    cand = tokenize('box_2"}')
    score = lib.se_score(s, ints(cand), len(cand))
    check("se_score restores the cache", lib.se_n_cached(s) == base, f"{lib.se_n_cached(s)} vs {base}")
    check("se_score restores the last scores", last_logits() == before)
    manual = lib.se_logprob(s, cand[0])
    for i in range(len(cand) - 1):
        lib.se_eval(s, ints([cand[i]]), 1, base + i)
        manual += lib.se_logprob(s, cand[i + 1])
    check("se_score = token by token", abs(score - manual) < 1e-4, f"{score:.5f} vs {manual:.5f}")
    lib.se_eval(s, None, 0, base)
    check("after a drop, no scores until the next evaluation", math.isnan(lib.se_logprob(s, cand[0])))
    lib.se_eval(s, ints(answer_prefix), len(answer_prefix), len(ids))
    lp = [lib.se_logprob(s, t) for t in range(n_vocab)]
    total = sum(math.exp(v) for v in lp)
    check("log-probabilities sum to 1", abs(total - 1.0) < 1e-3, f"{total:.6f}")
    check("argmax is the most likely token", lp[lib.se_argmax(s)] == max(lp))

    # 4. Errors are reported, not crashes.
    check("keep beyond the cache is refused", lib.se_eval(s, ints([1]), 1, lib.se_n_cached(s) + 5) < 0,
          lib.se_last_error().decode())
    too_many = [11] * (lib.se_n_ctx(s) + 1)
    check("a prompt longer than the context is refused", lib.se_eval(s, ints(too_many), len(too_many), 0) < 0,
          lib.se_last_error().decode())
    check("memory is reported", lib.se_memory_kb(1) > 0, f"peak {lib.se_memory_kb(1) / 1024:.0f} MB")
    lib.se_free(s)
    print(f"\n{len(FAILED)} failed" if FAILED else "\nAll checks passed.")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
