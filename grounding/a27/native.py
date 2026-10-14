"""ctypes binding of libse_llama's functions that A2.7 uses (the model, tokens, logits and grammar-constrained decoding).
The same signatures as native/tests/test_se_grammar.py; Unity binds the same functions in C#."""
from __future__ import annotations

import ctypes as C

STATUS = {0: "completed", 1: "length_exhausted", 2: "empty_allowed_set"}

_SIG = {
    "se_load": (C.c_void_p, [C.c_char_p, C.c_int32, C.c_int32]),
    "se_free": (None, [C.c_void_p]),
    "se_last_error": (C.c_char_p, []),
    "se_llama_version": (C.c_char_p, []),
    "se_system_info": (C.c_char_p, []),
    "se_n_vocab": (C.c_int32, [C.c_void_p]),
    "se_n_ctx": (C.c_int32, [C.c_void_p]),
    "se_is_eog": (C.c_int32, [C.c_void_p, C.c_int32]),
    "se_grammar_new": (C.c_void_p, [C.c_void_p, C.c_char_p, C.c_char_p, C.c_int32]),
    "se_grammar_free": (None, [C.c_void_p]),
    "se_grammar_reset": (None, [C.c_void_p]),
    "se_grammar_allowed": (C.c_int32, [C.c_void_p, C.POINTER(C.c_uint8), C.c_int32]),
    "se_grammar_accept": (C.c_int32, [C.c_void_p, C.c_int32]),
    "se_decode_constrained": (C.c_int32, [C.c_void_p, C.c_void_p, C.POINTER(C.c_int32), C.c_int32, C.c_int32,
                                          C.POINTER(C.c_int32), C.POINTER(C.c_int32)]),
}


class NativeError(RuntimeError):
    pass


class Native:
    """One loaded model with one grammar. Not thread-safe: one per process."""

    def __init__(self, lib_path, gguf_path, gbnf: str, end_token: int, n_ctx: int = 1024, n_threads: int = 4):
        self.lib = C.CDLL(str(lib_path))
        for name, (res, args) in _SIG.items():
            f = getattr(self.lib, name)
            f.restype, f.argtypes = res, args
        self.m = self.lib.se_load(str(gguf_path).encode("utf-8"), n_ctx, n_threads)
        if not self.m:
            raise NativeError(f"could not load {gguf_path}: {self.error()}")
        self.n_vocab = self.lib.se_n_vocab(self.m)
        self.g = self.lib.se_grammar_new(self.m, gbnf.encode("utf-8"), b"root", end_token)
        if not self.g:
            raise NativeError(f"the grammar was refused: {self.error()}")
        self.end_token = end_token

    def error(self) -> str:
        return (self.lib.se_last_error() or b"").decode("utf-8", "replace")

    def version(self) -> str:
        return (self.lib.se_llama_version() or b"").decode("utf-8", "replace")

    def allowed_after(self, prefix) -> set | None:
        """The tokens the grammar allows after this token prefix (one full-vocabulary application); None if the grammar
        refuses a token of the prefix."""
        self.lib.se_grammar_reset(self.g)
        for t in prefix:
            if self.lib.se_grammar_accept(self.g, int(t)) != 0:
                return None
        mask = (C.c_uint8 * self.n_vocab)()
        n = self.lib.se_grammar_allowed(self.g, mask, self.n_vocab)
        if n < 0:
            raise NativeError(self.error())
        raw = bytes(mask)
        return {i for i, b in enumerate(raw) if b}

    def decode(self, prompt_ids, max_new: int) -> dict:
        """One request: a fresh cache, then greedy constrained decoding (se_decode_constrained)."""
        p = (C.c_int32 * len(prompt_ids))(*prompt_ids)
        out = (C.c_int32 * max_new)()
        status = C.c_int32(-1)
        n = self.lib.se_decode_constrained(self.m, self.g, p, len(prompt_ids), max_new, out, C.byref(status))
        if n < 0 or status.value < 0:
            return {"status": "technical_failure", "token_ids": [], "error": self.error()}
        return {"status": STATUS[status.value], "token_ids": list(out[:n])}

    def close(self):
        if getattr(self, "g", None):
            self.lib.se_grammar_free(self.g)
            self.g = None
        if getattr(self, "m", None):
            self.lib.se_free(self.m)
            self.m = None
