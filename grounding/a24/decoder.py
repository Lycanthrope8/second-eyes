"""A2.4's constrained decoder for the v1 semantic language (D109, option a): small, bounded, plain Python and torch.

**Masks.** Every ordinary token of the pinned tokenizer is read as its bytes (byte-level BPE, through GPT-2's byte
table). At a decoding step, a token is allowed when appending all of its bytes keeps the output a prefix of some string
of the language: the automaton is walked over a sorted table of token bytes, so every tokenization that spells a valid
continuation stays possible, including tokens that span a separator and a word (" | n", " table"). Special tokens are
never allowed as content; the end token (`<|im_end|>`) is allowed only when the output so far is a complete string.
Masks depend only on the automaton state and are cached per state.

**Greedy choice:** the largest logit among the allowed IDs; an exact tie goes to the lowest token ID.

**Outcomes**, each explicit and distinct from ASK:

- `completed`: the end token chosen at a complete string;
- `length_exhausted`: no end token within the limit;
- `empty_allowed_set`: no token allowed and the string not complete (should be unreachable; reported if it happens);
- `premature_eos`: the end token at an incomplete string (unreachable under the mask; checked on every result).

The decoded text is then validated independently (`language.parse_output`, the query contract); this module never
repairs a string.
"""
from __future__ import annotations

import bisect
import time

from . import language as L


def byte_decoder() -> dict:
    """GPT-2's printable-character to byte table (the inverse of bytes_to_unicode)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class TokenTable:
    """The ordinary tokens' bytes, sorted, for prefix ranges. hf: the pinned Hugging Face tokenizer (its pin already
    checked by `load_pinned_tokenizer`)."""

    def __init__(self, hf, eos_token="<|im_end|>"):
        dec = byte_decoder()
        special = set(hf.all_special_ids) | {int(i) for i in getattr(hf, "added_tokens_decoder", {})}
        entries = []
        for tid in range(len(hf)):
            if tid in special:
                continue
            piece = hf.convert_ids_to_tokens(tid)
            if piece is None:
                continue
            entries.append((bytes(dec[c] for c in piece), tid))
        entries.sort()
        self.keys = [b for b, _ in entries]
        self.ids = [i for _, i in entries]
        self.bytes_of = {i: b for b, i in entries}
        self.eos_id = hf.convert_tokens_to_ids(eos_token)
        if self.eos_id is None or self.eos_id not in special:
            raise ValueError(f"{eos_token} is not a special token of this tokenizer")
        self.size = len(hf)

    def span(self, prefix: bytes, lo: int, hi: int) -> tuple:
        """[a, b): the tokens within [lo, hi) whose bytes start with prefix (prefix is ASCII, so its last byte < 0xff)."""
        a = bisect.bisect_left(self.keys, prefix, lo, hi)
        b = bisect.bisect_left(self.keys, prefix[:-1] + bytes([prefix[-1] + 1]), a, hi)
        return a, b


class Constraint:
    """The language over tokens: allowed IDs per automaton state, and each allowed token's next state."""

    def __init__(self, spec, table: TokenTable):
        self.A, self.T = L.Automaton(spec), table
        self._allowed = {}

    def _search(self, state, prefix: bytes, lo, hi, out):
        if lo < hi and self.T.keys[lo] == prefix and prefix:
            out[self.T.ids[lo]] = state
        for ch in self.A.chars(state):
            b = ch.encode("ascii")
            nxt = self.A.step(state, ch)
            if not nxt:
                continue
            a, z = self.T.span(prefix + b, lo, hi)
            if a < z:
                self._search(nxt, prefix + b, a, z, out)

    def allowed(self, state) -> dict:
        """{token ID: the state after it}; the end token maps to None when the string is complete."""
        if state not in self._allowed:
            out = {}
            self._search(state, b"", 0, len(self.T.keys), out)
            if self.A.accepting(state):
                out[self.T.eos_id] = None
            self._allowed[state] = dict(sorted(out.items()))
        return self._allowed[state]

    def accepts_ids(self, ids) -> tuple:
        """(complete, failing step): whether a token sequence, the end token last, is a path the masks allow."""
        state = self.A.start
        for k, t in enumerate(ids):
            al = self.allowed(state)
            if t not in al:
                return False, k
            if t == self.T.eos_id:
                return k == len(ids) - 1, k
            state = al[t]
        return False, len(ids)


def decode(step_logits, constraint: Constraint, max_new_tokens: int) -> dict:
    """Greedy constrained decoding. step_logits(token or None) returns the next position's logits as a sequence indexable
    by token ID: None for the first call (after the prompt), then the token just chosen. Model-agnostic, so it is
    tested with labelled doubles and runs on the real model through `torch_step`."""
    C, state, ids, text = constraint, constraint.A.start, [], b""
    mask_s = model_s = 0.0
    prev = None
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        logits = step_logits(prev)
        t1 = time.perf_counter()
        al = C.allowed(state)
        if not al:
            model_s += t1 - t0
            mask_s += time.perf_counter() - t1
            return _result("empty_allowed_set", ids, text, state, C, mask_s, model_s)
        best, best_v = None, None
        for tid in al:   # ascending IDs: the first maximum is the lowest ID
            v = float(logits[tid])
            if best_v is None or v > best_v:
                best, best_v = tid, v
        mask_s += time.perf_counter() - t1
        model_s += t1 - t0
        ids.append(best)
        if best == C.T.eos_id:
            return _result("completed" if C.A.accepting(state) else "premature_eos", ids, text, state, C, mask_s, model_s)
        text += C.T.bytes_of[best]
        state = al[best]
        prev = best
    return _result("length_exhausted", ids, text, state, C, mask_s, model_s)


def _result(status, ids, text, state, C, mask_s, model_s) -> dict:
    return {"status": status, "token_ids": list(ids), "text": text.decode("ascii", errors="replace"),
            "complete": bool(state) and C.A.accepting(state), "steps": len(ids),
            "mask_ms": round(mask_s * 1000, 3), "model_ms": round(model_s * 1000, 3)}


def torch_step(model, prompt_ids, device):
    """step_logits for a causal LM with a key-value cache: the prompt in one forward, then one token at a time.
    float32 logits, inference mode; the cache is internal to this closure."""
    import torch
    state = {"past": None}

    def step(prev):
        with torch.inference_mode():
            if prev is None:
                x = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
            else:
                x = torch.tensor([[prev]], dtype=torch.long, device=device)
            out = model(input_ids=x, past_key_values=state["past"], use_cache=True, logits_to_keep=1)
            state["past"] = out.past_key_values
            return out.logits[0, -1].float().cpu()
    return step


def validate(result, spec) -> dict:
    """The independent check of a decoder result: its text read by language.parse_output (not the automaton)."""
    out = dict(result)
    if result["status"] != "completed":
        out["valid"], out["parsed"], out["validation_error"] = False, None, None
        return out
    try:
        out["parsed"] = L.parse_output(result["text"], spec)
        out["valid"], out["validation_error"] = True, None
    except L.LanguageError as e:
        out["parsed"], out["valid"], out["validation_error"] = None, False, str(e)
    return out
