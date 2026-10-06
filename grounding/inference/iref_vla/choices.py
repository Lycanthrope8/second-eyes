"""The one-token choice interface (D81): mapping, prompt bytes, token boundary, selection and scoring.

Pure functions only: no model, no files. Object IDs are never changed; codes A..J alias the scene's objects in their
serialized order and K is ASK at every scene size. ASK is generic non-selection, not a reason.
"""
from __future__ import annotations

import hashlib
import json
import math

from ...evaluation.iref_vla.protocol import EvaluationInputError, issue


class NonFiniteOutput(RuntimeError):
    """A model output that is not a finite number: a technical failure, never an ASK."""


def choice_mapping(object_ids, protocol) -> list:
    codes = protocol["object_codes"]
    if len(object_ids) > len(codes):
        raise EvaluationInputError([issue("scene", "E_PILOT_SCENE", f"{len(object_ids)} objects; the interface offers "
                                                                     f"codes for at most {len(codes)}")])
    return [(c, o) for c, o in zip(codes, object_ids)] + [(protocol["ask_code"], protocol["ask_target"])]


def choices_line(mapping) -> str:
    """One compact JSON line, no spaces, final LF."""
    return json.dumps({"choices": [[m[0], m[1]] for m in mapping]}, separators=(",", ":"), ensure_ascii=False) + "\n"


def build_prompt(protocol, mapping, document: str) -> str:
    w = protocol["wrapper"]
    return w["before_system"] + protocol["system_message"] + w["between"] + choices_line(mapping) + document + w["after_user"]


def check_boundary(tokenizer, prompt: str, mapping, protocol) -> list:
    """Every offered code is one non-special token, its pinned ID, continuing this exact prompt; IDs distinct."""
    base, out, bad = tokenizer.encode(prompt), [], []
    for m in mapping:
        code, target = m[0], m[1]
        ids = tokenizer.encode(code)
        if len(ids) != 1:
            bad.append(f"code {code} encodes as {len(ids)} tokens")
        elif ids[0] != protocol["code_token_ids"][code]:
            bad.append(f"code {code} encodes as token {ids[0]}; the protocol pins {protocol['code_token_ids'][code]}")
        if tokenizer.encode(prompt + code) != base + ids:
            bad.append(f"code {code} does not continue the prompt as its own token")
        out.append((code, target, ids[0] if ids else None))
    if len({t for _, _, t in out}) != len(out):
        bad.append("two offered codes share a token ID")
    if bad:
        raise EvaluationInputError([issue("tokenizer", "E_PILOT_BOUNDARY", m) for m in bad])
    return out


def select_parents(eligible, salt: str, count: int):
    ordered = sorted(eligible, key=lambda p: (hashlib.sha256((salt + p).encode("utf-8")).hexdigest(), p))
    chosen = ordered[:count]
    return chosen, hashlib.sha256("".join(p + "\n" for p in chosen).encode("utf-8")).hexdigest()


def context_status(prompt_tokens: int, limit: int, continuation: int = 1) -> str:
    return "within_context_limit" if prompt_tokens + continuation <= limit else "context_budget_exceeded"


def ceiling_results(prompt_tokens: int, ceilings) -> dict:
    return {str(c): "within_input_ceiling" if prompt_tokens <= c else "exceeds_input_ceiling" for c in ceilings}


def last_position(logits) -> list:
    """The final prompt position's row, whether the model returned only it or every position."""
    if logits and isinstance(logits[0], (list, tuple)):
        return list(logits[-1])
    return list(logits)


def restricted_shares(offered) -> list:
    m = max(offered)
    e = [math.exp(x - m) for x in offered]
    s = math.fsum(e)
    return [x / s for x in e]


def score(mapping, row) -> dict:
    """mapping: (code, target, token ID) in choice order; row: the final position's full-vocabulary logits."""
    for x in row:
        if not math.isfinite(x):
            raise NonFiniteOutput(f"the model returned a non-finite logit ({x})")
    m = max(row)
    lse = m + math.log(math.fsum(math.exp(x - m) for x in row))
    offered = [row[t] for _, _, t in mapping]
    shares = restricted_shares(offered)
    best = max(offered)
    tied = [c for (c, _, _), x in zip(mapping, offered) if x == best]
    ask = mapping[-1][0]
    code, reason = (ask, "exact_score_tie") if len(tied) > 1 else (tied[0], "max_offered_logit")
    ranked = sorted(offered, reverse=True)
    return {"scores": [{"code": c, "target": t, "token_id": i, "logit": row[i], "log_prob": row[i] - lse,
                        "restricted_share": s} for (c, t, i), s in zip(mapping, shares)],
            "choice_code": code, "choice_object_id": None if code == ask else next(t for c, t, _ in mapping if c == code),
            "model_choice": "model_choice_ask" if code == ask else "model_choice_object", "selection_reason": reason,
            "tied_codes": tied if len(tied) > 1 else [], "top_restricted_share": max(shares),
            "logit_margin": ranked[0] - ranked[1] if len(ranked) > 1 else None}
