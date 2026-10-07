"""A2.3e cache check (D96): does exact-prefix KV reuse preserve the 0.5B model's output? On the RTX PC.

For each request of the 16 cost cases (both formats), the uncached production forward is compared with cached scoring of
the same full token sequence. Three kinds of reuse, each only of an exactly matching token prefix:
- repeated identical request: the prefix is every token but the last;
- across commands: the longest common prefix with the other format of the same parent and view, and with a request of a
  different parent (where the choices line, which precedes the scene, limits reuse);
- after an invalidating change: the original request's cache, offered to a copy whose scene evidence, pose or choices
  were edited, can serve only the tokens before the edit; a different model or tokenizer identity gets no reuse at all.
Each comparison applies D56: the same best candidate, the same order among candidates with at least 1% restricted share
(the plausible set checked on both paths), and a total variation distance of at most 0.05. A pass shows the cache
preserves the model's output, wrong choices included; it does not improve grounding. Prefix creation, suffix scoring,
the repeated-command latency and peak GPU memory are reported separately (PC measurements). The main accuracy runs stay
uncached; this adds no 7B cache study and changes no prompt.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl, issue,
                                             runtime, sha256, strict_json)
from ...preparation.iref_vla import tokens as T
from ..iref_vla.choices import last_position, score
from ..iref_vla.model import TorchModel, checkpoint_evidence
from ..iref_vla.prepare import rows_of
from ..iref_vla.protocol import load_protocol
from . import design as D
from .costs import select_cases
from .prepare import MODEL_KEYS, code_hashes, verify_compare_requests
from .run import model_protocol

KEY = MODEL_KEYS[0]
PLAUSIBLE = 0.01
TVD_MAX = 0.05
OUTPUTS = ("checks.jsonl", "report.md", "summary.json")


def _fail(code, problems):
    raise EvaluationInputError([issue(w, code, m) for w, m in problems])


def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class PrefixCache:
    """Reuse only an exactly matching token prefix, and only for the same model and tokenizer identity."""

    def __init__(self):
        self.entries = []

    def add(self, identity, ids):
        self.entries.append((identity, list(ids)))

    def lookup(self, identity, ids) -> int:
        best = 0
        for ident, stored in self.entries:
            if ident == identity:
                best = max(best, common_prefix(stored, ids))
        return best


def d56(a, b) -> dict:
    """D56 between two score() results over the same mapping."""
    sa = {x["code"]: x["restricted_share"] for x in a["scores"]}
    sb = {x["code"]: x["restricted_share"] for x in b["scores"]}
    pa = sorted((c for c in sa if sa[c] >= PLAUSIBLE), key=lambda c: (-sa[c], c))
    pb = sorted((c for c in sb if sb[c] >= PLAUSIBLE), key=lambda c: (-sb[c], c))
    tvd = 0.5 * sum(abs(sa[c] - sb[c]) for c in sorted(sa))
    same_best = a["choice_code"] == b["choice_code"]
    return {"same_best": same_best, "plausible_uncached": pa, "plausible_cached": pb, "same_plausible_order": pa == pb,
            "tvd": tvd, "max_abs_logit_difference": max(abs(x["logit"] - y["logit"]) for x, y in zip(a["scores"], b["scores"])),
            "passed": same_best and pa == pb and tvd <= TVD_MAX}


class TorchCacheModel:
    """A2.3a's TorchModel (float32, eager, evaluation mode) with a prefill and a cached-suffix path added."""

    def __init__(self, tm):
        self.tm, self.torch, self.model = tm, tm.torch, tm.model

    @classmethod
    def load(cls, model_dir, device, proto):
        return cls(TorchModel.load(model_dir, device, proto))

    def info(self):
        return self.tm.info()

    def synchronize(self):
        self.tm.synchronize()

    def reset_peak(self):
        if self.tm.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def peak(self):
        return self.tm.peak_memory()

    def uncached(self, ids) -> list:
        return last_position(self.tm.forward_last(ids))

    def prefill(self, ids):
        t = self.torch
        with t.inference_mode():
            x = t.tensor([list(ids)], dtype=t.long, device=self.tm.device)
            out = self.model(input_ids=x, attention_mask=t.ones_like(x), use_cache=True, logits_to_keep=1)
            return out.past_key_values

    def suffix(self, cache, prefix_len, ids) -> list:
        t = self.torch
        with t.inference_mode():
            x = t.tensor([list(ids)], dtype=t.long, device=self.tm.device)
            mask = t.ones((1, prefix_len + len(ids)), dtype=t.long, device=self.tm.device)
            out = self.model(input_ids=x, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1)
            row = out.logits[0, -1].float().cpu().tolist()
        cache.crop(prefix_len)  # restore the prefix for the next reuse
        return row


def _timed(model, fn, *a):
    model.synchronize()
    t0 = time.perf_counter_ns()
    r = fn(*a)
    model.synchronize()
    return r, (time.perf_counter_ns() - t0) / 1e6


def perturb(prompt: str, what: str):
    """A copy with one evidence change, or None if the prompt has no such field. Returns (text, character offset)."""
    if what == "choices":
        i = prompt.index('{"choices":[["') + len('{"choices":[["')
        j = prompt.index('],["', i) + len('],["')
        a, b = prompt[i], prompt[j]
        return prompt[:i] + b + prompt[i + 1:j] + a + prompt[j + 1:], i
    if what == "scene":
        i = prompt.index('{"objects"')
        j = next(k for k in range(i, len(prompt)) if prompt[k].isdigit())
        return prompt[:j] + ("7" if prompt[j] != "7" else "3") + prompt[j + 1:], j
    if what == "pose":
        key = '{"pose":{"pose_kind":"none"'
        i = prompt.find(key)
        if i < 0:
            return None
        return prompt[:i] + '{"pose":{"pose_kind":"user"' + prompt[i + len(key):], i + len('{"pose":{"pose_kind":"')
    raise ValueError(what)


def check_request(model, tok, r, ids, prompt, partners, proto, identity, repeats=5) -> list:
    mapping = [tuple(m) for m in r["mapping"]]
    out = []
    model.reset_peak()
    logits_u, ms_u = _timed(model, model.uncached, ids)
    su = score(mapping, logits_u)
    base = {"request_id": r["request_id"], "parent_command_id": r["parent_command_id"], "view_id": r["view_id"],
            "format": r["format"], "tokens": len(ids), "uncached_ms": ms_u}
    pc = PrefixCache()
    pc.add(identity, ids)
    # 1. repeated identical request
    L = len(ids) - 1
    model.reset_peak()
    cache, ms_p = _timed(model, model.prefill, ids[:L])
    peak_prefill = model.peak()
    lat = []
    for _ in range(repeats):
        logits_c, ms_s = _timed(model, model.suffix, cache, L, ids[L:])
        lat.append(ms_s)
    peak_suffix = model.peak()
    sc = score(mapping, logits_c)
    out.append({**base, "kind": "repeated_identical", "reused_tokens": L, "suffix_tokens": len(ids) - L,
                "prefix_ms": ms_p, "suffix_ms": lat[0], "repeated_latency_ms": statistics.median(lat),
                "peak_gpu_bytes_prefill": peak_prefill, "peak_gpu_bytes_suffix": peak_suffix, "uncached_choice": su["choice_code"],
                "cached_choice": sc["choice_code"], "d56": d56(su, sc), "cache_lookup_tokens": pc.lookup(identity, ids)})
    del cache
    # 2. across commands: the other format, and another parent
    for kind, other in partners:
        L = common_prefix(other, ids)
        cache, ms_p = _timed(model, model.prefill, other[:L])
        logits_c, ms_s = _timed(model, model.suffix, cache, L, ids[L:])
        sc = score(mapping, logits_c)
        out.append({**base, "kind": kind, "reused_tokens": L, "suffix_tokens": len(ids) - L, "prefix_ms": ms_p, "suffix_ms": ms_s,
                    "uncached_choice": su["choice_code"], "cached_choice": sc["choice_code"], "d56": d56(su, sc)})
        del cache
    # 3. invalidation: an edited copy may reuse the original's tokens only up to the edit
    for what in ("scene", "pose", "choices"):
        p = perturb(prompt, what)
        if p is None:
            out.append({**base, "kind": f"invalidation_{what}", "skipped": "the prompt has no such field"})
            continue
        text, char = p
        mod = tok.encode(text)
        edit_token = len(tok.encode(text[:char]))
        L = pc.lookup(identity, mod)
        mapping_m = mapping
        if what == "choices":  # the first two objects swap letters; their token IDs follow the letters
            (c0, o0, i0), (c1, o1, i1) = mapping[0], mapping[1]
            mapping_m = [(c1, o0, i1), (c0, o1, i0)] + list(mapping[2:])
        uu, _ = _timed(model, model.uncached, mod)
        cache, ms_p = _timed(model, model.prefill, ids[:L])
        cc, ms_s = _timed(model, model.suffix, cache, L, mod[L:])
        a, b = score(mapping_m, uu), score(mapping_m, cc)
        out.append({**base, "kind": f"invalidation_{what}", "reused_tokens": L, "edit_token": edit_token, "suffix_tokens": len(mod) - L,
                    "reuse_stops_at_edit": L <= edit_token, "prefix_ms": ms_p, "suffix_ms": ms_s,
                    "uncached_choice": a["choice_code"], "cached_choice": b["choice_code"], "d56": d56(a, b)})
        del cache
    other_ident = PrefixCache()
    other_ident.add(identity, ids)
    out.append({**base, "kind": "invalidation_identity",
                "reuse_other_model": other_ident.lookup(("another-model",) + tuple(identity[1:]), ids),
                "reuse_other_tokenizer": other_ident.lookup(tuple(identity[:1]) + ("another-tokenizer",), ids)})
    return out


def _passed(c) -> bool:
    if c.get("skipped"):
        return True
    if c["kind"] == "invalidation_identity":
        return c["reuse_other_model"] == 0 and c["reuse_other_tokenizer"] == 0
    ok = c["d56"]["passed"]
    if c["kind"].startswith("invalidation_"):
        ok = ok and c["reuse_stops_at_edit"]
    return ok


def summarize(checks, info) -> dict:
    kinds = sorted({c["kind"] for c in checks})
    by = {}
    for k in kinds:
        cs = [c for c in checks if c["kind"] == k and not c.get("skipped")]
        b = {"checks": len(cs), "passed": sum(_passed(c) for c in cs), "skipped": sum(1 for c in checks if c["kind"] == k and c.get("skipped"))}
        if cs and "reused_tokens" in cs[0]:
            v = sorted(c["reused_tokens"] for c in cs)
            b["reused_tokens"] = {"min": v[0], "median": statistics.median(v), "max": v[-1]}
            b["suffix_tokens"] = {"min": min(c["suffix_tokens"] for c in cs), "max": max(c["suffix_tokens"] for c in cs)}
            b["prefix_ms_median"] = statistics.median(c["prefix_ms"] for c in cs)
            b["suffix_ms_median"] = statistics.median(c["suffix_ms"] for c in cs)
            b["max_tvd"] = max(c["d56"]["tvd"] for c in cs)
            b["max_abs_logit_difference"] = max(c["d56"]["max_abs_logit_difference"] for c in cs)
        if k == "repeated_identical" and cs:
            b["repeated_latency_ms_median"] = statistics.median(c["repeated_latency_ms"] for c in cs)
            b["uncached_ms_median"] = statistics.median(c["uncached_ms"] for c in cs)
            b["peak_gpu_bytes_prefill_max"] = max((c["peak_gpu_bytes_prefill"] or 0) for c in cs)
            b["peak_gpu_bytes_suffix_max"] = max((c["peak_gpu_bytes_suffix"] or 0) for c in cs)
        by[k] = b
    return {"format_version": 1, "record_type": "iref_compare_cache_check", "model_key": KEY, "by_kind": by,
            "checks": len(checks), "passed": all(_passed(c) for c in checks),
            "criteria": {"plausible_share": PLAUSIBLE, "tvd_max": TVD_MAX, "rule": "D56: same best candidate, same order among "
                         "candidates with at least 1% restricted share on both paths, total variation distance at most 0.05"},
            "model_info": {k: (info or {}).get(k) for k in ("dtype", "attention_implementation", "device_name", "torch", "transformers")},
            "scope": "PC measurements (RTX PC); a pass preserves the model's output, wrong choices included"}


def render(s) -> str:
    L = ["# A2.3e cache check: Qwen2.5-0.5B-Instruct, exact-prefix reuse", "",
         f"Overall: {'every check passed' if s['passed'] else 'SOME CHECKS FAILED'} ({s['checks']} checks). {s['criteria']['rule']}. "
         "A pass means the cache preserves the model's output, wrong choices included; it does not improve grounding.", "",
         "| Reuse | Checks | Passed | Reused tokens (min / median / max) | Prefix creation, median | Suffix scoring, median | Largest TVD |",
         "|---|---|---|---|---|---|---|"]
    for k, b in s["by_kind"].items():
        rt = b.get("reused_tokens")
        L.append(f"| {k} | {b['checks']} | {b['passed']} | "
                 + (f"{rt['min']:,} / {rt['median']:,} / {rt['max']:,}" if rt else "-") + " | "
                 + (f"{b['prefix_ms_median']:.1f} ms" if rt else "-") + " | " + (f"{b['suffix_ms_median']:.1f} ms" if rt else "-")
                 + " | " + (f"{b['max_tvd']:.2e}" if rt else "-") + " |")
    r = s["by_kind"].get("repeated_identical")
    if r:
        L += ["", f"Repeated identical request: uncached forward median {r['uncached_ms_median']:.1f} ms; cached last-token scoring "
                  f"median {r['repeated_latency_ms_median']:.1f} ms; peak GPU bytes {r['peak_gpu_bytes_prefill_max']:,} (prefix) and "
                  f"{r['peak_gpu_bytes_suffix_max']:,} (suffix). PC measurements."]
    return "\n".join(L) + "\n"


def run_cache(*, requests, model_dir, tokenizer_dir, device, out, model_loader=None, tokenizer=None, evidence_fn=None) -> dict:
    out = output.refuse_existing(out)
    req = Path(requests)
    if device != "cuda" and model_loader is None:
        _fail("E_COMPARE_DEVICE", [("device", "the cache check needs --device cuda (no CPU fallback)")])
    bad = verify_compare_requests(req)
    if bad:
        _fail("E_COMPARE_REQUESTS", [(str(req), m) for m in bad[:20]])
    rman = json.loads((req / "manifest.json").read_bytes())
    pol, proto = D.load_policy(req / "policy.json"), load_protocol(req / "protocol.json")
    rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
    prompts = {x["request_id"]: x["prompt"] for x in rows_of("prompts.jsonl", (req / "prompts.jsonl").read_bytes(), "E_COMPARE_REQUESTS")}
    stored = {x["request_id"]: x["token_ids"] for x in rows_of(f"tokens/{KEY}.jsonl", (req / "tokens" / f"{KEY}.jsonl").read_bytes(), "E_COMPARE_REQUESTS")}
    tok = tokenizer if tokenizer is not None else T.load_pinned_tokenizer(tokenizer_dir)
    if tok.identity != rman["tokenizers"][KEY]["identity"]:
        _fail("E_COMPARE_TOKENIZER", [("tokenizer", "not the tokenizer the requests were prepared with")])
    mp, desc = model_protocol(proto, pol, KEY)
    evidence = (evidence_fn or checkpoint_evidence)(model_dir, mp, desc, None)
    cases = select_cases(rows)
    chosen = [r for pv in cases for r in rows if (r["parent_command_id"], r["view_id"]) == pv]
    for r in chosen:
        if tok.encode(prompts[r["request_id"]]) != stored[r["request_id"]]:
            _fail("E_COMPARE_TOKENS", [(r["request_id"], "re-tokenizing does not give the frozen token IDs")])
    model = (model_loader or TorchCacheModel.load)(model_dir, device, mp)
    info = model.info()
    identity = (f"{mp['model']['hf_repo']}@{mp['model']['hf_revision']}", tok.identity)
    checks = []
    for i, r in enumerate(chosen):
        same_pv = [x for x in chosen if (x["parent_command_id"], x["view_id"]) == (r["parent_command_id"], r["view_id"])
                   and x["request_id"] != r["request_id"]]
        other_parent = next(x for x in chosen[i + 1:] + chosen[:i] if x["parent_command_id"] != r["parent_command_id"]
                            and x["format"] == r["format"])
        partners = [("other_format_same_parent_view", stored[same_pv[0]["request_id"]])] if same_pv else []
        partners.append(("other_parent", stored[other_parent["request_id"]]))
        checks += check_request(model, tok, r, stored[r["request_id"]], prompts[r["request_id"]], partners, proto, identity)
    summary = summarize(checks, info)
    files = {"checks.jsonl": encode_jsonl(checks), "summary.json": encode_json(summary), "report.md": render(summary).encode("utf-8")}
    manifest = {"format_version": 1, "record_type": "iref_compare_cache_manifest", "policy_id": D.POLICY_ID, "model_key": KEY,
                "requests_manifest_sha256": sha256((req / "manifest.json").read_bytes()), "checkpoint": evidence,
                "cases": [list(c) for c in cases], "model_info": info, "outputs": {k: sha256(v) for k, v in files.items()},
                "code": code_hashes(), "runtime": runtime()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for k, v in files.items():
            output._write_file(staging / k, v)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        problems = verify_cache(staging)
        if problems:
            raise RuntimeError("the cache check failed readback: " + "; ".join(problems[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def verify_cache(folder) -> list:
    f = Path(folder)
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_COMPARE_CACHE")
        checks = rows_of("checks.jsonl", (f / "checks.jsonl").read_bytes(), "E_COMPARE_CACHE")
        summary = strict_json("summary.json", (f / "summary.json").read_bytes(), "E_COMPARE_CACHE")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable cache check: {e}"]
    bad = [f"{n}: changed" for n, want in manifest.get("outputs", {}).items() if sha256((f / n).read_bytes()) != want]
    if bad:
        return bad
    if summarize(checks, manifest.get("model_info")) != summary:
        return ["summary.json differs from a recomputation"]
    if render(summary).encode("utf-8") != (f / "report.md").read_bytes():
        return ["report.md differs from a rendering of the summary"]
    return []
