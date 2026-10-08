"""A2.5 delivery 1, the desktop diagnosis (D98's desktop llama.cpp; D103's schedule).

The same frozen bundle, the same input checks and the same U/R/P schedule as the headset replay, with the same GGUF, run
by the pinned llama.cpp and wrapper built for this PC (`python tools/build_llama.py host`) through `ctypes`. It keeps the
requested settings (8,192 tokens of context, 2 threads, 16 sequences, flags 0) and records the host's identity. It writes
the headset's files in the headset's format, so `replay-compare` reads both unchanged:

- `identity.json`, with a `host` block instead of the app's;
- `startup-log.txt`: llama.cpp's own startup lines, from this process's stderr, redirected around the load only;
- `selfchecks.json`, `fixtures.jsonl`, `results.jsonl` and `done.json`.

The model must be the accepted file by its full SHA-256. Nothing here changes quantization, repacking, attention,
prompts, boundaries or tolerances. A desktop result helps localize a headset discrepancy; it does not establish its cause
on its own.
"""
from __future__ import annotations

import base64
import ctypes
import datetime
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ..evaluation.iref_vla import output
from .publish import publish
from ..evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError, encode_json, issue
from . import replay_inputs as RI
from .replay_bundle import verify_replay_bundle
from .replay_compare import choose, shares
from .runtime_identity import REPO, file_sha256

REQUESTED = {"n_ctx": 8192, "threads": 2, "n_seq": 16, "flags": 0}
ACCEPTED_MODEL_SHA256 = "dd753cd62f163c8baa8d2e598e3b61385f31cd46ca04488cd88ba01a9c83eb18"
DEFAULT_MODEL = REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct" / "gguf" / "qwen2.5-0.5b-instruct-q8_0.gguf"
DEFAULT_LIBRARY = REPO / "native" / "out" / "host" / ("se_llama.dll" if os.name == "nt" else "libse_llama.so")
HOST_BUILD_RECORD = REPO / "native" / "out" / "host" / "build.json"


class Native:
    """The wrapper's C interface (native/se_llama.h) through ctypes."""

    def __init__(self, path):
        lib = ctypes.CDLL(str(path))
        P, I = ctypes.c_void_p, ctypes.c_int32
        sig = {"se_system_info": ([], ctypes.c_char_p), "se_llama_version": ([], ctypes.c_char_p),
               "se_last_error": ([], ctypes.c_char_p), "se_set_verbose": ([I], None),
               "se_load_ex": ([ctypes.c_char_p, I, I, I, I], P), "se_free": ([P], None), "se_flags": ([P], I),
               "se_n_seq": ([P], I), "se_n_vocab": ([P], I), "se_n_ctx": ([P], I),
               "se_tokenize": ([P, ctypes.c_char_p, ctypes.POINTER(I), I], I), "se_n_cached": ([P], I),
               "se_eval": ([P, ctypes.POINTER(I), I, I], I), "se_logprob": ([P, I], ctypes.c_double),
               "se_logits": ([P, ctypes.POINTER(ctypes.c_float), I], I), "se_memory_kb": ([I], ctypes.c_int64)}
        for name, (args, res) in sig.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = args, res
        self.lib, self.s = lib, None

    def version(self): return (self.lib.se_llama_version() or b"").decode("utf-8", "replace")
    def system_info(self): return (self.lib.se_system_info() or b"").decode("utf-8", "replace").strip()
    def last_error(self): return (self.lib.se_last_error() or b"").decode("utf-8", "replace")
    def set_verbose(self, on): self.lib.se_set_verbose(1 if on else 0)
    def memory_kb(self, peak): return int(self.lib.se_memory_kb(1 if peak else 0))

    def load(self, path, n_ctx, threads, n_seq, flags):
        self.s = self.lib.se_load_ex(str(path).encode("utf-8"), n_ctx, threads, n_seq, flags)
        return bool(self.s)

    def info(self):
        return [self.lib.se_n_ctx(self.s), self.lib.se_n_vocab(self.s), self.lib.se_n_seq(self.s), self.lib.se_flags(self.s)]

    def tokenize(self, data: bytes):
        buf = (ctypes.c_int32 * (len(data) + 16))()
        n = self.lib.se_tokenize(self.s, data, buf, len(buf))
        if n < 0:
            buf = (ctypes.c_int32 * (-n))()
            n = self.lib.se_tokenize(self.s, data, buf, len(buf))
        if n < 0:
            raise RuntimeError("se_tokenize failed: " + self.last_error())
        return list(buf[:n])

    def eval(self, ids, keep):
        return self.lib.se_eval(self.s, (ctypes.c_int32 * len(ids))(*ids), len(ids), keep)

    def n_cached(self): return self.lib.se_n_cached(self.s)
    def logprob(self, token): return self.lib.se_logprob(self.s, token)

    def logits(self, n):
        row = (ctypes.c_float * n)()
        got = self.lib.se_logits(self.s, row, n)
        return list(row[:max(got, 0)])

    def free(self):
        if self.s:
            self.lib.se_free(self.s)
        self.s = None


def _flush_c_stdio():
    for name in (("ucrtbase",) if os.name == "nt" else (None,)):
        try:
            ctypes.CDLL(name).fflush(None)
        except (OSError, AttributeError):
            pass


def captured_load(native, model, path):
    """Verbose on and stderr (descriptor 2, process-wide) redirected into path for the load only; both undone however
    the load ends. Returns (loaded, load ms, capture error or None)."""
    error = None
    saved = fd = None
    try:
        sys.stderr.flush()
        _flush_c_stdio()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o644)
        saved = os.dup(2)
        os.dup2(fd, 2)
    except OSError as e:
        error = f"capture could not start: {e}"
    finally:
        if fd is not None:
            os.close(fd)
    clock = time.perf_counter()
    try:
        if saved is not None:
            native.set_verbose(True)
        ok = native.load(model, REQUESTED["n_ctx"], REQUESTED["threads"], REQUESTED["n_seq"], REQUESTED["flags"])
    finally:
        native.set_verbose(False)
        if saved is not None:
            _flush_c_stdio()
            os.dup2(saved, 2)
            os.close(saved)
    return ok, (time.perf_counter() - clock) * 1000.0, error


def self_checks():
    """The on-device self-checks' cases, through this module's own scoring and the reference input checks."""
    out = []
    add = lambda name, ok, detail="": out.append({"name": name, "passed": bool(ok), "detail": detail})  # noqa: E731
    codes = ["B", "A", "K"]
    add("the larger offered logit wins (B, 2.5)", choose(codes, [2.5, 1.5, -1.0], "K")[0] == "B")
    add("shares over every offered code, K included, sum to 1", abs(math.fsum(shares([2.5, 1.5, -1.0])) - 1) < 1e-12)
    c = choose(codes, [2.5, 2.5, -1.0], "K")
    add("an exact tie at the top goes to K with both codes listed", c == ("K", "exact_score_tie", ["B", "A"]), str(c))
    add("a tie between an object and K also goes to K", choose(codes, [3.0, 0.0, 3.0], "K")[0] == "K")
    prompt = b"hand prompt"
    rec = RI.make_record(request_id="hand", kind="dataset_command", prompt=prompt, token_ids=[5, 6, 7],
                         mapping=[["B", "o2", 33], ["A", "o1", 32], ["K", "ASK", 42]], keep_before_last_token=2,
                         keep_before_command_line=1, expected_outcome="completed")
    cons = {"context_limit_tokens": 10, "continuation_tokens": 1, "vocab_size": 1000, "object_codes": list("ABCDEFGHIJ"),
            "ask_code": "K", "ask_target": "ASK", "code_token_ids": {c: 32 + k for k, c in enumerate("ABCDEFGHIJK")}}
    add("a valid record passes every input check", RI.check_inputs(rec, cons, tokenize=lambda t: [5, 6, 7])["outcome"] == "passed")
    r = dict(rec, prompt_b64=base64.b64encode(b"Hand prompt").decode(), token_ids_sha256="0" * 64)
    add("two defects: the prompt check comes first", RI.check_inputs(r, cons)["check"] == "prompt_bytes")
    r = RI.check_inputs(rec, cons, tokenize=lambda t: [5, 6, 8])
    add("a runtime tokenization that differs stops at tokenization", r["outcome"] == "tokenization_mismatch")
    return out


def _input_line(rec, cons, native):
    seen = {}

    def tokenize(text):
        ids = native.tokenize(text.encode("utf-8"))
        seen["ids"] = ids
        return ids

    t0 = time.perf_counter()
    chk = RI.check_inputs(rec, cons, tokenize=tokenize)
    ids, frozen = seen.get("ids"), rec.get("token_ids") or []
    first = -1
    if ids is not None and ids != frozen:
        first = next((k for k, (a, b) in enumerate(zip(ids, frozen)) if a != b), min(len(ids), len(frozen)))
    data = base64.b64decode(rec["prompt_b64"]) if chk["check"] != "prompt_bytes" else None
    passed_mapping = chk["check"] not in ("prompt_bytes", "mapping")
    return chk, {"outcome": chk["outcome"], "check": chk["check"], "reason": chk["reason"], "detail": chk["detail"],
                 "runtime_tokens": len(ids) if ids is not None else -1, "first_token_difference": first,
                 "prompt_sha256": RI.sha256_hex(data) if data is not None else None,
                 "token_ids_sha256": RI.token_ids_sha256(frozen) if chk["check"] not in ("prompt_bytes", "mapping") else None,
                 "mapping_sha256": RI.mapping_sha256(rec["codes"], rec["targets"], rec["code_token_ids"]) if passed_mapping else None,
                 "ms": (time.perf_counter() - t0) * 1000.0}


def _eval(native, t, offset, count, keep, expect):
    t0 = time.perf_counter()
    rc = native.eval(t[offset:offset + count], keep)
    ms = (time.perf_counter() - t0) * 1000.0
    cached = native.n_cached()
    return rc == 0 and cached == expect, {"offset": offset, "count": count, "keep": keep, "status": rc,
                                         "error": native.last_error() if rc != 0 else None, "n_cached": cached,
                                         "expected_n_cached": expect, "ms": ms}


def _scored(native, rec, ask, n_vocab):
    row = native.logits(n_vocab)
    out = {"logits_returned": len(row)}
    bad = [k for k, x in enumerate(row) if not math.isfinite(x)]
    if not row:
        return dict(out, error="no_scores", non_finite=0, first_non_finite=-1, choice_code=None, selection_reason=None,
                    tied_codes=[], offered=[])
    if bad:
        return dict(out, error="non_finite_output", non_finite=len(bad), first_non_finite=bad[0], choice_code=None,
                    selection_reason=None, tied_codes=[], offered=[])
    offered = [row[i] for i in rec["code_token_ids"]]
    code, reason, tied = choose(rec["codes"], offered, ask)
    sh = shares(offered)
    return dict(out, error=None, non_finite=0, first_non_finite=-1, choice_code=code, selection_reason=reason, tied_codes=tied,
                offered=[{"code": c, "target": t, "token_id": i, "logit": x, "log_prob": native.logprob(i), "restricted_share": s}
                         for c, t, i, x, s in zip(rec["codes"], rec["targets"], rec["code_token_ids"], offered, sh)])


def _paths(native, rec, ask, n_vocab):
    t, n, k = rec["token_ids"], len(rec["token_ids"]), rec["keep_before_command_line"]
    paths = {}
    ok_u, e = _eval(native, t, 0, n, 0, n)
    paths["U"] = dict({"evals": [e]}, **(_scored(native, rec, ask, n_vocab) if ok_u else {"error": "eval_failed_or_cache_mismatch"}))
    if ok_u:
        ok_r, e = _eval(native, t, n - 1, 1, n - 1, n)
        paths["R"] = dict({"evals": [e]}, **(_scored(native, rec, ask, n_vocab) if ok_r else {"error": "eval_failed_or_cache_mismatch"}))
    else:
        paths["R"] = {"error": "skipped: U did not complete"}
    ok_p1, e1 = _eval(native, t, 0, k, 0, k)
    evals = [e1]
    ok_p = ok_p1
    if ok_p1:
        ok_p, e2 = _eval(native, t, k, n - k, k, n)
        evals.append(e2)
    paths["P"] = dict({"evals": evals}, **(_scored(native, rec, ask, n_vocab) if ok_p else {"error": "eval_failed_or_cache_mismatch"}))
    return paths


def run_desktop(*, bundle, out, model=None, library=None, native=None, progress=print,
                accepted_sha256=ACCEPTED_MODEL_SHA256) -> dict:
    """Run the desktop replay into a new folder; returns the done record."""
    out = output.refuse_existing(out)
    b = Path(bundle)
    bad = verify_replay_bundle(b)
    if bad:
        raise EvaluationInputError([issue(str(b), "E_REPLAY_DESKTOP", "the bundle does not read back: " + "; ".join(bad[:3]))])
    model = Path(model) if model is not None else DEFAULT_MODEL
    if not model.is_file() or file_sha256(model) != accepted_sha256:
        raise EvaluationInputError([issue(str(model), "E_REPLAY_DESKTOP", "not the accepted model file (full SHA-256 dd753cd6...)")])
    library = Path(library) if library is not None else DEFAULT_LIBRARY
    if native is None:
        if not library.is_file():
            raise EvaluationInputError([issue(str(library), "E_REPLAY_DESKTOP", "no host build: run python tools/build_llama.py host")])
        native = Native(library)
    hm = json.loads((b / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))
    recs = [json.loads(x) for x in (b / "headset" / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    fixes = [json.loads(x) for x in (b / "headset" / "fixtures-written.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    cons = {"context_limit_tokens": hm["context_limit_tokens"], "continuation_tokens": hm["continuation_tokens"],
            "vocab_size": hm["vocab_size"], "object_codes": hm["object_codes"], "ask_code": hm["ask_code"],
            "ask_target": hm["ask_target"], "code_token_ids": dict(zip(hm["choice_codes"], hm["choice_token_ids"]))}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    total = time.perf_counter()
    try:
        checks = self_checks()
        all_ok = all(c["passed"] for c in checks)
        (staging / "selfchecks.json").write_bytes(encode_json({"record_type": "a25_replay_selfchecks", "checks": checks,
                                                               "all_passed": all_ok}))
        if not all_ok:
            raise RuntimeError("a self-check failed (selfchecks.json)")
        progress("loading the model (8,192 tokens of context, 2 threads)")
        before = (native.memory_kb(False), native.memory_kb(True))
        ok, load_ms, cap_error = captured_load(native, model, staging / "startup-log.txt")
        if not ok:
            raise RuntimeError("the model did not load: " + native.last_error())
        info = native.info()
        after = (native.memory_kb(False), native.memory_kb(True))
        build = HOST_BUILD_RECORD.read_bytes() if HOST_BUILD_RECORD.is_file() else None
        files = {"manifest_sha256": file_sha256(b / "headset" / "replay-manifest.json"),
                 "requests_sha256": file_sha256(b / "headset" / "requests.jsonl"),
                 "fixtures_sha256": file_sha256(b / "headset" / "fixtures-written.jsonl")}
        ident = {"record_type": "a25_replay_identity", "results": stamp, "runner": "desktop",
                 "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "bundle": dict({"policy_id": hm["policy_id"], "requests": len(recs), "fixtures": len(fixes)}, **files),
                 "model": {"path": str(model), "bytes": model.stat().st_size, "sha256": accepted_sha256},
                 "requested": dict(REQUESTED),
                 "runtime": {"n_ctx": info[0], "n_vocab": info[1], "n_seq": info[2], "flags": info[3], "version": native.version(),
                             "system_info": native.system_info(), "load_ms": load_ms},
                 "memory_kb": {"before_load_rss": before[0], "before_load_peak": before[1], "after_load_rss": after[0],
                               "after_load_peak": after[1]},
                 "capture": {"scope": "process-wide stderr (file descriptor 2), redirected around se_load_ex only",
                             "verbose": "on during the load only", "file": "startup-log.txt",
                             "bytes": (staging / "startup-log.txt").stat().st_size if (staging / "startup-log.txt").is_file() else -1,
                             "error": cap_error},
                 "code": {f"grounding/quest/{c.name}": file_sha256(c) for c in sorted(Path(__file__).parent.glob("*.py"))},
                 "host": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                          "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "library": str(library),
                          "library_sha256": file_sha256(library) if library.is_file() else None,
                          "build_record_sha256": RI.sha256_hex(build) if build else None,
                          "build_record": json.loads(build) if build else None}}
        (staging / "identity.json").write_bytes(encode_json(ident))
        if info[1] != hm["vocab_size"]:
            raise RuntimeError(f"the model's vocabulary ({info[1]}) is not the bundle's ({hm['vocab_size']})")
        fixes_ok = 0
        with open(staging / "fixtures.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for rec in fixes:
                chk, line = _input_line(rec, cons, native)
                as_expected = (chk["outcome"], chk["reason"]) == (rec["expected_outcome"], rec["expected_reason"])
                fixes_ok += as_expected
                f.write(json.dumps({"record_type": "a25_replay_fixture", "request_id": rec["request_id"], "kind": rec["kind"],
                                    "expected_outcome": rec["expected_outcome"], "expected_reason": rec["expected_reason"],
                                    "outcome": chk["outcome"], "check": chk["check"], "reason": chk["reason"],
                                    "detail": chk["detail"], "runtime_tokens": line["runtime_tokens"], "evaluated": False,
                                    "as_expected": as_expected}) + "\n")
        written = 0
        with open(staging / "results.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for k, rec in enumerate(recs):
                progress(f"{rec['request_id']} ({k + 1}/{len(recs)}), U, R and P")
                t0 = time.perf_counter()
                chk, line = _input_line(rec, cons, native)
                n = len(rec["token_ids"])
                res = {"record_type": "a25_replay_result", "request_id": rec["request_id"], "kind": rec["kind"], "input": line,
                       "boundaries": {"n": n, "keep_before_last_token": rec["keep_before_last_token"],
                                      "keep_before_command_line": rec["keep_before_command_line"]}, "paths": {}}
                if chk["outcome"] == "passed":
                    res["paths"] = _paths(native, rec, hm["ask_code"], info[1])
                res["ms"] = (time.perf_counter() - t0) * 1000.0
                f.write(json.dumps(res) + "\n")
                f.flush()
                written += 1
        done = {"record_type": "a25_replay_done", "results": stamp, "runner": "desktop", "requests": len(recs), "written": written,
                "fixtures": len(fixes), "fixtures_as_expected": fixes_ok, "self_checks_passed": all_ok,
                "total_ms": (time.perf_counter() - total) * 1000.0, "end_rss_kb": native.memory_kb(False),
                "end_peak_kb": native.memory_kb(True), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (staging / "done.json").write_bytes(encode_json(done))
        native.free()
        publish(staging, out)
    except OSError as e:
        native.free()
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        native.free()
        failed = out.parent / f"{out.name}.failed"
        if not failed.exists():
            try:
                publish(staging, failed)   # kept as evidence, never published as a finished result
            except OSError:
                pass                       # the staging folder then stays where it is, still evidence
        raise
    done["folder"] = str(out)
    return done
