"""A2.5 delivery 1: the desktop repeatability and context-history diagnostic.

Scope: ChatGPT's proposal of 8 October, prepared here; adoption is the project lead's. This is diagnostic evidence, not
a replacement acceptance set: replay acceptance stays as recorded, and D56/D101 are unchanged.

**The fixed run** (`run_repeat`, command `replay-repeat`):

- Requests: r0001, r0004 and r0329, in bundle order, with the same frozen bytes, tokens and mappings, the same GGUF
  (full SHA-256), the same host DLL as run 20261008_A2_r010 (its SHA-256), and the same requested settings.
- For each request, a freshly loaded context runs the input checks (native tokenization of the whole prompt included),
  then U, R, P and U' (U' is a second full evaluation with keep 0 in the same context). The context is freed, and the
  same sequence runs once more in another fresh context.
- Every evaluation's slice, keep, status and cache count is kept, and every scored path keeps its offered raw logits,
  log-probabilities, shares, decision and the SHA-256 of its full float32 row. Each load's startup log is captured.
- It stops after these six sequences: no retries, no other cases, no parameter changes. Any failure stops the run, and
  what it wrote is kept as `<folder>.failed`.

**The analysis** (`compare_repeat`, command `repeat-compare`):

- D101's five comparisons on each U/R/P sequence, as in step 3.
- Descriptive comparisons, with the same choice, ranking and distribution calculations and no pass or fail: U' against U
  in the same context, and each path in one context against the same path in the other. Raw equality, of the offered
  logits and of the full row's SHA-256, is reported as a fact, not as a threshold.
- Cautious statements on repeatability, context-history dependence and stable differences between paths. None of them
  alone establishes a particular kernel or cache defect.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

from ..evaluation.iref_vla import output
from ..evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl, issue, runtime, sha256
from . import replay_inputs as RI
from .publish import publish
from .replay_bundle import verify_replay_bundle
from .replay_compare import COMPARISONS, choose, compare, f32, shares
from .replay_desktop import ACCEPTED_MODEL_SHA256, DEFAULT_LIBRARY, DEFAULT_MODEL, HOST_BUILD_RECORD, REQUESTED, Native, captured_load
from .runtime_identity import file_sha256

REQUESTS = ("r0001", "r0004", "r0329")
REPEATS = 2
PATHS = ("U", "R", "P", "Uprime")
ACCEPTED_HOST_DLL_SHA256 = "aa241e18d48ad17006892f25f86bdf7c968f8c0e686e5d4eb5646b75e71bdedd"   # run 20261008_A2_r010
SCOPE = "ChatGPT's proposal of 8 October; diagnostic evidence, not acceptance; adoption is the project lead's"


def _row_sha(row) -> str:
    return hashlib.sha256(struct.pack(f"<{len(row)}f", *row)).hexdigest()


def _eval(native, t, offset, count, keep, expect):
    t0 = time.perf_counter()
    rc = native.eval(t[offset:offset + count], keep)
    ms = (time.perf_counter() - t0) * 1000.0
    cached = native.n_cached()
    ok = rc == 0 and cached == expect
    return ok, {"offset": offset, "count": count, "keep": keep, "status": rc, "error": native.last_error() if rc != 0 else None,
                "n_cached": cached, "expected_n_cached": expect, "ms": ms}


def _score(native, rec, ask, n_vocab):
    row = native.logits(n_vocab)
    bad = [k for k, x in enumerate(row) if not math.isfinite(x)]
    if len(row) != n_vocab or bad:
        return {"logits_returned": len(row), "error": "non_finite_output" if bad else "short_row", "row_sha256": None}
    offered = [row[i] for i in rec["code_token_ids"]]
    code, reason, tied = choose(rec["codes"], offered, ask)
    return {"logits_returned": len(row), "error": None, "row_sha256": _row_sha(row), "choice_code": code,
            "selection_reason": reason, "tied_codes": tied,
            "offered": [{"code": c, "target": t, "token_id": i, "logit": x, "log_prob": native.logprob(i), "restricted_share": s}
                        for c, t, i, x, s in zip(rec["codes"], rec["targets"], rec["code_token_ids"], offered, shares(offered))]}


def _sequence(native, rec, ask, n_vocab):
    """U, R, P, then U' in one context. Returns (paths, ok); stops at the first failed evaluation."""
    t, n, k = rec["token_ids"], len(rec["token_ids"]), rec["keep_before_command_line"]
    plan = {"U": [(0, n, 0, n)], "R": [(n - 1, 1, n - 1, n)], "P": [(0, k, 0, k), (k, n - k, k, n)], "Uprime": [(0, n, 0, n)]}
    paths = {}
    for name in PATHS:
        evals, ok = [], True
        for off, cnt, keep, expect in plan[name]:
            ok, e = _eval(native, t, off, cnt, keep, expect)
            evals.append(e)
            if not ok:
                break
        paths[name] = dict({"evals": evals}, **(_score(native, rec, ask, n_vocab) if ok else {"error": "eval_failed_or_cache_mismatch"}))
        if not ok or paths[name]["error"] is not None:
            return paths, False
    return paths, True


def run_repeat(*, bundle, out, model=None, library=None, native=None, progress=print, requests=REQUESTS,
               accepted_sha256=ACCEPTED_MODEL_SHA256, accepted_dll_sha256=ACCEPTED_HOST_DLL_SHA256) -> dict:
    """The fixed run. `requests` exists for tests; the command line always uses REQUESTS."""
    out = output.refuse_existing(out)
    b = Path(bundle)
    bad = verify_replay_bundle(b)
    if bad:
        raise EvaluationInputError([issue(str(b), "E_REPLAY_REPEAT", "the bundle does not read back: " + "; ".join(bad[:3]))])
    model = Path(model) if model is not None else DEFAULT_MODEL
    if not model.is_file() or file_sha256(model) != accepted_sha256:
        raise EvaluationInputError([issue(str(model), "E_REPLAY_REPEAT", "not the accepted model file (full SHA-256)")])
    library = Path(library) if library is not None else DEFAULT_LIBRARY
    if native is None:
        if not library.is_file() or file_sha256(library) != accepted_dll_sha256:
            raise EvaluationInputError([issue(str(library), "E_REPLAY_REPEAT",
                                              "not run 20261008_A2_r010's host DLL (SHA-256 aa241e18...)")])
        native = Native(library)
    hm = json.loads((b / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))
    recs = {json.loads(x)["request_id"]: json.loads(x)
            for x in (b / "headset" / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    order = [rid for rid in recs if rid in requests]
    if order != list(requests):
        raise EvaluationInputError([issue(str(b), "E_REPLAY_REPEAT", f"the bundle does not hold {tuple(requests)} in this order")])
    cons = {"context_limit_tokens": hm["context_limit_tokens"], "continuation_tokens": hm["continuation_tokens"],
            "vocab_size": hm["vocab_size"], "object_codes": hm["object_codes"], "ask_code": hm["ask_code"],
            "ask_target": hm["ask_target"], "code_token_ids": dict(zip(hm["choice_codes"], hm["choice_token_ids"]))}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    t_all = time.perf_counter()
    loads, lines = [], []
    try:
        for rid in order:
            rec = recs[rid]
            for rep in range(1, REPEATS + 1):
                progress(f"{rid}, context {rep} of {REPEATS}: load, input checks, U, R, P, U'")
                log = staging / f"startup-{rid}-{rep}.txt"
                ok, load_ms, cap_error = captured_load(native, model, log)
                if not ok:
                    raise RuntimeError(f"{rid} context {rep}: the model did not load: {native.last_error()}")
                info = native.info()
                loads.append({"request_id": rid, "repeat": rep, "load_ms": load_ms, "n_ctx": info[0], "n_vocab": info[1],
                              "n_seq": info[2], "flags": info[3], "capture_file": log.name,
                              "capture_bytes": log.stat().st_size if log.is_file() else -1, "capture_error": cap_error})
                seen = {}
                chk = RI.check_inputs(rec, cons, tokenize=lambda text: seen.setdefault("ids", native.tokenize(text.encode("utf-8"))))
                line = {"record_type": "a25_repeat_result", "request_id": rid, "repeat": rep,
                        "input": {"outcome": chk["outcome"], "check": chk["check"], "reason": chk["reason"],
                                  "runtime_tokens": len(seen["ids"]) if "ids" in seen else -1,
                                  "tokens_equal_frozen": seen.get("ids") == rec["token_ids"],
                                  "prompt_sha256": rec["prompt_sha256"], "token_ids_sha256": rec["token_ids_sha256"],
                                  "mapping_sha256": rec["mapping_sha256"]},
                        "boundaries": {"n": len(rec["token_ids"]), "keep_before_command_line": rec["keep_before_command_line"]}}
                if chk["outcome"] != "passed":
                    lines.append(line)
                    raise RuntimeError(f"{rid} context {rep}: the input checks stopped at {chk['check']} ({chk['reason']})")
                line["paths"], ok = _sequence(native, rec, hm["ask_code"], info[1])
                line["path_order"] = [x for x in PATHS if x in line["paths"]]   # execution order (the encoder sorts keys)
                lines.append(line)
                native.free()
                if not ok:
                    raise RuntimeError(f"{rid} context {rep}: an evaluation failed or a cache count differed; stopped")
        build = HOST_BUILD_RECORD.read_bytes() if HOST_BUILD_RECORD.is_file() else None
        ident = {"record_type": "a25_repeat_identity", "results": out.name, "scope": SCOPE, "requests": list(requests),
                 "repeats": REPEATS, "paths": list(PATHS), "requested": dict(REQUESTED),
                 "threads": "requested and passed to the loader; not read back by the runtime",
                 "bundle": {"manifest_sha256": file_sha256(b / "manifest.json"),
                            "headset_manifest_sha256": file_sha256(b / "headset" / "replay-manifest.json"),
                            "requests_sha256": file_sha256(b / "headset" / "requests.jsonl")},
                 "model": {"path": str(model), "sha256": accepted_sha256}, "loads": loads,
                 "runtime": {"version": native.version(), "system_info": native.system_info()},
                 "library": {"path": str(library), "sha256": file_sha256(library) if library.is_file() else None},
                 "build_record_sha256": RI.sha256_hex(build) if build else None,
                 "code": {f"grounding/quest/{c.name}": file_sha256(c) for c in sorted(Path(__file__).parent.glob("*.py"))},
                 "host": {"platform": sys.platform, "python": sys.version.split()[0]}}
        (staging / "identity.json").write_bytes(encode_json(ident))
        (staging / "results.jsonl").write_bytes(encode_jsonl(lines))
        done = {"record_type": "a25_repeat_done", "results": out.name, "sequences": len(lines),
                "expected_sequences": len(requests) * REPEATS, "total_ms": (time.perf_counter() - t_all) * 1000.0,
                "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (staging / "done.json").write_bytes(encode_json(done))
        publish(staging, out)
    except OSError as e:
        native.free()
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        native.free()
        try:
            (staging / "results.jsonl").write_bytes(encode_jsonl(lines))
            (staging / "loads.json").write_bytes(encode_json(loads))
            failed = out.parent / f"{out.name}.failed"
            if not failed.exists():
                publish(staging, failed)
        except OSError:
            pass
        raise
    done["folder"] = str(out)
    return done


def _logits(path):
    return [f32(o["logit"]) for o in path["offered"]]


def _describe(codes, a, b, ask, row_a, row_b):
    c = compare(codes, a, b, ask, False)
    return {"same_best": c["choice_a"] == c["choice_b"], "choice_a": c["choice_a"], "choice_b": c["choice_b"],
            "changed_pairs": c["changed_pairs"], "tvd": c["tvd"],
            "offered_logits_identical": all(x == y for x, y in zip(a, b)),
            "full_row_identical": row_a is not None and row_a == row_b}


def compare_repeat(*, bundle, results, out, requests=REQUESTS) -> dict:
    out = output.refuse_existing(out)
    b, r = Path(bundle), Path(results)
    recs = {json.loads(x)["request_id"]: json.loads(x)
            for x in (b / "headset" / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    refs = {json.loads(x)["request_id"]: json.loads(x)["reference"]
            for x in (b / "reference" / "references.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    ask = json.loads((b / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))["ask_code"]
    lines = [json.loads(x) for x in (r / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    problems = []
    want = [(rid, rep) for rid in requests for rep in range(1, REPEATS + 1)]
    try:
        done = json.loads((r / "done.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        done = None
    if not isinstance(done, dict) or done.get("sequences") != len(want) or done.get("results") != r.name:
        problems.append("completion: done.json is missing, unreadable, or does not name this folder and all sequences")
    if [(x.get("request_id"), x.get("repeat")) for x in lines] != want:
        problems.append(f"coverage: the sequences are not {want}")
    by = {(x["request_id"], x["repeat"]): x for x in lines}
    d101, desc = [], []
    for rid in requests:
        rec = recs[rid]
        n, k = len(rec["token_ids"]), rec["keep_before_command_line"]
        plan = {"U": [(0, n, 0, n)], "R": [(n - 1, 1, n - 1, n)], "P": [(0, k, 0, k), (k, n - k, k, n)], "Uprime": [(0, n, 0, n)]}
        codes = rec["codes"]
        got = {}
        for rep in range(1, REPEATS + 1):
            line = by.get((rid, rep))
            if line is None:
                continue
            inp = line.get("input") or {}
            if inp.get("outcome") != "passed" or not inp.get("tokens_equal_frozen"):
                problems.append(f"{rid} context {rep}: the input checks did not pass with the frozen tokenization")
                continue
            for name in PATHS:
                p = (line.get("paths") or {}).get(name) or {}
                ev = [(e.get("offset"), e.get("count"), e.get("keep"), e.get("status"), e.get("n_cached"))
                      for e in p.get("evals", [])]
                if ev != [(o, c, kp, 0, ex) for o, c, kp, ex in plan[name]] or p.get("error") is not None:
                    problems.append(f"{rid} context {rep} {name}: not the prescribed evaluations, or not scored")
                    continue
                if [(o["code"], o["target"], o["token_id"]) for o in p["offered"]] != list(zip(codes, rec["targets"], rec["code_token_ids"])):
                    problems.append(f"{rid} context {rep} {name}: the offered mapping differs from the bundle's")
                    continue
                got[(rep, name)] = (_logits(p), p.get("row_sha256"))
            ref = [s["logit"] for s in refs[rid]["scores"]]
            for a, bn, judge in COMPARISONS:
                if (rep, a) in got and (bn == "float32" or (rep, bn) in got):
                    other = ref if bn == "float32" else got[(rep, bn)][0]
                    c = compare(codes, got[(rep, a)][0], other, ask, judge)
                    d101.append(dict({"request_id": rid, "repeat": rep, "comparison": f"{a} vs {bn}"}, **c))
            if (rep, "Uprime") in got and (rep, "U") in got:
                desc.append(dict({"request_id": rid, "kind": "U' vs U, same context", "repeat": rep},
                                 **_describe(codes, got[(rep, "Uprime")][0], got[(rep, "U")][0], ask, got[(rep, "Uprime")][1], got[(rep, "U")][1])))
        for name in PATHS:
            if (1, name) in got and (2, name) in got:
                desc.append(dict({"request_id": rid, "kind": f"{name}, context 1 vs context 2"},
                                 **_describe(codes, got[(1, name)][0], got[(2, name)][0], ask, got[(1, name)][1], got[(2, name)][1])))
        for name in ("R", "P"):   # is each path's difference from U the same in both contexts?
            if all((rep, x) in got for rep in (1, 2) for x in (name, "U")):
                d1 = [x - y for x, y in zip(got[(1, name)][0], got[(1, "U")][0])]
                d2 = [x - y for x, y in zip(got[(2, name)][0], got[(2, "U")][0])]
                desc.append({"request_id": rid, "kind": f"{name} minus U, context 1 vs context 2",
                             "offered_differences_identical": d1 == d2, "max_abs_difference_ctx1": max(map(abs, d1)),
                             "max_abs_difference_ctx2": max(map(abs, d2))})
    rows_rep = [x for x in desc if x["kind"].endswith("context 1 vs context 2") and "full_row_identical" in x]
    rows_hist = [x for x in desc if x["kind"].startswith("U' vs U")]
    rows_path = [x for x in desc if "minus U" in x["kind"]]
    facts = {"repeat_pairs": len(rows_rep), "repeat_pairs_full_row_identical": sum(x["full_row_identical"] for x in rows_rep),
             "uprime_pairs": len(rows_hist), "uprime_full_row_identical": sum(x["full_row_identical"] for x in rows_hist),
             "path_difference_pairs": len(rows_path),
             "path_differences_identical": sum(x["offered_differences_identical"] for x in rows_path),
             "d101_comparisons": len(d101), "d101_failures": sum(not x["pass"] for x in d101)}
    summary = {"format_version": 1, "record_type": "a25_repeat_comparison", "scope": SCOPE, "results": r.name,
               "problems": problems, "facts": facts, "statements": statements(facts, problems),
               "d101": [{k: x.get(k) for k in ("request_id", "repeat", "comparison", "choice_a", "choice_b", "tvd", "failed",
                                               "changed_pairs", "pass")} for x in d101],
               "descriptive": desc}
    files = {"summary.json": encode_json(summary), "report.md": render(summary).encode("utf-8")}
    manifest = {"format_version": 1, "record_type": "a25_repeat_comparison_manifest",
                "inputs": {"bundle_manifest_sha256": file_sha256(b / "manifest.json"),
                           "results_files": {p.name: sha256(p.read_bytes()) for p in sorted(r.iterdir()) if p.is_file()}},
                "files": {k: sha256(v) for k, v in files.items()},
                "code": {f"grounding/quest/{p.name}": sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "runtime": runtime()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for name, data in files.items():
            (staging / name).write_bytes(data)
        (staging / "manifest.json").write_bytes(encode_json(manifest))
        publish(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    return summary


def statements(f, problems) -> list:
    """Cautious readings of the facts. None alone establishes a particular kernel or cache defect."""
    if problems:
        return ["The records have problems (listed); no reading is offered until they are resolved."]
    s = []
    if f["repeat_pairs"]:
        if f["repeat_pairs_full_row_identical"] == f["repeat_pairs"]:
            s.append(f"Every path gave a bit-identical full row in both fresh contexts ({f['repeat_pairs']} of {f['repeat_pairs']} "
                     "pairs): this supports repeatability of each path on this host, for these requests.")
        else:
            s.append(f"Only {f['repeat_pairs_full_row_identical']} of {f['repeat_pairs']} path pairs were bit-identical across fresh "
                     "contexts: this does not support repeatability; the same inputs, settings and schedule gave different rows.")
    if f["uprime_pairs"]:
        if f["uprime_full_row_identical"] == f["uprime_pairs"]:
            s.append(f"U' equalled U bit for bit in every context ({f['uprime_pairs']} of {f['uprime_pairs']}): no sign that "
                     "the evaluations between them (R, P) changed a later full evaluation.")
        else:
            s.append(f"U' differed from U in {f['uprime_pairs'] - f['uprime_full_row_identical']} of {f['uprime_pairs']} "
                     "contexts: a full evaluation depended on what the same context had evaluated before, which points to "
                     "context-history dependence in this schedule.")
    if f["path_difference_pairs"]:
        if f["path_differences_identical"] == f["path_difference_pairs"]:
            s.append("R and P differed from U by exactly the same offered-logit amounts in both contexts: this supports "
                     "stable differences between evaluation paths, rather than run-to-run variation.")
        else:
            s.append(f"In {f['path_difference_pairs'] - f['path_differences_identical']} of {f['path_difference_pairs']} cases, "
                     "a path's difference from U was not the same in both contexts.")
    s.append("None of these readings alone establishes a particular kernel or cache defect.")
    return s


def render(s) -> str:
    f = s["facts"]
    L = ["# Repeatability and context-history diagnostic (A2.5 delivery 1)", "", f"Scope: {s['scope']}.", "",
         f"Results `{s['results']}`: {f['d101_failures']} of {f['d101_comparisons']} D101 comparisons fail on the two "
         "U/R/P sequences of each request (diagnostic; replay acceptance is unchanged).", "", "Readings:", ""]
    L += [f"- {x}" for x in s["statements"]]
    L += ["", "| Request | Comparison | Same best | Changed pairs | TVD | Offered logits identical | Full row identical |",
          "|---|---|---|---|---|---|---|"]
    for d in s["descriptive"]:
        if "tvd" in d:
            tag = d["kind"] + (f" {d['repeat']}" if "repeat" in d else "")
            L.append(f"| {d['request_id']} | {tag} | {'yes' if d['same_best'] else 'no'} | {', '.join(d['changed_pairs']) or '-'} | "
                     f"{d['tvd']:.10f} | {'yes' if d['offered_logits_identical'] else 'no'} | {'yes' if d['full_row_identical'] else 'no'} |")
    L += ["", "| Request | Repeat | D101 comparison | Pass | Failed | TVD |", "|---|---|---|---|---|---|"]
    for x in s["d101"]:
        L.append(f"| {x['request_id']} | {x['repeat']} | {x['comparison']} | {'yes' if x['pass'] else 'no'} | "
                 f"{', '.join(x['failed']) or '-'} | {x['tvd']:.10f} |")
    if s["problems"]:
        L += ["", "Problems:", ""] + [f"- {p}" for p in s["problems"]]
    return "\n".join(L) + "\n"
