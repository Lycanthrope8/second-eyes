"""A2.5 delivery 1: the bounded batch-arithmetic diagnostic (`notes/phases/A2.5_d1_replay.md`, sections 11 and 12).

ChatGPT approved it on 10 October with modifications. It is a diagnostic, not an approved project decision.

**What runs.** Exactly four request/m pairs. Each gets a fresh context on run r010's host DLL and the accepted GGUF,
with the same settings, tokens and mappings as r013:

1. **U:** the complete prompt, keep 0.
2. **Recompute-m:** keep U's first n - m positions, and evaluate the last m in one batch.
3. **U':** the complete prompt again, keep 0.

m = 63 never runs from a cache modified by m = 64, because each pair has its own context. Every cache count is
verified. Full-row SHA-256 hashes and offered scores are kept.

**The comparison** checks the controls before any interpretation: U' must equal U, and each U must equal r013's U for
that request. Only then is each pair's prediction compared with what happened.

**The interpretations** are pre-registered:

- predictions that hold support the batching hypothesis on these cases; they prove no particular kernel explains every
  discrepancy;
- m = 63 equal to U does not establish that the one-token path is the sole cause.
"""
from __future__ import annotations

import datetime
import json
import shutil
import tempfile
import time
from pathlib import Path

from ..evaluation.iref_vla import output
from ..evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl, issue
from . import replay_inputs as RI
from .publish import publish
from .replay_bundle import verify_replay_bundle
from .replay_desktop import ACCEPTED_MODEL_SHA256, DEFAULT_LIBRARY, DEFAULT_MODEL, REQUESTED, Native, captured_load
from .replay_repeat import ACCEPTED_HOST_DLL_SHA256, _eval, _score
from .runtime_identity import file_sha256

# (request, m, prediction under the dispatch model at the pin): section 11's table, as approved
PAIRS = (("r0004", 2, "equal"), ("r0329", 2, "equal"), ("r0001", 64, "equal"), ("r0001", 63, "differ"))
PATHS = ("U", "Rm", "Uprime")


def _fail(where, msg):
    raise EvaluationInputError([issue(str(where), "E_BATCH_DIAGNOSTIC", msg)])


def _context(native, rec, m, ask, n_vocab):
    """U, recompute-m, then U' in one fresh context. Stops at the first failed evaluation or cache count."""
    t, n = rec["token_ids"], len(rec["token_ids"])
    plan = {"U": (0, n, 0, n), "Rm": (n - m, m, n - m, n), "Uprime": (0, n, 0, n)}
    paths = {}
    for name in PATHS:
        off, cnt, keep, expect = plan[name]
        ok, e = _eval(native, t, off, cnt, keep, expect)
        paths[name] = dict({"evals": [e]}, **(_score(native, rec, ask, n_vocab) if ok else {"error": "eval_failed_or_cache_mismatch"}))
        if not ok or paths[name]["error"] is not None:
            return paths, False
    return paths, True


def run_batch_diagnostic(*, bundle, out, model=None, library=None, native=None, progress=print, pairs=PAIRS,
                         accepted_sha256=ACCEPTED_MODEL_SHA256, accepted_dll_sha256=ACCEPTED_HOST_DLL_SHA256) -> dict:
    """The fixed run. `pairs` exists for tests; the command line always uses PAIRS."""
    out = output.refuse_existing(out)
    b = Path(bundle)
    bad = verify_replay_bundle(b)
    if bad:
        _fail(b, "the bundle does not read back: " + "; ".join(bad[:3]))
    model = Path(model) if model is not None else DEFAULT_MODEL
    if not model.is_file() or file_sha256(model) != accepted_sha256:
        _fail(model, "not the accepted model file (full SHA-256)")
    library = Path(library) if library is not None else DEFAULT_LIBRARY
    if native is None:
        if not library.is_file() or file_sha256(library) != accepted_dll_sha256:
            _fail(library, "not run 20261008_A2_r010's host DLL (SHA-256 aa241e18...)")
        native = Native(library)
    hm = json.loads((b / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))
    recs = {json.loads(x)["request_id"]: json.loads(x)
            for x in (b / "headset" / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    for rid, m, _ in pairs:
        if rid not in recs:
            _fail(b, f"the bundle does not hold {rid}")
        if not 1 <= m < len(recs[rid]["token_ids"]):
            _fail(b, f"m = {m} does not fit {rid}'s {len(recs[rid]['token_ids'])} tokens")
    cons = {"context_limit_tokens": hm["context_limit_tokens"], "continuation_tokens": hm["continuation_tokens"],
            "vocab_size": hm["vocab_size"], "object_codes": hm["object_codes"], "ask_code": hm["ask_code"],
            "ask_target": hm["ask_target"], "code_token_ids": dict(zip(hm["choice_codes"], hm["choice_token_ids"]))}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    t_all, loads, lines = time.perf_counter(), [], []
    try:
        for k, (rid, m, prediction) in enumerate(pairs, 1):
            rec = recs[rid]
            progress(f"pair {k} of {len(pairs)}: {rid}, m = {m}; a fresh context: load, input checks, U, recompute-{m}, U'")
            log = staging / f"startup-{k}-{rid}-m{m}.txt"
            ok, load_ms, cap_error = captured_load(native, model, log)
            if not ok:
                raise RuntimeError(f"pair {k}: the model did not load: {native.last_error()}")
            info = native.info()
            loads.append({"pair": k, "request_id": rid, "m": m, "load_ms": load_ms, "n_ctx": info[0], "n_vocab": info[1],
                          "n_seq": info[2], "flags": info[3], "capture_file": log.name, "capture_error": cap_error})
            seen = {}
            chk = RI.check_inputs(rec, cons, tokenize=lambda text: seen.setdefault("ids", native.tokenize(text.encode("utf-8"))))
            line = {"record_type": "a25_batch_diagnostic_result", "pair": k, "request_id": rid, "m": m, "prediction": prediction,
                    "input": {"outcome": chk["outcome"], "check": chk["check"], "reason": chk["reason"],
                              "tokens_equal_frozen": seen.get("ids") == rec["token_ids"], "token_ids_sha256": rec["token_ids_sha256"],
                              "mapping_sha256": rec["mapping_sha256"]},
                    "n": len(rec["token_ids"]), "u_last_batch": len(rec["token_ids"]) - 512 * ((len(rec["token_ids"]) - 1) // 512)}
            if chk["outcome"] != "passed":
                lines.append(line)
                raise RuntimeError(f"pair {k}: the input checks stopped at {chk['check']} ({chk['reason']})")
            line["paths"], ok = _context(native, rec, m, hm["ask_code"], info[1])
            line["path_order"] = [x for x in PATHS if x in line["paths"]]
            lines.append(line)
            native.free()
            if not ok:
                raise RuntimeError(f"pair {k}: an evaluation failed or a cache count differed; stopped")
        ident = {"record_type": "a25_batch_diagnostic_identity", "results": out.name, "pairs": [list(p) for p in pairs],
                 "paths": list(PATHS), "requested": dict(REQUESTED),
                 "threads": "requested and passed to the loader; not read back by the runtime",
                 "bundle": {"manifest_sha256": file_sha256(b / "manifest.json"), "requests_sha256": file_sha256(b / "headset" / "requests.jsonl")},
                 "model": {"path": str(model), "sha256": accepted_sha256}, "loads": loads,
                 "library": {"path": str(library), "sha256": file_sha256(library) if library.is_file() else None},
                 "runtime": {"version": native.version(), "system_info": native.system_info()},
                 "code": {f"grounding/quest/{c.name}": file_sha256(c) for c in sorted(Path(__file__).parent.glob("*.py"))}}
        (staging / "identity.json").write_bytes(encode_json(ident))
        (staging / "results.jsonl").write_bytes(encode_jsonl(lines))
        done = {"record_type": "a25_batch_diagnostic_done", "results": out.name, "contexts": len(lines), "expected_contexts": len(pairs),
                "path_evaluations": sum(len(x.get("paths", {})) for x in lines), "total_ms": (time.perf_counter() - t_all) * 1000.0,
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
            failed = out.parent / f"{out.name}.failed"
            if not failed.exists():
                publish(staging, failed)
        except OSError:
            pass
        raise
    done["folder"] = str(out)
    return done


def compare_batch_diagnostic(*, results, r013, out) -> dict:
    """Controls first (U' = U; U = r013's U for the request; cache counts; input checks), then each pair's prediction."""
    out = output.refuse_existing(out)
    res = Path(results)
    lines = [json.loads(x) for x in (res / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    r013_rows = [json.loads(x) for x in (Path(r013) / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    r013_u = {}
    for r in r013_rows:
        r013_u.setdefault(r["request_id"], set()).add(r["paths"]["U"]["row_sha256"])
    pairs, problems = [], []
    for x in lines:
        p = x.get("paths", {})
        complete = all(name in p and p[name].get("error") is None for name in PATHS)
        counts = complete and all(e["n_cached"] == e["expected_n_cached"] and e["status"] == 0 for v in p.values() for e in v["evals"])
        u = p.get("U", {}).get("row_sha256")
        c = {"input_passed": x["input"]["outcome"] == "passed" and x["input"]["tokens_equal_frozen"], "complete": complete,
             "cache_counts": bool(counts), "uprime_equals_u": complete and p["Uprime"]["row_sha256"] == u,
             "u_equals_r013": u is not None and r013_u.get(x["request_id"]) == {u}}
        observed = None if not complete else ("equal" if p["Rm"]["row_sha256"] == u else "differ")
        offered = lambda name: [o["logit"] for o in p[name]["offered"]] if complete else []  # noqa: E731
        pairs.append({"pair": x["pair"], "request_id": x["request_id"], "m": x["m"], "u_last_batch": x["u_last_batch"],
                      "prediction": x["prediction"], "observed": observed, "held": observed == x["prediction"], "controls": c,
                      "choice_u": p.get("U", {}).get("choice_code"), "choice_rm": p.get("Rm", {}).get("choice_code"),
                      "max_abs_offered_difference": max((abs(a - b) for a, b in zip(offered("Rm"), offered("U"))), default=None)})
        problems += [f"pair {x['pair']} ({x['request_id']}, m = {x['m']}): {k} failed" for k, v in c.items() if not v]
    planned = json.loads((res / "identity.json").read_text(encoding="utf-8"))["pairs"]
    controls_ok = not problems and [(q["request_id"], q["m"]) for q in pairs] == [(r, m) for r, m, _ in planned]
    if not controls_ok:
        reading = ["The controls did not all hold, so no interpretation is drawn: " + "; ".join(problems or ["a pair is missing"]) + "."]
    else:
        eq_fail = [q for q in pairs if q["prediction"] == "equal" and not q["held"]]
        df_fail = [q for q in pairs if q["prediction"] == "differ" and not q["held"]]
        reading = []
        if not eq_fail and not df_fail:
            reading.append("All four predictions held. On these cases, the results support the batching hypothesis: a recomputed "
                           "final batch reproduced U bit for bit where the source predicts the same kernel classes, and differed "
                           "where it predicts another. This is not proof that a particular kernel explains every discrepancy.")
        if eq_fail:
            reading.append("Batch dependence within a predicted kernel class on these cases (" + ", ".join(
                f"{q['request_id']}, m = {q['m']}" for q in eq_fail) + "): the dispatch model is incomplete. Stop and report.")
        if df_fail:
            reading.append("m = 63 reproduced U: the attention change at 64 queries did not alter this row. That does not "
                           "establish that the one-token path is the sole cause of the R/U differences.")
    summary = {"format_version": 1, "record_type": "a25_batch_diagnostic_comparison", "results": res.name, "r013": Path(r013).name,
               "controls_hold": controls_ok, "pairs": pairs, "reading": reading,
               "scope": "a diagnostic on four fixed request/m pairs on the desktop; not an approved decision and not numerical acceptance"}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        (staging / "summary.json").write_bytes(encode_json(summary))
        rows = "\n".join(f"| {q['request_id']} | {q['m']} | {q['u_last_batch']} | {q['prediction']} | {q['observed']} | "
                         f"{'yes' if q['held'] else 'no'} | {q['choice_u']} / {q['choice_rm']} | {q['max_abs_offered_difference']} | "
                         f"{'all hold' if all(q['controls'].values()) else ', '.join(k for k, v in q['controls'].items() if not v)} |"
                         for q in pairs)
        (staging / "report.md").write_text("# The bounded batch-arithmetic diagnostic: comparison\n\n"
            "| Request | m | U's last batch | Predicted (Rm vs U) | Observed | Held | Choice U / Rm | Max offered difference | Controls |\n"
            "|---|---|---|---|---|---|---|---|---|\n" + rows + "\n\n" + "\n\n".join(reading) + f"\n\nScope: {summary['scope']}.\n", encoding="utf-8")
        publish(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    return dict(summary, folder=str(out))
