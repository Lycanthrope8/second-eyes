"""A2.3e costs (D96): what each stage costs, on PC hardware. Not Quest figures.

Two parts. From the existing artifacts, without rerunning anything: prompt tokens per model, view and format with the
paired format difference, and the uncached forward timings, load times and peak GPU memory of runs r006 and r007. Then,
measured here, the deterministic stages for 16 answer-blind parent/view cases (both formats), one warm-up and five
measured repetitions each: relation construction, the rest of serialization, prompt assembly and tokenization with
each tokenizer. Relation construction is timed by wrapping the serializer's relation-table builders from outside for
the duration of the measurement; the accepted code is not changed. Disk reads and artifact validation are timed
separately and never inside the stage timings. Every rebuilt document, prompt and token list must equal the frozen
one before its timings count. Percentiles use the nearest rank: the sorted value at zero-based index ceil(p x n) - 1
(corrected on A2.3's closure; summaries made before it record the earlier index round(p x (n - 1)) and still verify).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl, issue,
                                             runtime, sha256, strict_json)
from ...resolution.validate import canonical_sha256
from ...serialization import serialize
from ...serialization import serializer as SER
from ..iref_vla.choices import build_prompt
from ..iref_vla.prepare import inside, read_bundle, rows_of
from ..iref_vla.protocol import load_protocol
from . import design as D
from .prepare import MODEL_KEYS, code_hashes, load_tokenizers, verify_compare_requests
from .run import verify_compare_results

CASE_SALT = "second-eyes/a23e/cases/v1"
CASE_COUNT = 16
OUTPUTS = ("measurements.jsonl", "report.md", "summary.json")


def _fail(code, problems):
    raise EvaluationInputError([issue(w, code, m) for w, m in problems])


def select_cases(rows, count=CASE_COUNT, salt=CASE_SALT) -> list:
    """Answer-blind: parent/view pairs sorted by SHA-256(salt + LF + parent + LF + view), ties by the pair; first `count`."""
    pairs = list(dict.fromkeys((r["parent_command_id"], r["view_id"]) for r in rows))
    pairs.sort(key=lambda pv: (hashlib.sha256((salt + "\n" + pv[0] + "\n" + pv[1]).encode("utf-8")).hexdigest(), pv))
    return pairs[:count]


RULES = {"nearest_rank_ceil": "sorted value at zero-based index ceil(p x n) - 1",
         "round_p_n_minus_1": "sorted value at zero-based index round(p x (n - 1)) (before the correction)"}
RULE = "nearest_rank_ceil"


def percentile_index(n: int, p: float, rule: str = RULE) -> int:
    if rule == "nearest_rank_ceil":
        return max(0, math.ceil(p * n) - 1)
    if rule == "round_p_n_minus_1":
        return int(round(p * (n - 1)))
    raise ValueError(rule)


def stats(values, rule: str = RULE) -> dict:
    v = sorted(values)
    if not v:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {"n": len(v), "median": statistics.median(v), "p95": v[percentile_index(len(v), 0.95, rule)], "min": v[0], "max": v[-1]}


def existing_costs(rows, runs, rule=RULE) -> dict:
    """Token counts and paired differences from the requests; timings and memory from the accepted runs."""
    stats_ = lambda v: stats(v, rule)  # noqa: E731
    out = {"tokens": {}, "paired_token_difference": {}, "forward_ms": {}, "runs": {}}
    for key in MODEL_KEYS:
        out["tokens"][key] = {v: {f: stats_([r["models"][key]["input_tokens"] for r in rows if r["view_id"] == v and r["format"] == f])
                                  for f in D.FORMATS} for v in D.VIEWS}
        out["paired_token_difference"][key] = {}
        for v in D.VIEWS:
            a = {r["parent_command_id"]: r["models"][key]["input_tokens"] for r in rows if r["view_id"] == v and r["format"] == D.FORMATS[0]}
            b = {r["parent_command_id"]: r["models"][key]["input_tokens"] for r in rows if r["view_id"] == v and r["format"] == D.FORMATS[1]}
            out["paired_token_difference"][key][v] = stats_([b[p] - a[p] for p in a])
    for key, run in runs.items():
        res, man, sessions = run["results"], run["manifest"], run["sessions"]
        out["forward_ms"][key] = {v: {f: stats_([x["timing_ms"]["forward"] for x, r in zip(res, rows)
                                                if r["view_id"] == v and r["format"] == f and x["timing_ms"]["forward"] is not None])
                                      for f in D.FORMATS} for v in D.VIEWS}
        out["runs"][key] = {"run_manifest_sha256": run["manifest_sha256"], "main_run_load_ms": [s["load_ms"] for s in sessions],
                            "peak_gpu_bytes": (man.get("model_info") or {}).get("peak_gpu_bytes"),
                            "device_name": (man.get("model_info") or {}).get("device_name")}
    return out


class RelationTimer:
    """Wraps the serializer's relation-table builders for the duration of a measurement; restores them afterwards."""
    NAMES = ("_relation_block", "_distance_block")

    def __init__(self):
        self.ns, self.calls, self._orig = 0, 0, {}

    def __enter__(self):
        for n in self.NAMES:
            f = getattr(SER, n)
            self._orig[n] = f

            def wrapped(*a, _f=f, **k):
                t0 = time.perf_counter_ns()
                try:
                    return _f(*a, **k)
                finally:
                    self.ns += time.perf_counter_ns() - t0
                    self.calls += 1
            setattr(SER, n, wrapped)
        return self

    def __exit__(self, *exc):
        for n, f in self._orig.items():
            setattr(SER, n, f)
        return False


def _hardware() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(), "timer": "time.perf_counter_ns"}


def summarize(measurements, existing, cases, hardware, rule=RULE) -> dict:
    stages = ("relation_ms", "serialization_ms", "serialize_total_ms", "assembly_ms")
    stats_ = lambda v: stats(v, rule)  # noqa: E731
    det = {}
    for v in D.VIEWS:
        for f in D.FORMATS:
            ms = [m for m in measurements if m["view_id"] == v and m["format"] == f]
            if not ms:
                continue
            det[f"{v}/{f}"] = {**{s: stats_([m[s] for m in ms]) for s in stages},
                               **{f"tokenize_ms/{k}": stats_([m["tokenize_ms"][k] for m in ms]) for k in MODEL_KEYS},
                               "relation_calls": stats_([m["relation_calls"] for m in ms]),
                               "io_ms": stats_([m["io_ms"] for m in ms]), "validation_ms": stats_([m["validation_ms"] for m in ms])}
    return {"format_version": 1, "record_type": "iref_compare_costs", "policy_id": D.POLICY_ID,
            "cases": [list(c) for c in cases], "case_rule": f"SHA-256 of UTF-8('{CASE_SALT}' + LF + parent + LF + view); first {CASE_COUNT}",
            "repetitions": {"warmup": 1, "measured": 5}, "existing": existing, "deterministic": det,
            "all_reconstructions_match": all(m["document_matches"] and m["prompt_matches"] and all(m["tokens_match"].values())
                                             for m in measurements),
            "hardware": hardware, "scope": "PC measurements; not Quest latency and not a combined runtime figure",
            **({"percentile_rule": rule} if rule != "round_p_n_minus_1" else {})}


def _f(x, unit="ms"):
    return "-" if x is None else (f"{x:,.2f} {unit}" if isinstance(x, float) else f"{x:,} {unit}".strip())


def render(s) -> str:
    e = s["existing"]
    L = ["# A2.3e costs (PC measurements, not Quest)", "",
         f"Hardware: {s['hardware']['processor']}, {s['hardware']['logical_cpus']} logical CPUs, {s['hardware']['platform']}, "
         f"Python {s['hardware']['python']}. Medians; p95 = " + RULES[s.get("percentile_rule", "round_p_n_minus_1")] + ".", "",
         "## Prompt tokens and uncached forward passes (existing artifacts)", "",
         "| View / format | Tokens median (p95) | 0.5B forward median / p95 | 7B forward median / p95 |", "|---|---|---|---|"]
    for v in D.VIEWS:
        for f in D.FORMATS:
            t = e["tokens"][MODEL_KEYS[0]][v][f]
            fw = [e["forward_ms"].get(k, {}).get(v, {}).get(f) for k in MODEL_KEYS]
            L.append(f"| {v} / {f} | {t['median']:,} ({t['p95']:,}) | "
                     + " | ".join("-" if x is None else f"{x['median']:.1f} / {x['p95']:.1f} ms" for x in fw) + " |")
    L += ["", "Paired token difference, augmented minus coordinates, same parent and view:", ""]
    for v in D.VIEWS:
        d = e["paired_token_difference"][MODEL_KEYS[0]][v]
        L.append(f"- {v}: min {d['min']:,}, median {d['median']:,}, p95 {d['p95']:,}, max {d['max']:,} tokens")
    def run_line(k, r):
        load = ", ".join(f"{x / 1000:.2f} s" for x in r["main_run_load_ms"])
        peak = "-" if r["peak_gpu_bytes"] is None else f"{r['peak_gpu_bytes']:,} bytes"
        return f"{k}: main-run load {load}, peak GPU {peak}"
    L += ["", "Runs: " + "; ".join(run_line(k, r) for k, r in e["runs"].items()), "",
          "## Deterministic stages (measured here; 16 cases x 5 repetitions per format)", "",
          "| View / format | Relation construction | Rest of serialization | Prompt assembly | Tokenize 0.5B | Tokenize 7B |",
          "|---|---|---|---|---|---|"]
    for k in [f"{v}/{f}" for v in D.VIEWS for f in D.FORMATS if f"{v}/{f}" in s["deterministic"]]:
        x = s["deterministic"][k]
        cell = lambda q: f"{q['median']:.3f} / {q['p95']:.3f} ms"  # noqa: E731
        L.append(f"| {k} | {cell(x['relation_ms'])} | {cell(x['serialization_ms'])} | {cell(x['assembly_ms'])} | "
                 f"{cell(x['tokenize_ms/' + MODEL_KEYS[0]])} | {cell(x['tokenize_ms/' + MODEL_KEYS[1]])} |")
    L += ["", f"Every rebuilt document, prompt and token list matched the frozen one: {s['all_reconstructions_match']}. Disk "
          "reads and record validation are timed apart (summary.json) and are not in these figures.", ""]
    return "\n".join(L) + "\n"


def run_costs(*, requests, bundle, relation_config, direction_config, tokenizer_small, tokenizer_large, small_run, large_run,
              out, tokenizers=None, repetitions=5) -> dict:
    out = output.refuse_existing(out)
    req, bundle = Path(requests), Path(bundle)
    bad = verify_compare_requests(req)
    for label, folder in (("0.5B run", small_run), ("7B run", large_run)):
        bad += [f"{label}: {m}" for m in verify_compare_results(folder, req)]
    if bad:
        _fail("E_COMPARE_INPUT", [("inputs", m) for m in bad[:20]])
    rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
    prompts = {x["request_id"]: x["prompt"] for x in rows_of("prompts.jsonl", (req / "prompts.jsonl").read_bytes(), "E_COMPARE_REQUESTS")}
    frozen = {k: {x["request_id"]: x["token_ids"] for x in rows_of(f"tokens/{k}.jsonl", (req / "tokens" / f"{k}.jsonl").read_bytes(),
                                                                       "E_COMPARE_REQUESTS")} for k in MODEL_KEYS}
    pol, proto = D.load_policy(req / "policy.json"), load_protocol(req / "protocol.json")
    runs = {}
    for key, folder in zip(MODEL_KEYS, (small_run, large_run)):
        f = Path(folder)
        runs[key] = {"results": rows_of("results.jsonl", (f / "results.jsonl").read_bytes(), "E_COMPARE_RESULTS"),
                     "sessions": rows_of("sessions.jsonl", (f / "sessions.jsonl").read_bytes(), "E_COMPARE_RESULTS"),
                     "manifest": json.loads((f / "manifest.json").read_bytes()), "manifest_sha256": sha256((f / "manifest.json").read_bytes())}
        if runs[key]["manifest"]["model_key"] != key:
            _fail("E_COMPARE_INPUT", [(str(folder), f"this run is of {runs[key]['manifest']['model_key']}, not {key}")])
    existing = existing_costs(rows, runs)
    src = read_bundle(bundle)
    rman = json.loads((req / "manifest.json").read_bytes())
    if sha256(src["raw"]["manifest.json"]) != rman["source_bundle"]["manifest_sha256"]:
        _fail("E_COMPARE_BUNDLE", [(str(bundle), "not the bundle the requests were prepared from")])
    cap = src["summary"].get("work_cap")
    cmap = strict_json("category map", src["raw"]["model_records/category-map.json"], "E_COMPARE_BUNDLE")
    toks = tokenizers if tokenizers is not None else load_tokenizers(pol, tokenizer_small, tokenizer_large)
    cases = select_cases(rows)
    chosen = [r for pv in cases for r in rows if (r["parent_command_id"], r["view_id"]) == pv]
    measurements = []
    for r in chosen:
        irow = src["index"][(r["parent_command_id"], r["view_id"])]
        t0 = time.perf_counter_ns()
        scene_b = inside(bundle, irow["scene_path"], "E_COMPARE_BUNDLE").read_bytes()
        command_b = inside(bundle, irow["command_path"], "E_COMPARE_BUNDLE").read_bytes()
        scene, command = json.loads(scene_b), json.loads(command_b)
        io_ms = (time.perf_counter_ns() - t0) / 1e6
        t0 = time.perf_counter_ns()
        valid = (canonical_sha256(scene) == irow["scene_sha256"] and canonical_sha256(command) == irow["command_sha256"])
        validation_ms = (time.perf_counter_ns() - t0) / 1e6
        if not valid:
            _fail("E_COMPARE_BUNDLE", [(r["request_id"], "the scene or command record differs from the preparation index")])
        kw = dict(format=r["format"], relation_config_path=Path(relation_config), direction_config_path=Path(direction_config),
                  max_relation_work_units=None if r["format"] == D.FORMATS[0] else cap, category_maps=[cmap])
        mapping = [m[:2] for m in r["mapping"]]
        for rep in range(1 + repetitions):
            with RelationTimer() as rt:
                t0 = time.perf_counter_ns()
                res = serialize(scene, command, **kw)
                total = time.perf_counter_ns() - t0
            t0 = time.perf_counter_ns()
            prompt = build_prompt(proto, mapping, res.document)
            assembly = time.perf_counter_ns() - t0
            tok_ns, ids = {}, {}
            for key in MODEL_KEYS:
                t0 = time.perf_counter_ns()
                ids[key] = toks[key].encode(prompt)
                tok_ns[key] = time.perf_counter_ns() - t0
            if rep == 0:
                continue  # the warm-up
            measurements.append({"request_id": r["request_id"], "parent_command_id": r["parent_command_id"], "view_id": r["view_id"],
                                 "format": r["format"], "object_count": r["object_count"], "repetition": rep,
                                 "relation_ms": rt.ns / 1e6, "relation_calls": rt.calls,
                                 "serialization_ms": (total - rt.ns) / 1e6, "serialize_total_ms": total / 1e6,
                                 "assembly_ms": assembly / 1e6, "tokenize_ms": {k: tok_ns[k] / 1e6 for k in MODEL_KEYS},
                                 "io_ms": io_ms, "validation_ms": validation_ms,
                                 "document_matches": res.status == "ok" and sha256(res.document.encode("utf-8")) == r["source_document_sha256"],
                                 "prompt_matches": prompt == prompts[r["request_id"]],
                                 "tokens_match": {k: ids[k] == frozen[k][r["request_id"]] for k in MODEL_KEYS}})
    summary = summarize(measurements, existing, cases, _hardware())
    if not summary["all_reconstructions_match"]:
        _fail("E_COMPARE_RECONSTRUCTION", [("costs", "a rebuilt document, prompt or token list differs from the frozen one; "
                                                     "no timing is reported")])
    files = {"measurements.jsonl": encode_jsonl(measurements), "summary.json": encode_json(summary), "report.md": render(summary).encode("utf-8")}
    manifest = {"format_version": 1, "record_type": "iref_compare_costs_manifest", "policy_id": D.POLICY_ID,
                "requests_manifest_sha256": sha256((req / "manifest.json").read_bytes()),
                "bundle_manifest_sha256": sha256(src["raw"]["manifest.json"]),
                "run_manifests": {k: runs[k]["manifest_sha256"] for k in MODEL_KEYS}, "work_cap": cap,
                "outputs": {k: sha256(v) for k, v in files.items()}, "code": code_hashes(), "runtime": runtime()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for k, v in files.items():
            output._write_file(staging / k, v)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = verify_costs(staging)
        if bad:
            raise RuntimeError("the costs failed readback: " + "; ".join(bad[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def verify_costs(folder) -> list:
    f = Path(folder)
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_COMPARE_COSTS")
        ms = rows_of("measurements.jsonl", (f / "measurements.jsonl").read_bytes(), "E_COMPARE_COSTS")
        summary = strict_json("summary.json", (f / "summary.json").read_bytes(), "E_COMPARE_COSTS")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable costs: {e}"]
    bad = [f"{n}: changed" for n, want in manifest.get("outputs", {}).items() if sha256((f / n).read_bytes()) != want]
    if bad:
        return bad
    rule = summary.get("percentile_rule", "round_p_n_minus_1")
    again = summarize(ms, summary["existing"], [tuple(c) for c in summary["cases"]], summary["hardware"], rule)
    if again != summary:
        return ["summary.json differs from a recomputation"]
    if render(summary).encode("utf-8") != (f / "report.md").read_bytes():
        return ["report.md differs from a rendering of the summary"]
    return []


def regenerate_costs(*, old, requests, small_run, large_run, out) -> dict:
    """Recompute a costs folder's summary and report with the corrected percentile rule, from its saved measurements and
    the accepted runs; nothing is re-timed. The original folder is left as it is; the new manifest links it."""
    out = output.refuse_existing(out)
    old, req = Path(old), Path(requests)
    bad = verify_costs(old) + verify_compare_requests(req)
    for label, folder in (("0.5B run", small_run), ("7B run", large_run)):
        bad += [f"{label}: {m}" for m in verify_compare_results(folder, req)]
    if bad:
        _fail("E_COMPARE_INPUT", [("inputs", m) for m in bad[:20]])
    old_summary = strict_json("summary.json", (old / "summary.json").read_bytes(), "E_COMPARE_COSTS")
    old_manifest_bytes = (old / "manifest.json").read_bytes()
    old_manifest = json.loads(old_manifest_bytes)
    rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
    runs = {}
    for key, folder in zip(MODEL_KEYS, (small_run, large_run)):
        f = Path(folder)
        runs[key] = {"results": rows_of("results.jsonl", (f / "results.jsonl").read_bytes(), "E_COMPARE_RESULTS"),
                     "sessions": rows_of("sessions.jsonl", (f / "sessions.jsonl").read_bytes(), "E_COMPARE_RESULTS"),
                     "manifest": json.loads((f / "manifest.json").read_bytes()), "manifest_sha256": sha256((f / "manifest.json").read_bytes())}
    if {k: runs[k]["manifest_sha256"] for k in MODEL_KEYS} != old_manifest["run_manifests"] \
            or sha256((req / "manifest.json").read_bytes()) != old_manifest["requests_manifest_sha256"]:
        _fail("E_COMPARE_INPUT", [("inputs", "these are not the requests and runs the original costs were made from")])
    ms_bytes = (old / "measurements.jsonl").read_bytes()
    ms = rows_of("measurements.jsonl", ms_bytes, "E_COMPARE_COSTS")
    summary = summarize(ms, existing_costs(rows, runs, RULE), [tuple(c) for c in old_summary["cases"]], old_summary["hardware"], RULE)
    files = {"measurements.jsonl": ms_bytes, "summary.json": encode_json(summary), "report.md": render(summary).encode("utf-8")}
    manifest = dict(old_manifest, outputs={k: sha256(v) for k, v in files.items()}, code=code_hashes(), runtime=runtime(),
                    regenerated_from={"manifest_sha256": sha256(old_manifest_bytes), "folder": old.name,
                                      "reason": "percentile corrected to the nearest rank, index ceil(p x n) - 1; measurements "
                                                "copied unchanged; nothing re-timed"})
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for k, v in files.items():
            output._write_file(staging / k, v)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = verify_costs(staging)
        if bad:
            raise RuntimeError("the regenerated costs failed readback: " + "; ".join(bad[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary
