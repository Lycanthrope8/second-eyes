"""A2.3c worker (D94): canaries, 128 repeat controls, then the 4,408 other cells, on the RTX PC.

The accepted A2.3a model path is reused unchanged: TorchModel (float32, eager attention, batch 1, one final-position
forward, no cache, no generation), the independent last-hidden-state path for canaries, and the accepted offered-logit
scoring with exact ties resolved to K. Every cell is checked against the frozen bundle before the model loads. The
identity cells run first and must reproduce run r003's offered logits and choices within the policy's tolerance;
otherwise a failure diagnostic is kept and no other cell runs. Passed controls are reused as their grid cells.
Nothing here reads annotations or scores correctness.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl,
                                             issue, runtime, sha256, strict_json)
from ...preparation.iref_vla import tokens as T
from ..iref_vla.choices import check_boundary, last_position, score
from ..iref_vla.model import TorchModel, checkpoint_evidence
from ..iref_vla.prepare import rows_of
from ..iref_vla.protocol import load_protocol
from ..iref_vla.run import MODEL_DESC, _canary, verify_results
from . import design as D
from .prepare import read_variants, verify_order_requests

OUTPUTS = ("canaries.json", "controls.jsonl", "report.md", "results.jsonl", "summary.json")


class OrderingFailure(RuntimeError):
    """A failed canary or repeat control, or a technical failure during the run: exit 3, nothing published."""


def _sync(model) -> None:
    if hasattr(model, "synchronize"):
        model.synchronize()


def _stats(values) -> dict:
    v = sorted(values)
    if not v:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {"n": len(v), "median": statistics.median(v), "p95": v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
            "max": v[-1]}


def _driver():
    """The GPU's name and driver version as nvidia-smi reports them (best effort; None if unavailable)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _decode(mapping, code):
    target = dict((m[0], m[1]) for m in mapping)[code]
    return None if code == mapping[-1][0] else target


def _row(r, s, verify_ms, forward_ms, execution_index) -> dict:
    obj = s["choice_object_id"]
    codes = [m[0] for m in r["mapping"]]
    return {"format_version": 1, "record_type": "iref_order_result", "policy_id": r["policy_id"],
            **{k: r[k] for k in ("variant_index", "variant_id", "base_request_index", "base_request_id", "selection_rank",
                                 "parent_command_id", "view_id", "format", "derived_scene_id", "derived_command_id",
                                 "object_count", "assignment_shift", "order_shift")},
            "execution_index": execution_index, "prompt_sha256": r["prompt_sha256"], "token_ids_sha256": r["token_ids_sha256"],
            "input_tokens": r["input_tokens"], "mapping": r["mapping"], "technical_status": "completed",
            "model_choice": s["model_choice"], "choice_code": s["choice_code"], "choice_object_id": obj,
            "choice_list_position": None if obj is None else codes.index(s["choice_code"]) + 1,
            "choice_scene_position": None if obj is None else r["object_ids"].index(obj) + 1,
            "selection_reason": s["selection_reason"], "tied_codes": s["tied_codes"], "scores": s["scores"],
            "top_restricted_share": s["top_restricted_share"], "logit_margin": s["logit_margin"],
            "repeat_control": (r["assignment_shift"], r["order_shift"]) == (0, 0),
            "timing_ms": {"verify": verify_ms, "forward": forward_ms}}


def _control(r, s, base, tol) -> dict:
    new = {x["code"]: x["logit"] for x in s["scores"]}
    old = {x["code"]: x["logit"] for x in base["scores"]}
    diffs = [abs(new[c] - old[c]) for c in old] if set(new) == set(old) else [float("inf")]
    within = set(new) == set(old) and all(abs(new[c] - old[c]) <= tol["atol"] + tol["rtol"] * abs(old[c]) for c in old)
    same = s["choice_code"] == base["choice_code"] and s["choice_object_id"] == base["choice_object_id"]
    return {"variant_id": r["variant_id"], "base_request_id": r["base_request_id"], "base_choice_code": base["choice_code"],
            "new_choice_code": s["choice_code"], "base_choice_object_id": base["choice_object_id"],
            "new_choice_object_id": s["choice_object_id"], "max_abs_logit_difference": max(diffs),
            "within_tolerance": within, "same_choice": same, "passed": within and same}


def summarize(results, controls, canaries, info, load_ms) -> dict:
    by = {}
    for r in results:
        key = f"{r['view_id']}/{r['format']}"
        by.setdefault(key, {"cells": 0, "object_choices": 0, "ask": 0, "codes": {}})
        b = by[key]
        b["cells"] += 1
        b["object_choices" if r["model_choice"] == "model_choice_object" else "ask"] += 1
        b["codes"][r["choice_code"]] = b["codes"].get(r["choice_code"], 0) + 1
    for b in by.values():
        b["codes"] = dict(sorted(b["codes"].items()))
    return {"format_version": 1, "record_type": "iref_order_result_summary",
            "counts": {"cells": len(results), "completed": sum(r["technical_status"] == "completed" for r in results),
                       "identity_controls": len(controls), "identity_controls_passed": sum(c["passed"] for c in controls),
                       "nonidentity_cells": sum(not r["repeat_control"] for r in results), "canaries": len(canaries),
                       "canaries_accepted": sum(c["accepted"] for c in canaries), "technical_failures": 0,
                       "context_exclusions": 0},
            "choices_by_view_format": dict(sorted(by.items())),
            "repeat_control_max_abs_logit_difference": max((c["max_abs_logit_difference"] for c in controls), default=None),
            "canary_max_abs_difference": max((c["max_abs_difference"] for c in canaries), default=None),
            "timing_ms": {"verify": _stats([r["timing_ms"]["verify"] for r in results]),
                          "forward": _stats([r["timing_ms"]["forward"] for r in results]), "model_load": load_ms},
            "peak_gpu_bytes": (info or {}).get("peak_gpu_bytes")}


def render_report(s, manifest) -> str:
    c = s["counts"]
    lines = ["# A2.3c crossed code-assignment and choices-order run", "",
             "Model choices only: nothing here is scored against a target (see the scoring step).", "",
             f"- Cells: {c['cells']} completed ({c['identity_controls']} identity controls, all passed: "
             f"{c['identity_controls_passed'] == c['identity_controls']}; {c['nonidentity_cells']} other cells)",
             f"- Canaries: {c['canaries_accepted']} of {c['canaries']} accepted (largest difference "
             f"{s['canary_max_abs_difference']}); repeat-control largest offered-logit difference "
             f"{s['repeat_control_max_abs_logit_difference']}",
             f"- Forward: median {s['timing_ms']['forward']['median']} ms, p95 {s['timing_ms']['forward']['p95']} ms; "
             f"model load {s['timing_ms']['model_load']} ms; peak GPU bytes {s['peak_gpu_bytes']}",
             f"- Device {manifest['device_requested']}; requests manifest {manifest['requests_manifest_sha256']}", "",
             "| View / format | Cells | Object choices | ASK | Chosen codes |", "|---|---|---|---|---|"]
    for k, b in s["choices_by_view_format"].items():
        lines.append(f"| {k} | {b['cells']} | {b['object_choices']} | {b['ask']} | "
                     + ", ".join(f"{x}: {n}" for x, n in b["codes"].items()) + " |")
    return "\n".join(lines) + "\n"


def _diagnostic(out: Path, exc, extra) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = out.parent / f"{out.name}.failed-{stamp}"
    d.mkdir(parents=True, exist_ok=False)
    (d / "diagnostic.json").write_text(json.dumps({"status": "FAILED: not an experiment result",
                                                   "error": f"{type(exc).__name__}: {exc}", **extra},
                                                  indent=2, default=str) + "\n", encoding="utf-8")
    return d


def run_ordering(*, requests, base_pilot, model_dir, device, out, tokenizer_dir, model_description=None,
                 expected_weights_sha256=None, model_loader=None, tokenizer=None, evidence_fn=None, progress=print) -> dict:
    """The `run` command; returns the summary.

    `model_loader`, `tokenizer` and `evidence_fn` are injectable for fixture tests and labelled rehearsals only; the
    command line never passes them, so a real run always loads TorchModel and checks the checkpoint files' hashes.
    """
    out = output.refuse_existing(out)
    req, bp = Path(requests), Path(base_pilot)
    if device != "cuda" and model_loader is None:
        raise EvaluationInputError([issue("device", "E_ORDER_DEVICE", "the run needs --device cuda (no CPU fallback)")])
    problems = verify_order_requests(req)
    if problems:
        raise EvaluationInputError([issue(str(req), "E_ORDER_REQUESTS", m) for m in problems[:20]])
    rman_bytes = (req / "manifest.json").read_bytes()
    rman = json.loads(rman_bytes)
    pol, proto = D.load_policy(req / "policy.json"), load_protocol(req / "protocol.json")
    rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_ORDER_REQUESTS")
    if rman["counts"]["context_budget_exceeded"]:
        raise EvaluationInputError([issue(str(req), "E_ORDER_EXCLUSIONS", f"{rman['counts']['context_budget_exceeded']} "
                                          "cells exceed the context limit; the grids are incomplete and are not run")])
    bad = verify_results(bp)
    if bad:
        raise EvaluationInputError([issue(str(bp), "E_ORDER_BASE_PILOT", m) for m in bad[:20]])
    bman_bytes = (bp / "manifest.json").read_bytes()
    bman = json.loads(bman_bytes)
    bres_bytes = (bp / "results.jsonl").read_bytes()
    if bman["requests_manifest_sha256"] != rman["base_requests"]["manifest_sha256"]:
        raise EvaluationInputError([issue(str(bp), "E_ORDER_BASE_PILOT", "this pilot ran other base requests")])
    if sha256(bres_bytes) != pol["pins"]["base_results_sha256"]:
        raise EvaluationInputError([issue(str(bp), "E_ORDER_PIN", f"results.jsonl is {sha256(bres_bytes)}, not the pinned "
                                                                   f"{pol['pins']['base_results_sha256']}")])
    base_rows = {r["request_id"]: r for r in rows_of("results.jsonl", bres_bytes, "E_ORDER_BASE_PILOT")}
    for r in rows:
        if (r["assignment_shift"], r["order_shift"]) != (0, 0):
            continue
        b = base_rows.get(r["base_request_id"])
        if b is None or b["technical_status"] != "completed" or b["prompt_sha256"] != r["prompt_sha256"] \
                or b["token_ids_sha256"] != r["token_ids_sha256"] or b["mapping"] != r["mapping"]:
            raise EvaluationInputError([issue(r["variant_id"], "E_ORDER_BASE_PILOT", "the identity cell is not the pilot's "
                                              "completed request (prompt, token IDs or mapping differ)")])
    desc_path = Path(model_description) if model_description else MODEL_DESC
    desc_bytes = Path(desc_path).read_bytes()
    if sha256(desc_bytes) != rman["model_description"]["sha256"]:
        raise EvaluationInputError([issue(str(desc_path), "E_ORDER_MODEL", "not the model description of the requests")])
    evidence = (evidence_fn or checkpoint_evidence)(model_dir, proto, strict_json("model description", desc_bytes,
                                                                                  "E_ORDER_MODEL"), expected_weights_sha256)
    want_files = {k: v.get("sha256") for k, v in bman["checkpoint"].get("files", {}).items()}
    got_files = {k: v.get("sha256") for k, v in evidence.get("files", {}).items()}
    if want_files != got_files:
        raise EvaluationInputError([issue(str(model_dir), "E_ORDER_CHECKPOINT", "the checkpoint, config or tokenizer files "
                                          "differ from those run r003 used")])
    tok = tokenizer if tokenizer is not None else T.load_pinned_tokenizer(tokenizer_dir)
    if (tok.identity, tok.kind) != (rman["tokenizer"]["identity"], rman["tokenizer"]["kind"]):
        raise EvaluationInputError([issue("tokenizer", "E_ORDER_TOKENIZER", "not the tokenizer of the requests")])
    texts = read_variants(req, rows)
    ids_of, verify_ms = {}, {}
    for r in rows:
        t0 = time.perf_counter()
        prompt, stored = texts[r["variant_id"]]
        ids = tok.encode(prompt)
        if ids != stored or [list(x) for x in check_boundary(tok, prompt, [m[:2] for m in r["mapping"]], proto)] != r["mapping"]:
            raise EvaluationInputError([issue(r["variant_id"], "E_ORDER_TOKENS", "re-tokenizing the prompt does not give its "
                                              "stored token IDs and mapping")])
        ids_of[r["variant_id"]], verify_ms[r["variant_id"]] = ids, (time.perf_counter() - t0) * 1000
    by_vid = {r["variant_id"]: r for r in rows}
    order = json.loads((req / "schedule.json").read_bytes())["order"]
    first_base = next(r for r in rows if r["base_request_index"] == pol["canary"]["base_request_index"])["base_request_id"]
    canary_ids = [D.variant_id(first_base, a, p) for a, p in pol["canary"]["cells"]]
    model = info = None
    canaries, controls, results, load_ms = [], [], {}, None
    stage = "loading the model"
    try:
        t0 = time.perf_counter()
        model = (model_loader or TorchModel.load)(model_dir, device, proto)
        load_ms = (time.perf_counter() - t0) * 1000
        info = model.info()
        if device == "cuda":
            info["nvidia_smi"] = _driver()
        if (info.get("max_position_embeddings") or 0) < pol["context_limit_tokens"]:
            raise OrderingFailure("the loaded model supports fewer positions than the context limit")
        stage = "canaries"
        for vid in canary_ids:
            r = by_vid[vid]
            c = _canary(model, {"request_id": vid, "mapping": r["mapping"]}, ids_of[vid], {"canary": pol["canary"]})
            c = {"variant_id": vid, **{k: v for k, v in c.items() if k != "request_id"},
                 "production_object": _decode(r["mapping"], c["production_choice"]),
                 "independent_object": _decode(r["mapping"], c["independent_choice"])}
            canaries.append(c)
            if not c["accepted"]:
                raise OrderingFailure(f"canary {vid} failed: max |difference| {c['max_abs_difference']}, choices "
                                      f"{c['production_choice']}/{c['independent_choice']}")
        stage = "repeat controls"
        n_ident = rman["counts"]["identity_cells"]
        for k, vid in enumerate(order, 1):
            if k == n_ident + 1:
                failed = [c for c in controls if not c["passed"]]
                if failed:
                    raise OrderingFailure(f"{len(failed)} of {len(controls)} repeat controls failed, first "
                                          f"{failed[0]['variant_id']} (max |difference| "
                                          f"{failed[0]['max_abs_logit_difference']}, choices "
                                          f"{failed[0]['base_choice_code']}/{failed[0]['new_choice_code']})")
                stage = "nonidentity cells"
            r = by_vid[vid]
            _sync(model)
            t1 = time.perf_counter()
            logits = last_position(model.forward_last(ids_of[vid]))
            _sync(model)
            ms = (time.perf_counter() - t1) * 1000
            s = score([tuple(m) for m in r["mapping"]], logits)
            results[vid] = _row(r, s, verify_ms[vid], ms, k)
            if k <= n_ident:
                controls.append(_control(r, s, base_rows[r["base_request_id"]], pol["repeat_control"]))
            if k % pol["progress_interval"] == 0 or k == len(order):
                progress(f"  {k}/{len(order)} cells ({stage})")
        failed = [c for c in controls if not c["passed"]]
        if failed:
            raise OrderingFailure(f"{len(failed)} repeat controls failed")
        if verify_order_requests(req):
            raise OrderingFailure("the request bundle changed during the run")
    except EvaluationInputError:
        raise
    except BaseException as e:
        _diagnostic(out, e, {"stage": stage, "requests_manifest_sha256": sha256(rman_bytes),
                             "base_pilot_manifest_sha256": sha256(bman_bytes), "model_info": info, "canaries": canaries,
                             "controls": controls, "cells_completed": len(results),
                             "rows": [results[v] for v in order if v in results]})
        if isinstance(e, OrderingFailure):
            raise
        raise OrderingFailure(f"{type(e).__name__}: {e}") from e
    if info is not None and hasattr(model, "peak_memory"):
        info["peak_gpu_bytes"] = model.peak_memory()
    canonical = [results[r["variant_id"]] for r in rows]
    summary = summarize(canonical, controls, canaries, info, load_ms)
    manifest = {"format_version": 1, "record_type": "iref_order_result_manifest", "policy_id": pol["policy_id"],
                "policy_sha256": rman["policy_sha256"], "requests_manifest_sha256": sha256(rman_bytes),
                "schedule_sha256": rman["schedule_sha256"],
                "base_pilot": {"manifest_sha256": sha256(bman_bytes), "results_sha256": sha256(bres_bytes),
                               "requests_manifest_sha256": bman["requests_manifest_sha256"]},
                "checkpoint": evidence, "model_info": info, "device_requested": device, "counts": summary["counts"],
                "runtime": dict(runtime(), **(getattr(tok, "versions", None) or {})), "outputs": {},
                "notes": ["Model choices with complete score evidence; no annotation was read and nothing is scored as correct.",
                          "One final-position forward per cell, no cache within or across cells, no generation.",
                          "The 128 identity cells are fresh repeat controls of run r003 and serve as the grid's (0, 0) cells."]}
    files = {"results.jsonl": encode_jsonl(canonical), "controls.jsonl": encode_jsonl(controls),
             "canaries.json": encode_json({"canaries": canaries}), "summary.json": encode_json(summary),
             "report.md": render_report(summary, manifest).encode("utf-8")}
    manifest["outputs"] = {k: sha256(v) for k, v in files.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for k, v in files.items():
            output._write_file(staging / k, v)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = verify_order_results(staging, req)
        if bad:
            raise RuntimeError("the results failed readback: " + "; ".join(bad[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def verify_order_results(folder, requests=None) -> list:
    """Files, hashes, rows, choice consistency, controls, canaries, summary; with `requests`, the join to every cell."""
    f, bad = Path(folder), []
    present = sorted(p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file()) if f.is_dir() else []
    if present != sorted(OUTPUTS + ("manifest.json",)):
        return [f"expected exactly {sorted(OUTPUTS + ('manifest.json',))}; found {present[:8]}"]
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_ORDER_RESULTS")
        results = rows_of("results.jsonl", (f / "results.jsonl").read_bytes(), "E_ORDER_RESULTS")
        controls = rows_of("controls.jsonl", (f / "controls.jsonl").read_bytes(), "E_ORDER_RESULTS")
        canaries = strict_json("canaries.json", (f / "canaries.json").read_bytes(), "E_ORDER_RESULTS")["canaries"]
        summary = strict_json("summary.json", (f / "summary.json").read_bytes(), "E_ORDER_RESULTS")
    except (OSError, EvaluationInputError, KeyError, TypeError) as e:
        return [f"unreadable results: {e}"]
    bad += [f"manifest: {m}" for m in D.schema_errors("result_manifest", manifest)]
    for name, want in (manifest.get("outputs") or {}).items():
        if sha256((f / name).read_bytes()) != want:
            bad.append(f"{name}: changed since the manifest was written")
    if set((manifest.get("outputs") or {})) != set(OUTPUTS):
        bad.append("the manifest does not list exactly the result files")
    if bad:
        return bad
    for k, r in enumerate(results, 1):
        errs = D.schema_errors("result_row", r) if isinstance(r, dict) else ["not a JSON object"]
        if errs:
            return [f"results line {k}: {errs[0]}"]
        if r["variant_index"] != k:
            return [f"results line {k}: not in canonical order"]
        codes = [m[0] for m in r["mapping"]]
        obj = _decode(r["mapping"], r["choice_code"]) if r["choice_code"] in codes else "?"
        if (r["choice_code"] not in codes or [s["code"] for s in r["scores"]] != codes or obj != r["choice_object_id"]
                or (obj is None) != (r["model_choice"] == "model_choice_ask")
                or r["choice_list_position"] != (None if obj is None else codes.index(r["choice_code"]) + 1)):
            bad.append(f"{r['variant_id']}: the choice is not consistent with its mapping and scores")
    for c in controls:
        bad += [f"control {c.get('variant_id')}: {m}" for m in D.schema_errors("control_row", c)]
    for c in canaries:
        bad += [f"canary {c.get('variant_id')}: {m}" for m in D.schema_errors("canary_row", c)]
    if not controls or not all(c["passed"] for c in controls) or not canaries or not all(c["accepted"] for c in canaries):
        bad.append("a published result needs every canary accepted and every repeat control passed")
    ident = sorted((r for r in results if r["repeat_control"]), key=lambda r: r["execution_index"])
    if [c["variant_id"] for c in controls] != [r["variant_id"] for r in ident]:
        bad.append("the repeat controls are not the identity cells in execution order")
    if sorted(r["execution_index"] for r in results) != list(range(1, len(results) + 1)):
        bad.append("the execution indices are not a permutation of 1..N")
    if requests is not None:
        req = Path(requests)
        rman = json.loads((req / "manifest.json").read_bytes())
        rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_ORDER_REQUESTS")
        if sha256((req / "manifest.json").read_bytes()) != manifest["requests_manifest_sha256"]:
            bad.append("the results were run from another request bundle")
        if len(rows) != len(results):
            bad.append(f"{len(results)} results for {len(rows)} cells")
        else:
            keys = ("variant_index", "variant_id", "base_request_id", "parent_command_id", "view_id", "format",
                    "assignment_shift", "order_shift", "prompt_sha256", "token_ids_sha256", "input_tokens", "mapping",
                    "object_count")
            for r, q in zip(results, rows):
                if any(r[k] != q[k] for k in keys) or r["execution_index"] != q["schedule_position"]:
                    bad.append(f"{r['variant_id']}: differs from its request cell")
                    break
                if r["choice_scene_position"] != (None if r["choice_object_id"] is None
                                                  else q["object_ids"].index(r["choice_object_id"]) + 1):
                    bad.append(f"{r['variant_id']}: the chosen object's scene position is wrong")
                    break
        if rman["schedule_sha256"] != manifest["schedule_sha256"]:
            bad.append("the schedule differs from the request bundle's")
    if not bad:
        again = summarize(results, controls, canaries, manifest.get("model_info"), summary["timing_ms"]["model_load"])
        if again != summary:
            bad.append("summary.json differs from a recomputation")
    return bad
