"""A2.3a worker (D81): verify the frozen requests and the checkpoint, run the canary, then one forward per request.

Nothing is scored for correctness here: results are model choices with their complete score evidence, saved for a
later comparison. Planned exclusions (empty scenes, context budget) are outcomes; any technical failure stops the run,
leaves a distinctly named diagnostic folder and publishes no success bundle.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl,
                                             issue, runtime, sha256, strict_json)
from ...preparation.iref_vla import tokens as T
from .choices import check_boundary, last_position, score
from .model import TorchModel, checkpoint_evidence
from .prepare import REPO, rows_of, validator, verify_request_dir
from .protocol import load_protocol

MODEL_DESC = REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json"
LIMITATIONS = [
    "One previously inspected development room with oracle (annotated) geometry; inventory views are not evidence profiles.",
    "Parser-conditioned, category-complete subscenes (A2.2b, A2.2c): not representative of all commands, and not held out.",
    "No viewpoint-dependent commands; no natural-language generalization, ambiguity coverage or localization robustness.",
    "Choices are not scored against any target here; agreement between formats is not accuracy.",
    "Restricted shares are conditional ranking shares, not calibrated probabilities that a valid target exists.",
    "The fixed alias order (A..J by object ID, K for ASK) is a limitation; later comparisons need ordering and ID-bias checks.",
    "PC timings only: not Quest latency, memory or end-to-end timing; no prefix or KV reuse was used or measured.",
]


class PilotFailure(RuntimeError):
    """A technical failure during the run: nothing is published as a success."""


def _stats(values) -> dict:
    v = sorted(values)
    n = len(v)
    if not n:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return {"n": n, "min": v[0], "median": med, "p95": v[math.ceil(0.95 * n) - 1], "max": v[-1]}


def _sync(model) -> None:
    if hasattr(model, "synchronize"):
        model.synchronize()


def _canary(model, row, ids, proto) -> dict:
    mapping = [tuple(m) for m in row["mapping"]]
    t0 = time.perf_counter()
    a = last_position(model.forward_last(ids))
    _sync(model)
    b = last_position(model.independent_last(ids))
    _sync(model)
    ms = (time.perf_counter() - t0) * 1000
    pa, pb = [a[i] for _, _, i in mapping], [b[i] for _, _, i in mapping]
    atol, rtol = proto["canary"]["atol"], proto["canary"]["rtol"]
    within = all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(pa, pb))
    ca, cb = score(mapping, a)["choice_code"], score(mapping, b)["choice_code"]
    return {"request_id": row["request_id"], "production_offered_logits": pa, "independent_offered_logits": pb,
            "max_abs_difference": max(abs(x - y) for x, y in zip(pa, pb)), "atol": atol, "rtol": rtol,
            "within_tolerance": within, "production_choice": ca, "independent_choice": cb,
            "accepted": within and ca == cb, "ms": ms,
            "paths": "production: forward(logits_to_keep=1); independent: last hidden state through lm_head"}


def _row(r, verify_ms, status, s=None, forward_ms=None) -> dict:
    out = {"format_version": 1, "record_type": "iref_pilot_result", "protocol_id": r["protocol_id"],
           "request_index": r["request_index"], "request_id": r["request_id"], "parent_command_id": r["parent_command_id"],
           "view_id": r["view_id"], "format": r["format"], "derived_scene_id": r["derived_scene_id"],
           "derived_command_id": r["derived_command_id"], "source_document_sha256": r["source_document_sha256"],
           "prompt_sha256": r["prompt_sha256"], "token_ids_sha256": r["token_ids_sha256"], "input_tokens": r["input_tokens"],
           "mapping": r["mapping"], "technical_status": status, "model_choice": None, "choice_code": None,
           "choice_object_id": None, "selection_reason": None, "tied_codes": [], "scores": None,
           "top_restricted_share": None, "logit_margin": None, "timing_ms": {"verify": verify_ms, "forward": forward_ms}}
    if status == "empty_scene_bypass":
        out.update(model_choice="model_choice_ask", choice_code=r["mapping"][-1][0], selection_reason="empty_candidate_set")
    elif status == "completed":
        out.update({k: s[k] for k in ("model_choice", "choice_code", "choice_object_id", "selection_reason", "tied_codes",
                                      "scores", "top_restricted_share", "logit_margin")})
    return out


def summarize(results, info, canary, counts_in) -> dict:
    views, agreement = {}, {}
    for v in ("full_inventory", "source_known_nyu"):
        for f in ("coordinates_v2", "coordinates_relations_v2"):
            rs = [x for x in results if x["view_id"] == v and x["format"] == f]
            done = [x for x in rs if x["technical_status"] == "completed"]
            views[f"{v}/{f}"] = {
                "requests": len(rs), "completed": len(done),
                "empty_scene_bypass": sum(x["technical_status"] == "empty_scene_bypass" for x in rs),
                "context_budget_exceeded": sum(x["technical_status"] == "context_budget_exceeded" for x in rs),
                "prompt_tokens": _stats(x["input_tokens"] for x in rs),
                "object_choices": sum(x["model_choice"] == "model_choice_object" for x in done),
                "ask_choices": sum(x["model_choice"] == "model_choice_ask" for x in done),
                "exact_score_ties": sum(x["selection_reason"] == "exact_score_tie" for x in done),
                "top_restricted_share": _stats(x["top_restricted_share"] for x in done),
                "logit_margin": _stats(x["logit_margin"] for x in done if x["logit_margin"] is not None)}
        same = differ = excluded = 0
        for p in sorted({x["parent_command_id"] for x in results if x["view_id"] == v}):
            pair = [x for x in results if x["parent_command_id"] == p and x["view_id"] == v]
            if any(x["choice_code"] is None for x in pair):
                excluded += 1
            elif pair[0]["choice_code"] == pair[1]["choice_code"]:
                same += 1
            else:
                differ += 1
        agreement[v] = {"same_choice": same, "different_choice": differ, "not_comparable": excluded,
                        "label": "agreement between formats, not accuracy"}
    done = [x for x in results if x["technical_status"] == "completed"]
    return {"format_version": 1, "record_type": "iref_pilot_summary", "protocol_id": results[0]["protocol_id"] if results else None,
            "counts": dict(counts_in, completed_forwards=len(done),
                           empty_scene_bypass=sum(x["technical_status"] == "empty_scene_bypass" for x in results),
                           context_budget_exceeded=sum(x["technical_status"] == "context_budget_exceeded" for x in results),
                           technical_failures=0),
            "by_view_and_format": views, "format_agreement": agreement,
            "timing": {"model_load_s": (info or {}).get("load_s"), "canary_ms": (canary or {}).get("ms"),
                       "verify_ms": _stats(x["timing_ms"]["verify"] for x in results),
                       "forward_ms": _stats(x["timing_ms"]["forward"] for x in done),
                       "scope": "PC timings: tokenization and verification are separate from model forwards; not Quest"},
            "device": (info or {}).get("device"), "dtype": (info or {}).get("dtype"),
            "peak_gpu_bytes": (info or {}).get("peak_gpu_bytes"), "limitations": LIMITATIONS}


def render_report(s, manifest) -> str:
    c = s["counts"]
    lines = ["# IRef-VLA zero-shot direct-selection pilot (A2.3a)", "",
             "Technically valid execution only: no choice here is checked against a target, nothing is accuracy, and "
             "no format or design is chosen. Shares are uncalibrated ranking diagnostics.", "",
             f"Selected parents {c['parents']}, prepared requests {c['requests']}, completed forwards {c['completed_forwards']}, "
             f"empty-scene bypasses {c['empty_scene_bypass']}, context-budget exclusions {c['context_budget_exceeded']}, "
             f"technical failures {c['technical_failures']}.", "",
             f"Selection: {manifest['selection']['eligible']} eligible parents, conditioned on {manifest['selection']['conditioned_on']}.",
             f"Device {s['device']}, dtype {s['dtype']}, model load {s['timing']['model_load_s']} s, canary "
             f"{s['timing']['canary_ms']} ms, peak GPU allocation {s['peak_gpu_bytes']} bytes.", "",
             "| View / format | Requests | Completed | Objects | ASK | Exact ties | Median tokens | Max tokens |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, x in s["by_view_and_format"].items():
        lines.append(f"| {k} | {x['requests']} | {x['completed']} | {x['object_choices']} | {x['ask_choices']} | "
                     f"{x['exact_score_ties']} | {x['prompt_tokens']['median']} | {x['prompt_tokens']['max']} |")
    lines += ["", "| View | Same choice in both formats | Different | Not comparable |", "|---|---:|---:|---:|"]
    for v, a in s["format_agreement"].items():
        lines.append(f"| {v} | {a['same_choice']} | {a['different_choice']} | {a['not_comparable']} |")
    lines += ["", "Agreement between formats is not accuracy.", "", "## Limitations", ""] + [f"- {x}" for x in s["limitations"]] + [""]
    return "\n".join(lines)


def _diagnostic(out: Path, exc, results, extra) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = out.parent / f"{out.name}.failed-{stamp}"
    d.mkdir(parents=True, exist_ok=False)
    (d / "diagnostic.json").write_text(json.dumps({"status": "FAILED: not a pilot result", "error": f"{type(exc).__name__}: {exc}",
                                                   "rows_completed_before_failure": len(results), "rows": results, **extra},
                                                  indent=2, default=str) + "\n", encoding="utf-8")
    return d


def run_pilot(*, requests, model_dir, device, out, tokenizer_dir, model_description=None, expected_weights_sha256=None,
              model_loader=None, tokenizer=None) -> dict:
    """The `run` command; returns the summary. `model_loader` and `tokenizer` are injectable for fixture tests only."""
    out = output.refuse_existing(out)
    req = Path(requests)
    if device not in ("cpu", "cuda"):
        raise EvaluationInputError([issue("device", "E_PILOT_DEVICE", "device must be cpu or cuda")])
    problems = verify_request_dir(req)
    if problems:
        raise EvaluationInputError([issue(str(req), "E_PILOT_REQUESTS", m) for m in problems[:20]])
    proto = load_protocol(req / "protocol.json")
    rman_bytes = (req / "manifest.json").read_bytes()
    rman = json.loads(rman_bytes)
    rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_PILOT_REQUESTS")
    desc_path = Path(model_description) if model_description else MODEL_DESC
    desc_bytes = desc_path.read_bytes()
    if sha256(desc_bytes) != rman["model_description"]["sha256"]:
        raise EvaluationInputError([issue(str(desc_path), "E_PILOT_MODEL", "not the model description the requests were prepared with")])
    evidence = checkpoint_evidence(model_dir, proto, strict_json("model description", desc_bytes, "E_PREP_MODEL"),
                                   expected_weights_sha256)
    tok = tokenizer if tokenizer is not None else T.load_pinned_tokenizer(tokenizer_dir)
    ids_of, verify_ms = {}, {}
    for r in rows:
        t0 = time.perf_counter()
        prompt = (req / r["prompt_path"]).read_bytes().decode("utf-8")
        ids = tok.encode(prompt)
        if ids != json.loads((req / r["token_ids_path"]).read_bytes()):
            raise EvaluationInputError([issue(r["request_id"], "E_PILOT_TOKENS", "re-tokenizing the prompt does not give its stored token IDs")])
        if [list(x) for x in check_boundary(tok, prompt, [m[:2] for m in r["mapping"]], proto)] != r["mapping"]:
            raise EvaluationInputError([issue(r["request_id"], "E_PILOT_TOKENS", "the mapping's token IDs disagree")])
        ids_of[r["request_id"]], verify_ms[r["request_id"]] = ids, (time.perf_counter() - t0) * 1000
    runnable = [r for r in rows if r["object_ids"] and r["context_status"] == "within_context_limit"]
    model = info = canary = None
    results = []
    try:
        if runnable:
            model = (model_loader or TorchModel.load)(model_dir, device, proto)
            info = model.info()
            if (info.get("max_position_embeddings") or 0) < proto["context_limit_tokens"]:
                raise EvaluationInputError([issue("model", "E_PILOT_CHECKPOINT", "the loaded model supports fewer positions than the pilot's limit")])
            canary = _canary(model, runnable[0], ids_of[runnable[0]["request_id"]], proto)
            if not canary["accepted"]:
                raise PilotFailure(f"canary disagreement on {canary['request_id']}: max |difference| "
                                   f"{canary['max_abs_difference']}, choices {canary['production_choice']}/{canary['independent_choice']}")
        for r in rows:
            if r["context_status"] != "within_context_limit":  # a planned exclusion, empty or not
                results.append(_row(r, verify_ms[r["request_id"]], "context_budget_exceeded"))
            elif not r["object_ids"]:
                results.append(_row(r, verify_ms[r["request_id"]], "empty_scene_bypass"))
            else:
                _sync(model)
                t0 = time.perf_counter()
                logits = last_position(model.forward_last(ids_of[r["request_id"]]))
                _sync(model)
                ms = (time.perf_counter() - t0) * 1000
                results.append(_row(r, verify_ms[r["request_id"]], "completed", score([tuple(m) for m in r["mapping"]], logits), ms))
        if verify_request_dir(req):
            raise PilotFailure("the request folder changed during the run")
    except EvaluationInputError:
        raise
    except BaseException as e:
        _diagnostic(out, e, results, {"requests_manifest_sha256": sha256(rman_bytes), "model_info": info, "canary": canary})
        raise
    if info is not None and hasattr(model, "peak_memory"):
        info["peak_gpu_bytes"] = model.peak_memory()
    counts = {"parents": rman["requests"]["parents"], "requests": len(rows)}
    summary = summarize(results, info, canary, counts)
    manifest = {"format_version": 1, "record_type": "iref_pilot_result_manifest", "protocol_id": proto["protocol_id"],
                "requests_manifest_sha256": sha256(rman_bytes), "protocol_sha256": rman["protocol_sha256"],
                "selection": rman["selection"], "source_bundle": rman["source_bundle"], "checkpoint": evidence,
                "model_info": info, "canary": canary, "device_requested": device, "counts": summary["counts"],
                "runtime": dict(runtime(), **(getattr(tok, "versions", None) or {})), "outputs": {},
                "notes": ["Model choices with complete score evidence; no annotation was read and nothing is scored as correct.",
                          "One forward per non-empty request, no cache within or across requests, no generation."]}
    files = {"results.jsonl": encode_jsonl(results), "summary.json": encode_json(summary),
             "report.md": render_report(summary, manifest).encode("utf-8")}
    manifest["outputs"] = {k: sha256(v) for k, v in files.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for k, v in files.items():
            output._write_file(staging / k, v)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = verify_results(staging)
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


def verify_results(folder) -> list:
    f, bad = Path(folder), []
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_PILOT_RESULTS")
        results = rows_of("results.jsonl", (f / "results.jsonl").read_bytes(), "E_PILOT_RESULTS")
        summary = strict_json("summary.json", (f / "summary.json").read_bytes(), "E_PILOT_RESULTS")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable results: {e}"]
    bad += [f"manifest: {e.message}" for e in validator("result_manifest").iter_errors(manifest)]
    for name, want in manifest.get("outputs", {}).items():
        if not (f / name).is_file() or sha256((f / name).read_bytes()) != want:
            bad.append(f"{name}: missing or changed")
    if len(results) != manifest.get("counts", {}).get("requests") or [r.get("request_index") for r in results] != list(range(1, len(results) + 1)):
        bad.append("results are not one row per request, in order")
    for r in results:
        errs = [e.message for e in validator("result_row").iter_errors(r)]
        if errs:
            bad.append(f"{r.get('request_id')}: {errs[0]}")
            continue
        codes = [m[0] for m in r["mapping"]]
        st, code = r["technical_status"], r["choice_code"]
        if st == "completed" and (code not in codes or len(r["scores"]) != len(codes)
                                  or (code == codes[-1]) != (r["choice_object_id"] is None)
                                  or (code != codes[-1] and r["choice_object_id"] != dict((m[0], m[1]) for m in r["mapping"])[code])):
            bad.append(f"{r['request_id']}: the choice is not consistent with its mapping and scores")
        if st != "completed" and r["scores"] is not None:
            bad.append(f"{r['request_id']}: scores without a model call")
    if not bad and summarize(results, manifest.get("model_info"), manifest.get("canary"),
                             {"parents": manifest["counts"]["parents"], "requests": manifest["counts"]["requests"]}) != summary:
        bad.append("summary.json differs from a recomputation")
    return bad
