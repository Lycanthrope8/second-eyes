"""A2.3d model runs (D95): a resource smoke check that freezes the settings, then one resumable run per model, on the RTX PC.

Both models use A2.3a's accepted path unchanged: TorchModel (float32, eager attention, evaluation mode, batch size one,
one final-position forward with logits_to_keep=1 and no cache, no generation), the independent last-hidden-state path
for canaries, and offered-letter scoring with exact ties to K. Before the model loads, every prompt is re-tokenized with
the model's own pinned tokenizer and compared with the frozen token IDs, every offered letter's boundary is rechecked,
and the checkpoint's files are tied to the pinned revision.

`smoke` loads the model, runs the canaries and the longest requests, and records the settings, timings and peak memory
(`smoke-<model>.json`). `run` refuses to start unless that record is accepted and names the same requests, checkpoint and
code, and refuses to continue if the loaded model's settings differ from it: precision, attention, device and library
versions are never switched silently, and nothing is quantized or offloaded. Each run appends one row per request to
`<out>.partial/rows.jsonl`; `--resume` continues only when every frozen input and configuration hash agrees. Requests
over the context ceiling are kept with their status and never sent; a request whose forward fails is kept as
`execution_failed`. Nothing here reads annotations or scores correctness.
"""
from __future__ import annotations

import hashlib
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
from ..iref_vla.run import _canary
from . import design as D
from .prepare import MODEL_KEYS, code_hashes, load_file_tokenizer, verify_compare_requests

REPO = Path(__file__).resolve().parents[3]
DESCRIPTIONS = {MODEL_KEYS[0]: REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json",
                MODEL_KEYS[1]: REPO / "grounding" / "models" / "qwen2.5-7b-instruct.json"}
OUTPUTS = ("canaries.json", "report.md", "results.jsonl", "sessions.jsonl", "summary.json")
SETTINGS_KEYS = ("dtype", "attention_implementation", "device", "device_name", "torch", "transformers", "cuda",
                 "deterministic_algorithms", "tf32", "logits_api")
MAX_CONSECUTIVE_FAILURES = 3


class CompareFailure(RuntimeError):
    """A failed canary, smoke check or repeated execution failure: exit 3; a partial run stays resumable."""


def _fail(code, problems):
    raise EvaluationInputError([issue(w, code, m) for w, m in problems])


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sync(model):
    if hasattr(model, "synchronize"):
        model.synchronize()


def _nvidia_smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _stats(values) -> dict:
    v = sorted(values)
    if not v:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {"n": len(v), "median": statistics.median(v), "p95": v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
            "max": v[-1]}


def model_protocol(proto, policy, key) -> tuple:
    """The A2.3a protocol and architecture record, adapted to the model's own pinned revision and tokenizer files."""
    spec = policy["models"][key]
    desc = json.loads(DESCRIPTIONS[key].read_text(encoding="utf-8"))
    if (desc["hf_repo"], desc["hf_revision"]) != (spec["hf_repo"], spec["hf_revision"]):
        _fail("E_COMPARE_MODEL", [(str(DESCRIPTIONS[key]), "the model description and the policy name different revisions")])
    if key == MODEL_KEYS[0]:
        if (proto["model"]["hf_repo"], proto["model"]["hf_revision"]) != (spec["hf_repo"], spec["hf_revision"]):
            _fail("E_COMPARE_MODEL", [("protocol", "A2.3a's protocol names another 0.5B revision")])
        return proto, desc
    arch = desc["architecture"]
    mp = dict(proto)
    mp["model"] = dict(proto["model"], hf_repo=spec["hf_repo"], hf_revision=spec["hf_revision"])
    mp["tokenizer"] = dict(proto["tokenizer"], files=dict(spec["tokenizer_files"]))
    return mp, {"architecture": {"max_layers": arch["num_hidden_layers"], "num_key_value_heads": arch["num_key_value_heads"],
                                 "head_dim": arch["hidden_size"] // arch["num_attention_heads"]}}


def setup(*, requests, model_key, model_dir, tokenizer_dir, tokenizer=None, evidence_fn=None) -> dict:
    """Everything a smoke check or a run needs, verified before any model is loaded."""
    req = Path(requests)
    if model_key not in MODEL_KEYS:
        _fail("E_COMPARE_MODEL", [("model", f"unknown model {model_key!r}; expected one of {MODEL_KEYS}")])
    bad = verify_compare_requests(req)
    if bad:
        _fail("E_COMPARE_REQUESTS", [(str(req), m) for m in bad[:20]])
    rman_bytes = (req / "manifest.json").read_bytes()
    rman = json.loads(rman_bytes)
    pol, proto = D.load_policy(req / "policy.json"), load_protocol(req / "protocol.json")
    rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
    prompts = {x["request_id"]: x["prompt"] for x in rows_of("prompts.jsonl", (req / "prompts.jsonl").read_bytes(), "E_COMPARE_REQUESTS")}
    stored = {x["request_id"]: x["token_ids"] for x in rows_of(f"tokens/{model_key}.jsonl",
                                                                (req / "tokens" / f"{model_key}.jsonl").read_bytes(), "E_COMPARE_REQUESTS")}
    mp, desc = model_protocol(proto, pol, model_key)
    if tokenizer is None:
        tokenizer = T.load_pinned_tokenizer(tokenizer_dir) if model_key == MODEL_KEYS[0] \
            else load_file_tokenizer(tokenizer_dir, model_key, pol)
    if tokenizer.identity != rman["tokenizers"][model_key]["identity"]:
        _fail("E_COMPARE_TOKENIZER", [("tokenizer", "not the tokenizer the requests were prepared with")])
    ids_of, verify_ms = {}, {}
    for r in rows:
        t0 = time.perf_counter()
        prompt = prompts[r["request_id"]]
        ids = tokenizer.encode(prompt)
        if ids != stored[r["request_id"]] or [list(x) for x in check_boundary(tokenizer, prompt, [m[:2] for m in r["mapping"]], proto)] != r["mapping"]:
            _fail("E_COMPARE_TOKENS", [(r["request_id"], "re-tokenizing the prompt does not give its frozen token IDs and mapping")])
        ids_of[r["request_id"]], verify_ms[r["request_id"]] = ids, (time.perf_counter() - t0) * 1000
    evidence = (evidence_fn or checkpoint_evidence)(model_dir, mp, desc, None)
    return {"req": req, "rman": rman, "rman_sha256": sha256(rman_bytes), "policy": pol, "proto": proto, "model_proto": mp,
            "rows": rows, "ids": ids_of, "verify_ms": verify_ms, "evidence": evidence, "evidence_sha256": _digest(evidence),
            "model_key": model_key, "tokenizer": tokenizer}


def _settings(info) -> dict:
    return {k: (info or {}).get(k) for k in SETTINGS_KEYS}


def _canaries(model, ctx) -> list:
    rows = ctx["rows"]
    longest = max(rows, key=lambda r: (r["models"][ctx["model_key"]]["input_tokens"], r["request_index"]))
    picks = [rows[0]] + ([longest] if longest["request_id"] != rows[0]["request_id"] else [])
    out = []
    for r in picks:
        if r["models"][ctx["model_key"]]["context_status"] != "within_context_limit":
            continue
        c = _canary(model, r, ctx["ids"][r["request_id"]], ctx["proto"])
        out.append(c)
    return out


def smoke(*, requests, model_key, model_dir, tokenizer_dir, device, out, model_loader=None, tokenizer=None,
          evidence_fn=None, ctx=None) -> dict:
    """The resource smoke check: canaries and the three longest requests, then the frozen settings record. ctx: a
    context another verified preflight built (A2.6c); None reads and verifies A2.3d's request folder, as before."""
    out = Path(out)
    if out.exists():
        _fail("E_EVAL_OUTPUT_EXISTS", [(str(out), "the smoke record already exists")])
    if device != "cuda" and model_loader is None:
        _fail("E_COMPARE_DEVICE", [("device", "the smoke check needs --device cuda (no CPU fallback)")])
    if ctx is None:
        ctx = setup(requests=requests, model_key=model_key, model_dir=model_dir, tokenizer_dir=tokenizer_dir,
                    tokenizer=tokenizer, evidence_fn=evidence_fn)
    t0 = time.perf_counter()
    model = (model_loader or TorchModel.load)(model_dir, device, ctx["model_proto"])
    load_ms = (time.perf_counter() - t0) * 1000
    info = model.info()
    if (info.get("max_position_embeddings") or 0) < ctx["policy"]["context_limit_tokens"]:
        raise CompareFailure("the loaded model supports fewer positions than the context ceiling")
    canaries = _canaries(model, ctx)
    longest = sorted((r for r in ctx["rows"] if r["models"][model_key]["context_status"] == "within_context_limit"),
                     key=lambda r: (-r["models"][model_key]["input_tokens"], r["request_index"]))[:3]
    probes = []
    for r in longest:
        _sync(model)
        t1 = time.perf_counter()
        s = score([tuple(m) for m in r["mapping"]], last_position(model.forward_last(ctx["ids"][r["request_id"]])))
        _sync(model)
        probes.append({"request_id": r["request_id"], "input_tokens": r["models"][model_key]["input_tokens"],
                       "forward_ms": (time.perf_counter() - t1) * 1000, "choice_code": s["choice_code"]})
    peak = model.peak_memory() if hasattr(model, "peak_memory") else None
    accepted = bool(canaries) and all(c["accepted"] for c in canaries)
    record = {"format_version": 1, "record_type": "iref_compare_smoke", "policy_id": ctx["policy"]["policy_id"],
              "model_key": model_key, "hf_revision": ctx["model_proto"]["model"]["hf_revision"],
              "requests_manifest_sha256": ctx["rman_sha256"], "checkpoint_evidence_sha256": ctx["evidence_sha256"],
              "checkpoint_tie": ctx["evidence"].get("tie"), "settings": _settings(info), "batch_size": 1,
              "model_info": info, "nvidia_smi": _nvidia_smi() if device == "cuda" else None, "load_ms": load_ms,
              "canaries": canaries, "longest_requests": probes, "peak_gpu_bytes": peak, "code": code_hashes(),
              "runtime": runtime(), "accepted": accepted,
              "notes": ["PC measurements only (RTX PC), not Quest figures.",
                        "The run refuses any other settings, requests, checkpoint or code than this record names."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_json(record))
    if not accepted:
        raise CompareFailure(f"a canary failed: {[(c['request_id'], c['max_abs_difference']) for c in canaries]}")
    return record


def _row(r, key, status, verify_ms, k, s=None, forward_ms=None, error=None) -> dict:
    obj = s["choice_object_id"] if s else None
    codes = [m[0] for m in r["mapping"]]
    row = {"format_version": 1, "record_type": "iref_compare_result", "policy_id": r["policy_id"], "model_key": key,
           **{x: r[x] for x in ("request_index", "request_id", "selection_rank", "parent_command_id", "view_id", "format",
                                "derived_scene_id", "derived_command_id", "object_count", "prompt_sha256")},
           "token_ids_sha256": r["models"][key]["token_ids_sha256"], "input_tokens": r["models"][key]["input_tokens"],
           "mapping": r["mapping"], "technical_status": status, "execution_index": k,
           "model_choice": None, "choice_code": None, "choice_object_id": None, "choice_list_position": None,
           "selection_reason": None, "tied_codes": [], "scores": [], "top_restricted_share": None, "logit_margin": None,
           "timing_ms": {"verify": verify_ms, "forward": forward_ms}, "error": error}
    if s:
        row.update(model_choice=s["model_choice"], choice_code=s["choice_code"], choice_object_id=obj,
                   choice_list_position=None if obj is None else codes.index(s["choice_code"]) + 1,
                   selection_reason=s["selection_reason"], tied_codes=s["tied_codes"], scores=s["scores"],
                   top_restricted_share=s["top_restricted_share"], logit_margin=s["logit_margin"])
    return row


def _read_partial_rows(path: Path, rows, key) -> list:
    """Rows already written, verified against the frozen requests; a torn final line from an interruption is dropped."""
    if not path.is_file():
        return []
    lines, out = path.read_bytes().split(b"\n"), []
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            x = json.loads(line)
        except ValueError:
            if k >= len(lines) - 2:
                break
            _fail("E_COMPARE_RESUME", [(str(path), f"line {k + 1} is not JSON")])
        r = rows[len(out)] if len(out) < len(rows) else None
        if r is None or x.get("request_id") != r["request_id"] or x.get("model_key") != key \
                or x.get("prompt_sha256") != r["prompt_sha256"] or x.get("token_ids_sha256") != r["models"][key]["token_ids_sha256"]:
            _fail("E_COMPARE_RESUME", [(str(path), f"line {k + 1} is not the next frozen request")])
        out.append(x)
    return out


def _append(path: Path, row) -> None:
    with open(path, "ab") as f:
        f.write((json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def run_compare(*, requests, model_key, model_dir, tokenizer_dir, device, smoke_record, out, resume=False,
                model_loader=None, tokenizer=None, evidence_fn=None, progress=print, ctx=None, verify_results=None) -> dict:
    """One model's run over all its requests; returns the summary (exit 1 if any request is excluded or failed). ctx and
    verify_results: another verified preflight's context and its results readback (A2.6c); None keeps A2.3d's own."""
    out = output.refuse_existing(out)
    partial = out.parent / f"{out.name}.partial"
    if device != "cuda" and model_loader is None:
        _fail("E_COMPARE_DEVICE", [("device", "the run needs --device cuda (no CPU fallback)")])
    if ctx is None:
        ctx = setup(requests=requests, model_key=model_key, model_dir=model_dir, tokenizer_dir=tokenizer_dir,
                    tokenizer=tokenizer, evidence_fn=evidence_fn)
    sm_path = Path(smoke_record)
    try:
        sm_bytes = sm_path.read_bytes()
    except OSError as e:
        _fail("E_COMPARE_SMOKE", [(str(sm_path), f"cannot read the smoke record: {e}")])
    sm = strict_json("smoke record", sm_bytes, "E_COMPARE_SMOKE")
    want = {"model_key": model_key, "requests_manifest_sha256": ctx["rman_sha256"],
            "checkpoint_evidence_sha256": ctx["evidence_sha256"], "code": code_hashes(), "accepted": True}
    if {k: sm.get(k) for k in want} != want:
        _fail("E_COMPARE_SMOKE", [(str(sm_path), "the smoke record is not an accepted check of this model, these requests, "
                                                 "this checkpoint and this code")])
    lock = {"format_version": 1, "record_type": "iref_compare_run_lock", "model_key": model_key,
            "requests_manifest_sha256": ctx["rman_sha256"], "checkpoint_evidence_sha256": ctx["evidence_sha256"],
            "smoke_record_sha256": sha256(sm_bytes), "settings": sm["settings"], "code": code_hashes(), "device": device}
    if partial.exists():
        if not resume:
            _fail("E_COMPARE_RESUME", [(str(partial), "a partial run exists: pass --resume to continue it, or move it away")])
        try:
            old = json.loads((partial / "lock.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _fail("E_COMPARE_RESUME", [(str(partial), "the partial run has no readable lock.json")])
        if old != lock:
            _fail("E_COMPARE_RESUME", [(str(partial), "the frozen inputs or configuration differ from the partial run's")])
    else:
        if resume:
            _fail("E_COMPARE_RESUME", [(str(partial), "there is no partial run to resume")])
        partial.mkdir(parents=True)
        (partial / "lock.json").write_bytes(encode_json(lock))
    done = _read_partial_rows(partial / "rows.jsonl", ctx["rows"], model_key)
    if done:
        (partial / "rows.jsonl").write_bytes(b"".join((json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                                                       + "\n").encode("utf-8") for x in done))
    t0 = time.perf_counter()
    model = (model_loader or TorchModel.load)(model_dir, device, ctx["model_proto"])
    load_ms = (time.perf_counter() - t0) * 1000
    info = model.info()
    if _settings(info) != sm["settings"]:
        raise CompareFailure(f"the loaded model's settings {_settings(info)} differ from the smoke record's {sm['settings']}")
    canaries = _canaries(model, ctx)
    prior = (partial / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if (partial / "sessions.jsonl").exists() else []
    session = {"session": 1 + sum(1 for x in prior if x.strip()), "resumed_after": len(done), "load_ms": load_ms, "canaries": canaries,
               "nvidia_smi": _nvidia_smi() if device == "cuda" else None}
    _append(partial / "sessions.jsonl", session)
    if not canaries or not all(c["accepted"] for c in canaries):
        raise CompareFailure("a canary failed; the partial run is kept for --resume after the cause is fixed")
    consecutive = 0
    rows = ctx["rows"]
    for k in range(len(done), len(rows)):
        r = rows[k]
        rid = r["request_id"]
        if r["models"][model_key]["context_status"] != "within_context_limit":
            _append(partial / "rows.jsonl", _row(r, model_key, "context_budget_exceeded", ctx["verify_ms"][rid], k + 1))
            continue
        try:
            _sync(model)
            t1 = time.perf_counter()
            logits = last_position(model.forward_last(ctx["ids"][rid]))
            _sync(model)
            ms = (time.perf_counter() - t1) * 1000
            s = score([tuple(m) for m in r["mapping"]], logits)
            row, consecutive = _row(r, model_key, "completed", ctx["verify_ms"][rid], k + 1, s, ms), 0
        except Exception as e:  # noqa: BLE001  (kept as an explicit status, never dropped)
            consecutive += 1
            row = _row(r, model_key, "execution_failed", ctx["verify_ms"][rid], k + 1, error=f"{type(e).__name__}: {e}"[:500])
        _append(partial / "rows.jsonl", row)
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            raise CompareFailure(f"{consecutive} consecutive execution failures, the last {row['error']}; stopped, resumable")
        if (k + 1) % 100 == 0 or k + 1 == len(rows):
            progress(f"  {k + 1}/{len(rows)} requests")
    results = _read_partial_rows(partial / "rows.jsonl", rows, model_key)
    if len(results) != len(rows):
        raise CompareFailure(f"{len(results)} rows for {len(rows)} requests")
    sessions = [json.loads(x) for x in (partial / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if info is not None and hasattr(model, "peak_memory"):
        info["peak_gpu_bytes"] = model.peak_memory()
    summary = summarize(results, sessions, info)
    manifest = {"format_version": 1, "record_type": "iref_compare_result_manifest", "policy_id": ctx["policy"]["policy_id"],
                "model_key": model_key, "hf_revision": ctx["model_proto"]["model"]["hf_revision"],
                "requests_manifest_sha256": ctx["rman_sha256"], "smoke_record_sha256": sha256(sm_bytes),
                "checkpoint": ctx["evidence"], "model_info": info, "settings": sm["settings"], "device_requested": device,
                "counts": summary["counts"], "code": code_hashes(), "runtime": runtime(),
                "notes": ["Model choices with complete offered-letter scores; no annotation was read and nothing is "
                          "scored as correct.", "PC measurements only (RTX PC), not Quest figures."]}
    files = {"results.jsonl": encode_jsonl(results), "sessions.jsonl": encode_jsonl(sessions),
             "canaries.json": encode_json({"canaries": [c for s in sessions for c in s["canaries"]]}),
             "summary.json": encode_json(summary), "report.md": render_report(summary, manifest).encode("utf-8")}
    manifest["outputs"] = {k: sha256(v) for k, v in files.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.publish-", dir=str(out.parent)))
    try:
        for k, v in files.items():
            output._write_file(staging / k, v)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = (verify_results or verify_compare_results)(staging, ctx["req"])
        if bad:
            raise RuntimeError("the results failed readback: " + "; ".join(bad[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(partial, ignore_errors=True)
    return summary


def summarize(results, sessions, info) -> dict:
    by = {}
    for r in results:
        b = by.setdefault(f"{r['view_id']}/{r['format']}", {"requests": 0, "completed": 0, "object_choices": 0, "ask": 0,
                                                            "context_budget_exceeded": 0, "execution_failed": 0, "codes": {}})
        b["requests"] += 1
        if r["technical_status"] != "completed":
            b[r["technical_status"]] += 1
            continue
        b["completed"] += 1
        b["object_choices" if r["model_choice"] == "model_choice_object" else "ask"] += 1
        b["codes"][r["choice_code"]] = b["codes"].get(r["choice_code"], 0) + 1
    for b in by.values():
        b["codes"] = dict(sorted(b["codes"].items()))
    st = lambda s: sum(r["technical_status"] == s for r in results)  # noqa: E731  (integer counts only)
    return {"format_version": 1, "record_type": "iref_compare_result_summary", "model_key": results[0]["model_key"] if results else None,
            "counts": {"requests": len(results), "completed": st("completed"), "context_budget_exceeded": st("context_budget_exceeded"),
                       "execution_failed": st("execution_failed"), "sessions": len(sessions),
                       "canaries": sum(len(s["canaries"]) for s in sessions),
                       "canaries_accepted": sum(c["accepted"] for s in sessions for c in s["canaries"])},
            "by_view_format": dict(sorted(by.items())),
            "canary_max_abs_difference": max((c["max_abs_difference"] for s in sessions for c in s["canaries"]), default=None),
            "timing_ms": {"verify": _stats([r["timing_ms"]["verify"] for r in results]),
                          "forward": _stats([r["timing_ms"]["forward"] for r in results if r["timing_ms"]["forward"] is not None]),
                          "model_load": [s["load_ms"] for s in sessions]},
            "peak_gpu_bytes": (info or {}).get("peak_gpu_bytes"), "scope": "PC measurements (RTX PC), not Quest figures"}


def render_report(s, manifest) -> str:
    c, f = s["counts"], s["timing_ms"]["forward"]
    lines = [f"# A2.3d model run: {manifest['model_key']}", "", "Model choices only; scoring is a separate step.", "",
             f"- Requests {c['requests']}: completed {c['completed']}, over the context ceiling {c['context_budget_exceeded']}, "
             f"execution failures {c['execution_failed']}; sessions {c['sessions']}; canaries {c['canaries_accepted']} of "
             f"{c['canaries']} (largest difference {s['canary_max_abs_difference']})",
             f"- Forward median {f['median']} ms, p95 {f['p95']} ms, max {f['max']} ms; peak GPU bytes {s['peak_gpu_bytes']} "
             "(RTX PC, not Quest)", f"- Revision {manifest['hf_revision']}; requests manifest {manifest['requests_manifest_sha256']}",
             "", "| View / format | Requests | Completed | Object choices | ASK | Letters chosen |", "|---|---|---|---|---|---|"]
    for k in [f"{v}/{fm}" for v in D.VIEWS for fm in D.FORMATS if f"{v}/{fm}" in s["by_view_format"]]:
        b = s["by_view_format"][k]
        lines.append(f"| {k} | {b['requests']} | {b['completed']} | {b['object_choices']} | {b['ask']} | "
                     + ", ".join(f"{x}: {n}" for x, n in sorted(b["codes"].items())) + " |")
    return "\n".join(lines) + "\n"


def verify_compare_results(folder, requests=None) -> list:
    f, bad = Path(folder), []
    present = sorted(p.name for p in f.iterdir() if p.is_file()) if f.is_dir() else []
    if present != sorted(OUTPUTS + ("manifest.json",)):
        return [f"expected exactly {sorted(OUTPUTS + ('manifest.json',))}; found {present}"]
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_COMPARE_RESULTS")
        results = rows_of("results.jsonl", (f / "results.jsonl").read_bytes(), "E_COMPARE_RESULTS")
        sessions = rows_of("sessions.jsonl", (f / "sessions.jsonl").read_bytes(), "E_COMPARE_RESULTS")
        summary = strict_json("summary.json", (f / "summary.json").read_bytes(), "E_COMPARE_RESULTS")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable results: {e}"]
    bad += [f"{n}: changed" for n, want in manifest.get("outputs", {}).items() if sha256((f / n).read_bytes()) != want]
    if set(manifest.get("outputs", {})) != set(OUTPUTS):
        bad.append("the manifest does not list exactly the result files")
    if bad:
        return bad
    for k, r in enumerate(results, 1):
        if r.get("request_index") != k or r.get("model_key") != manifest["model_key"] or not D.plain_int(r.get("execution_index")):
            return [f"results line {k}: order, model or counter wrong"]
        if r["technical_status"] == "completed":
            codes = [m[0] for m in r["mapping"]]
            target = dict((m[0], m[1]) for m in r["mapping"]).get(r["choice_code"], "?")
            obj = None if r["choice_code"] == r["mapping"][-1][0] else target
            if [s["code"] for s in r["scores"]] != codes or obj != r["choice_object_id"] or \
                    (obj is None) != (r["model_choice"] == "model_choice_ask"):
                bad.append(f"{r['request_id']}: the choice is not consistent with its mapping and scores")
        elif r["technical_status"] not in ("context_budget_exceeded", "execution_failed") or r["choice_code"] is not None:
            bad.append(f"{r['request_id']}: an unknown status, or a choice on a request that was not completed")
    if requests is not None:
        req = Path(requests)
        rows = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
        if sha256((req / "manifest.json").read_bytes()) != manifest["requests_manifest_sha256"]:
            bad.append("the results were run on another request bundle")
        if len(rows) != len(results) or any(
                (x["request_id"], x["prompt_sha256"], x["mapping"], x["models"][manifest["model_key"]]["token_ids_sha256"])
                != (r["request_id"], r["prompt_sha256"], r["mapping"], r["token_ids_sha256"]) for x, r in zip(rows, results)):
            bad.append("the results do not match the request bundle row by row")
        else:
            over = [x["request_id"] for x in rows if x["models"][manifest["model_key"]]["context_status"] != "within_context_limit"]
            if [r["request_id"] for r in results if r["technical_status"] == "context_budget_exceeded"] != over:
                bad.append("context exclusions differ from the request bundle's")
    if not bad and summarize(results, sessions, manifest.get("model_info")) != summary:
        bad.append("summary.json differs from a recomputation")
    return bad
