"""A2.5 delivery 1, step 3 (D101, D103): a replay's raw results against the frozen bundle, on the laptop.

The results folder comes from the headset (`replay-pull`) or from the desktop diagnosis (`replay-desktop`); both write the
same files. Collection completeness is not acceptance: this comparison enforces everything itself.

- **Inputs:** the bundle reads back on its own. The results' identity names the bundle's own headset files by SHA-256,
  and the optional push receipt names the same files and the bundle's root manifest. That ties the root manifest to
  what the headset read.
- **Coverage:** the result records are exactly the bundle's requests, in the bundle's order, and the fixtures exactly
  its written fixtures.
- **Records:** every request passed its input checks with the device's own hashes equal to the bundle's and the native
  tokenization equal to the frozen IDs. Its evaluations are exactly the D103 schedule:
  - U: offset 0, all n tokens, keep 0;
  - R: offset n - 1, 1 token, keep n - 1;
  - P: offset 0, k tokens, keep 0, then offset k, n - k tokens, keep k.

  Every evaluation returns 0 with the expected cache count. Every scored path has a full, finite row, the bundle's
  mapping, and a recorded decision that its own saved logits reproduce.
- **D101,** five comparisons per request:
  - U against float32: best candidate and ranking; TVD reported.
  - R and P against float32, and R and P against U: best candidate, ranking, and TVD at most 0.05.

  Shares are recomputed from the saved float32 logits over every offered code, K included; exact ties go to K. Ranking
  compares every pair in the union of codes with at least 1% share on either path. Every mismatch is listed.
- **Runtime reporting:** the values the results report, and the startup-log lines that support each field, kept at
  their reported precision. K/V types come only from those lines.

The verdict: replay acceptance passes only when every check and all comparisons pass. Output goes to a new folder:
`comparisons.jsonl`, `summary.json`, `report.md` and `manifest.json`.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

from ..evaluation.iref_vla import output
from ..evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl, issue, runtime, sha256
from .replay_bundle import verify_replay_bundle

PLAUSIBLE, TVD_MAX = 0.01, 0.05
COMPARISONS = (("U", "float32", False), ("R", "float32", True), ("P", "float32", True), ("R", "U", True), ("P", "U", True))
ACCEPTED_RUNTIME = {"threads": 2, "n_seq": 16, "flags": 0, "version": "b11277 (eae11d22)"}   # D58, D59; runtime-identity policy
LOG_FIELDS = {
    "file_type": re.compile(r"^print_info: file type\s*=\s*(.+?)\s*$"),
    "n_seq_max": re.compile(r"^llama_context: n_seq_max\s*=\s*(\S+)\s*$"),
    "n_ctx": re.compile(r"^llama_context: n_ctx\s*=\s*(\S+)\s*$"),
    "n_ctx_seq": re.compile(r"^llama_context: n_ctx_seq\s*=\s*(\S+)\s*$"),
    "n_batch": re.compile(r"^llama_context: n_batch\s*=\s*(\S+)\s*$"),
    "n_ubatch": re.compile(r"^llama_context: n_ubatch\s*=\s*(\S+)\s*$"),
    "flash_attn_setting": re.compile(r"^llama_context: flash_attn\s*=\s*(\S+)\s*$"),
    "flash_attn_resolved": re.compile(r"Flash Attention (enabled|disabled)"),
    "kv_unified": re.compile(r"^llama_context: kv_unified\s*=\s*(\S+)\s*$"),
    "kv_buffer": re.compile(r"^llama_kv_cache:\s+\S+ KV buffer size\s*=\s*(.+?)\s*$"),
    "kv_cache": re.compile(r"^llama_kv_cache: size\s*=\s*(.+?) \((.+?)\), K \((\w+)\):\s*(.+?), V \((\w+)\):\s*(.+?)\s*$"),
}


def f32(x) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def shares(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    z = math.fsum(e)
    return [v / z for v in e]


def choose(codes, logits, ask):
    m = max(logits)
    tied = [c for c, x in zip(codes, logits) if x == m]
    return (ask, "exact_score_tie", tied) if len(tied) > 1 else (tied[0], "max_offered_logit", [])


def compare(codes, a_logits, b_logits, ask, judge_tvd) -> dict:
    """D101 on one pair of paths: a is the path under test, b the one it is compared with."""
    sa, sb = shares(a_logits), shares(b_logits)
    ca, cb = choose(codes, a_logits, ask)[0], choose(codes, b_logits, ask)[0]
    union = sorted(c for c, x, y in zip(codes, sa, sb) if x >= PLAUSIBLE or y >= PLAUSIBLE)
    s_a, s_b = dict(zip(codes, sa)), dict(zip(codes, sb))
    sign = lambda v: (v > 0) - (v < 0)  # noqa: E731
    changed = []
    for i, x in enumerate(union):
        for y in union[i + 1:]:
            if sign(s_a[x] - s_a[y]) != sign(s_b[x] - s_b[y]):
                hi, lo = (x, y) if s_a[x] >= s_a[y] else (y, x)
                changed.append(f"{hi}/{lo}")
    tvd = 0.5 * math.fsum(abs(x - y) for x, y in zip(sa, sb))
    failed = [name for name, bad in (("best candidate", ca != cb), ("plausible ranking", bool(changed)),
                                      ("TVD", judge_tvd and tvd > TVD_MAX)) if bad]
    return {"choice_a": ca, "choice_b": cb, "union": union, "changed_pairs": changed, "tvd": tvd,
            "tvd_judged": judge_tvd, "failed": failed, "pass": not failed}


def startup_fields(text: str) -> dict:
    """Each field with its value at the reported precision and the line numbers that support it; missing fields are
    listed, never filled in from defaults."""
    found = {}
    for k, line in enumerate(text.splitlines(), start=1):
        for name, rx in LOG_FIELDS.items():
            m = rx.search(line)
            if m:
                found.setdefault(name, []).append({"line": k, "groups": list(m.groups()), "text": line.strip()})
    out = {"missing": [n for n in LOG_FIELDS if n not in found]}
    for name, hits in found.items():
        out[name] = {"value": hits[-1]["groups"] if len(hits[-1]["groups"]) > 1 else hits[-1]["groups"][0],
                     "lines": [h["line"] for h in hits], "text": hits[-1]["text"]}
    kv = out.get("kv_cache")
    out["kv_types_reported"] = bool(kv) and len(kv["value"]) == 6
    if out["kv_types_reported"]:
        size, cells, k_type, k_size, v_type, v_size = kv["value"]
        out["kv"] = {"size": size, "cells_layers_seqs": cells, "k_type": k_type, "k_size": k_size, "v_type": v_type,
                     "v_size": v_size, "line": kv["lines"][-1]}
    return out


def _lines(path):
    rows = []
    for k, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({"__unparseable__": k})
    return rows


def _path_problems(rid, name, path, prescribed, mapping, n_vocab, ask):
    bad = []
    evals = path.get("evals") if isinstance(path, dict) else None
    got = [(e.get("offset"), e.get("count"), e.get("keep"), e.get("status"), e.get("n_cached"), e.get("expected_n_cached"))
           for e in (evals or [])]
    want = [(o, c, kp, 0, ex, ex) for o, c, kp, ex in prescribed]
    if got != want:
        bad.append(f"{rid} {name}: evaluations {got} are not the D103 schedule {want}")
        return bad, None
    if path.get("error") is not None or path.get("logits_returned") != n_vocab or path.get("non_finite") != 0:
        bad.append(f"{rid} {name}: not a full finite row (error {path.get('error')!r}, {path.get('logits_returned')} values)")
        return bad, None
    offered = path.get("offered") or []
    if [(o.get("code"), o.get("target"), o.get("token_id")) for o in offered] != [tuple(m) for m in mapping]:
        bad.append(f"{rid} {name}: offered codes, targets or token IDs differ from the bundle's mapping")
        return bad, None
    logits = [f32(o.get("logit")) for o in offered]
    if any(not math.isfinite(x) for x in logits):
        bad.append(f"{rid} {name}: a saved logit is not finite")
        return bad, None
    codes = [m[0] for m in mapping]
    code, reason, tied = choose(codes, logits, ask)
    if (path.get("choice_code"), path.get("selection_reason"), path.get("tied_codes")) != (code, reason, tied):
        bad.append(f"{rid} {name}: the recorded decision differs from the one its saved logits give ({code})")
    if any(abs(a - (o.get("restricted_share") or -1.0)) > 1e-12 for a, o in zip(shares(logits), offered)):
        bad.append(f"{rid} {name}: a recorded share differs from its recomputation by more than 1e-12")
    return bad, logits


def compare_replay(*, bundle, results, out, push_receipt=None) -> dict:
    """Compare, publish the folder whatever the verdict, and return the summary."""
    out = output.refuse_existing(out)
    b, r = Path(bundle), Path(results)
    bad = verify_replay_bundle(b)
    if bad:
        raise EvaluationInputError([issue(str(b), "E_REPLAY_COMPARE", "the bundle does not read back: " + "; ".join(bad[:3]))])
    try:
        root = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
        hm = json.loads((b / "headset" / "replay-manifest.json").read_text(encoding="utf-8"))
        recs = _lines(b / "headset" / "requests.jsonl")
        fixes = _lines(b / "headset" / "fixtures-written.jsonl")
        refs = {x["request_id"]: x["reference"] for x in _lines(b / "reference" / "references.jsonl")}
        ident = json.loads((r / "identity.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as e:
        raise EvaluationInputError([issue(str(r), "E_REPLAY_COMPARE", f"unreadable input: {e}")]) from None
    problems, ask, n_vocab = [], hm["ask_code"], hm["vocab_size"]
    # inputs: the chain from the root manifest to what the device read
    fh = root["files"]
    want_chain = {"manifest_sha256": fh["headset/replay-manifest.json"], "requests_sha256": fh["headset/requests.jsonl"],
                  "fixtures_sha256": fh["headset/fixtures-written.jsonl"]}
    got_chain = {k: (ident.get("bundle") or {}).get(k) for k in want_chain}
    if got_chain != want_chain:
        problems.append(f"inputs: the results read other bundle files than this bundle's ({got_chain})")
    chain = {"root_manifest_sha256": sha256((b / "manifest.json").read_bytes()), "headset_files": want_chain,
             "results_name_them": got_chain == want_chain}
    if push_receipt is not None:
        pr = json.loads(Path(push_receipt).read_text(encoding="utf-8"))
        pushed = {f["name"]: f["sha256"] for f in pr.get("files", [])}
        ok = (pr.get("bundle_manifest_sha256") == chain["root_manifest_sha256"]
              and pushed == {"replay-manifest.json": want_chain["manifest_sha256"],
                             "requests.jsonl": want_chain["requests_sha256"],
                             "fixtures-written.jsonl": want_chain["fixtures_sha256"]})
        chain["push_receipt"] = {"sha256": sha256(Path(push_receipt).read_bytes()), "links_root_to_pushed_files": ok}
        if not ok:
            problems.append("inputs: the push receipt does not name this bundle's root manifest and headset files")
    # runtime identity, as the results report it
    req, rt = ident.get("requested") or {}, ident.get("runtime") or {}
    want_rt = {"requested_n_ctx": hm["context_limit_tokens"], "requested_threads": ACCEPTED_RUNTIME["threads"],
               "requested_n_seq": ACCEPTED_RUNTIME["n_seq"], "requested_flags": ACCEPTED_RUNTIME["flags"],
               "n_vocab": n_vocab, "version": ACCEPTED_RUNTIME["version"]}
    got_rt = {"requested_n_ctx": req.get("n_ctx"), "requested_threads": req.get("threads"), "requested_n_seq": req.get("n_seq"),
              "requested_flags": req.get("flags"), "n_vocab": rt.get("n_vocab"), "version": rt.get("version")}
    problems += [f"runtime: {k} is {got_rt[k]!r}, not {v!r}" for k, v in want_rt.items() if got_rt[k] != v]
    allocated = rt.get("n_ctx")
    if not isinstance(allocated, int) or allocated < hm["context_limit_tokens"]:
        problems.append(f"runtime: the allocated context {allocated!r} is below the protocol's {hm['context_limit_tokens']}")
    log = startup_fields((r / "startup-log.txt").read_text(encoding="utf-8", errors="replace")) \
        if (r / "startup-log.txt").is_file() else {"missing": list(LOG_FIELDS), "kv_types_reported": False}
    # self-checks and fixtures
    sc = json.loads((r / "selfchecks.json").read_text(encoding="utf-8")) if (r / "selfchecks.json").is_file() else {}
    if not (sc.get("all_passed") is True and sc.get("checks") and all(c.get("passed") for c in sc["checks"])):
        problems.append("self-checks: missing or not all passed")
    fx = _lines(r / "fixtures.jsonl") if (r / "fixtures.jsonl").is_file() else []
    if [x.get("request_id") for x in fx] != [x["request_id"] for x in fixes]:
        problems.append("fixtures: not exactly the bundle's written fixtures, in order")
    else:
        for x, want in zip(fx, fixes):
            if (x.get("outcome"), x.get("reason"), x.get("evaluated")) != (want["expected_outcome"], want["expected_reason"], False):
                problems.append(f"fixtures: {want['request_id']} gave {x.get('outcome')}/{x.get('reason')}, evaluated "
                                f"{x.get('evaluated')}, not {want['expected_outcome']}/{want['expected_reason']} before evaluation")
    # coverage and records
    results = _lines(r / "results.jsonl") if (r / "results.jsonl").is_file() else []
    ids = [x.get("request_id") for x in results]
    if ids != [x["request_id"] for x in recs]:
        problems.append(f"coverage: {len(ids)} result records are not exactly the bundle's {len(recs)} requests in order")
    by_id = {x.get("request_id"): x for x in results}
    rows = []
    for rec in recs:
        rid, res = rec["request_id"], by_id.get(rec["request_id"])
        if res is None:
            continue
        n, k = len(rec["token_ids"]), rec["keep_before_command_line"]
        mapping = [[c, t, i] for c, t, i in zip(rec["codes"], rec["targets"], rec["code_token_ids"])]
        inp = res.get("input") or {}
        if (inp.get("outcome"), inp.get("prompt_sha256"), inp.get("token_ids_sha256"), inp.get("mapping_sha256"),
                inp.get("runtime_tokens"), inp.get("first_token_difference")) != \
                ("passed", rec["prompt_sha256"], rec["token_ids_sha256"], rec["mapping_sha256"], n, -1):
            problems.append(f"{rid}: the input checks did not pass with the bundle's hashes and an equal native tokenization")
            continue
        bnd = res.get("boundaries") or {}
        if (bnd.get("n"), bnd.get("keep_before_last_token"), bnd.get("keep_before_command_line")) != (n, n - 1, k):
            problems.append(f"{rid}: boundaries differ from the bundle's")
            continue
        paths = res.get("paths") or {}
        schedule = {"U": [(0, n, 0, n)], "R": [(n - 1, 1, n - 1, n)], "P": [(0, k, 0, k), (k, n - k, k, n)]}
        logits = {}
        for name, prescribed in schedule.items():
            p_bad, lg = _path_problems(rid, name, paths.get(name), prescribed, mapping, n_vocab, ask)
            problems += p_bad
            if lg is not None:
                logits[name] = lg
        ref = refs[rid]
        logits["float32"] = [s["logit"] for s in ref["scores"]]
        codes = [m[0] for m in mapping]
        for a, bname, judge in COMPARISONS:
            row = {"request_id": rid, "comparison": f"{a} vs {bname}", "a": a, "b": bname}
            if a in logits and bname in logits:
                row.update(compare(codes, logits[a], logits[bname], ask, judge))
            else:
                row.update({"pass": False, "failed": ["path not scored"], "tvd": None})
            rows.append(row)
    # summary
    per = {}
    for a, bname, judge in COMPARISONS:
        sel = [x for x in rows if x["comparison"] == f"{a} vs {bname}"]
        tvds = [x["tvd"] for x in sel if x.get("tvd") is not None]
        per[f"{a} vs {bname}"] = {
            "compared": len(sel), "pass": sum(x["pass"] for x in sel), "fail": sum(not x["pass"] for x in sel),
            "best_candidate_failures": sum("best candidate" in x["failed"] for x in sel),
            "ranking_failures": sum("plausible ranking" in x["failed"] for x in sel),
            "tvd_over_0_05": sum(t > TVD_MAX for t in tvds), "tvd_judged": judge,
            "max_tvd": max(tvds) if tvds else None,
            "failing_requests": [x["request_id"] for x in sel if not x["pass"]]}
    failures = [x for x in rows if not x["pass"]]
    accepted = not problems and not failures and len(rows) == 5 * len(recs)
    summary = {"format_version": 1, "record_type": "a25_replay_comparison", "results": r.name,
               "verdict": {"replay_acceptance": "pass" if accepted else "fail", "structural_problems": len(problems),
                           "d101_failures": len(failures), "comparisons": len(rows),
                           "requests_with_a_failure": len({x["request_id"] for x in failures}),
                           "k_v_reporting": "reported by the runtime" if log.get("kv_types_reported") else "incomplete"},
               "per_comparison": per, "problems": problems,
               "failures": [{k: x.get(k) for k in ("request_id", "comparison", "choice_a", "choice_b", "tvd", "failed",
                                                     "changed_pairs")} for x in failures],
               "inputs": chain, "runtime": {"reported": got_rt, "allocated_n_ctx": allocated,
                                            "system_info": rt.get("system_info"), "load_ms": rt.get("load_ms")},
               "startup_log": log, "fixtures": len(fx), "self_checks": len(sc.get("checks") or [])}
    files = {"comparisons.jsonl": encode_jsonl(rows), "summary.json": encode_json(summary),
             "report.md": render(summary).encode("utf-8")}
    inputs = {"bundle_manifest_sha256": chain["root_manifest_sha256"],
              "results_files": {p.name: sha256(p.read_bytes()) for p in sorted(r.iterdir()) if p.is_file()}}
    if push_receipt is not None:
        inputs["push_receipt_sha256"] = chain["push_receipt"]["sha256"]
    manifest = {"format_version": 1, "record_type": "a25_replay_comparison_manifest", "inputs": inputs,
                "files": {k: sha256(v) for k, v in files.items()},
                "code": {f"grounding/quest/{p.name}": sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "runtime": runtime()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for name, data in files.items():
            output._write_file(staging / name, data)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    return summary


def render(s) -> str:
    v = s["verdict"]
    L = ["# Replay comparison (A2.5 delivery 1, step 3; D101, D103)", "",
         f"Results `{s['results']}`. **Replay acceptance: {v['replay_acceptance']}.** "
         f"{v['d101_failures']} of {v['comparisons']} D101 comparisons fail, across {v['requests_with_a_failure']} requests; "
         f"{v['structural_problems']} structural problem(s). K/V reporting: {v['k_v_reporting']}.", "",
         "| Comparison | Pass | Fail | Best-candidate failures | Ranking failures | TVD > 0.05 | Max TVD | TVD judged |",
         "|---|---|---|---|---|---|---|---|"]
    for name, p in s["per_comparison"].items():
        mx = "-" if p["max_tvd"] is None else f"{p['max_tvd']:.10f}"
        L.append(f"| {name} | {p['pass']}/{p['compared']} | {p['fail']} | {p['best_candidate_failures']} | {p['ranking_failures']} | "
                 f"{p['tvd_over_0_05']} | {mx} | {'yes' if p['tvd_judged'] else 'reported only'} |")
    L += ["", "Every failed comparison:", "", "| Request | Comparison | Choice (a) | Choice (b) | TVD | Failed | Changed pairs |",
          "|---|---|---|---|---|---|---|"]
    for f in s["failures"]:
        tvd = "-" if f["tvd"] is None else f"{f['tvd']:.10f}"
        L.append(f"| {f['request_id']} | {f['comparison']} | {f.get('choice_a')} | {f.get('choice_b')} | {tvd} | "
                 f"{', '.join(f['failed'])} | {', '.join(f.get('changed_pairs') or []) or '-'} |")
    log = s["startup_log"]
    L += ["", "Runtime reporting (the startup log's own lines):", ""]
    for name in LOG_FIELDS:
        if name in log:
            L.append(f"- `{name}`: {log[name]['value']} (line {', '.join(map(str, log[name]['lines']))})")
        else:
            L.append(f"- `{name}`: not found")
    if s["problems"]:
        L += ["", "Structural problems:", ""] + [f"- {p}" for p in s["problems"]]
    return "\n".join(L) + "\n"
