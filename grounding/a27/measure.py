"""A2.7 Delivery 3 (D115): the deployed path's measurements across sessions.

    python -m grounding.a27 measure-summary --run RUN

A2.5's measure-report (grounding/quest/measure.py) reports one pulled session, and for a parser session adds its parser
section: the time from submission to a validated target or a decline, the session's first such request (cold) and the
rest (warm), tokens, stage times and outcome kinds. One session gives one cold sample, so Delivery 3 runs several. This
combines every parser report in runs/RUN/raw/measure/: cold across sessions; warm pooled, overall and per scene size;
prompt and generated tokens; decode and resolve times; kinds, with technical failures apart; memory per session. It
writes runs/RUN/raw/measure-a27/<UTC time>/summary.json and report.md and never overwrites. Descriptive measurements
only (D115: measured, not judged).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ..evaluation.iref_vla.protocol import encode_json
from ..quest.measure import DECIDED, stats
from ..quest.replay_device import _fail, _run_folder, _stamp
from ..quest.runtime_identity import REPO


def summarize(*, run_id, repo=None) -> dict:
    repo = Path(repo) if repo is not None else REPO
    run = _run_folder(repo, run_id)
    reports = []
    for p in sorted((run / "raw" / "measure").glob("*/summary.json")) if (run / "raw" / "measure").is_dir() else []:
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("parser"):
            reports.append((p.parent.name, s))
    seen, uniq = set(), []
    for name, s in reports:   # one report per session: a session measured twice counts once (its latest report)
        if s.get("session") in seen:
            uniq = [u for u in uniq if u[1].get("session") != s.get("session")]
        seen.add(s.get("session"))
        uniq.append((name, s))
    if not uniq:
        _fail(f"no parser session reports under {run / 'raw' / 'measure'}; run measure-report after each session's outbox-pull", "measure")
    cold, warm, by_size, gen, prompt, dec_ms, res_ms, kinds, tech, sessions = [], [], {}, [], [], [], [], {}, 0, []
    for name, s in uniq:
        rows = [r for r in s["rows"] if r.get("path") == "parser"]
        decided = [r for r in rows if r["parser_kind"] in DECIDED and r["execution_path"] in ("U", "P") and r["app_observed_ms"] is not None]
        if decided:
            cold.append(decided[0]["app_observed_ms"])
        for r in decided[1:]:
            warm.append(r["app_observed_ms"])
            by_size.setdefault(str(r["objects"]), []).append(r["app_observed_ms"])
        for r in decided:
            gen.append(r["generated_tokens"])
            prompt.append(r["tokens"])
            dec_ms.append((r["stage_ms"] or {}).get("eval"))
            res_ms.append((r["stage_ms"] or {}).get("score"))
        for r in rows:
            kinds[str(r["parser_kind"])] = kinds.get(str(r["parser_kind"]), 0) + 1
        tech += s["parser"]["technical_failures"]
        sessions.append({"report": name, "session": s.get("session"), "cold_ms": decided[0]["app_observed_ms"] if decided else None,
                         "decided": len(decided), "memory": s.get("memory"), "settings": s.get("session_settings")})
    summary = {"format_version": 1, "record_type": "a27_measure_summary", "run_id": run_id, "sessions": sessions,
               "definition": "the time from submission to a validated target or a decline, on the headset clock; cold is each "
                             "session's first decided request, warm the others; technical failures are counted apart",
               "cold_ms": stats(cold), "warm_ms": stats(warm), "warm_ms_by_objects": {k: stats(v) for k, v in sorted(by_size.items())},
               "prompt_tokens": stats(prompt), "generated_tokens": stats(gen), "decode_ms": stats(dec_ms), "resolve_ms": stats(res_ms),
               "kinds": dict(sorted(kinds.items())), "technical_failures": tech,
               "peak_memory_kb": max((x["memory"] or {}).get("after_requests_peak_kb", {}).get("max", 0) or 0 for x in sessions)}
    out = run / "raw" / "measure-a27" / _stamp()
    if out.exists():
        _fail(f"{out} already exists (nothing is overwritten)", "measure")
    out.mkdir(parents=True)
    (out / "summary.json").write_bytes(encode_json(summary))
    f = lambda st, k: "-" if not st.get("n") else f"{st[k] / 1000:.2f}"  # noqa: E731
    L = [f"# A2.7 measurements: run {run_id}, {len(sessions)} sessions", "", summary["definition"] + ".", "",
         "| | n | median s | p90 s | max s |", "|---|---|---|---|---|",
         f"| cold | {summary['cold_ms']['n']} | {f(summary['cold_ms'], 'median')} | {f(summary['cold_ms'], 'p90')} | {f(summary['cold_ms'], 'max')} |",
         f"| warm | {summary['warm_ms']['n']} | {f(summary['warm_ms'], 'median')} | {f(summary['warm_ms'], 'p90')} | {f(summary['warm_ms'], 'max')} |"]
    for k, st in summary["warm_ms_by_objects"].items():
        L.append(f"| warm, {k} objects | {st['n']} | {f(st, 'median')} | {f(st, 'p90')} | {f(st, 'max')} |")
    L += ["", f"Tokens (median): prompt {summary['prompt_tokens'].get('median', '-')}, generated {summary['generated_tokens'].get('median', '-')} "
              "(the end token included). Decode s (median / p90): "
              f"{f(summary['decode_ms'], 'median')} / {f(summary['decode_ms'], 'p90')}; resolve ms (median): {summary['resolve_ms'].get('median', '-')}.",
          f"Kinds: {summary['kinds']}; technical failures: {tech}. Peak memory after requests: {summary['peak_memory_kb']} kB.", "",
          "Frames and heat: tools/profile.py's summary for the same run (OVR Metrics' CSV). Descriptive measurements (D115)."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return dict(summary, folder=str(out))


def measure_sessions(*, run_id, goldens, references=None, repo=None, progress=print) -> dict:
    """Every session of the run's newest outbox pull (outbox-pull --sessions N): each session copied into its own pull
    under raw/outbox-sessions/ (beside raw/outbox, so A2.5's choice of the newest pull is unchanged), A2.5's
    measure-report on each, then the summary across them."""
    from ..quest import measure as MS
    repo = Path(repo) if repo is not None else REPO
    run = _run_folder(repo, run_id)
    box = run / "raw" / "outbox"
    pulls = sorted(p for p in box.iterdir() if p.is_dir()) if box.is_dir() else []
    if not pulls:
        _fail(f"no outbox pull under {box}; run outbox-pull --sessions N first", "measure")
    src = pulls[-1]
    folders = [src / "session"] + (sorted(p for p in (src / "sessions").iterdir() if p.is_dir()) if (src / "sessions").is_dir() else [])
    folders = [q for q in folders if (q / "outcomes.jsonl").is_file() and (q / "session.json").is_file()]
    if not folders:
        _fail(f"the pull {src.name} has no session with outcomes", "measure")
    done = []
    for q in folders:
        sid = json.loads((q / "session.json").read_text(encoding="utf-8")).get("session") or q.name
        dest = run / "raw" / "outbox-sessions" / src.name / sid
        if not dest.exists():
            (dest / "session").mkdir(parents=True)
            for n in ("session.json", "outcomes.jsonl", "startup-log.txt", "session-end.json"):
                if (q / n).is_file():
                    shutil.copyfile(q / n, dest / "session" / n)
        progress(f"measuring session {sid}")
        MS.measure_report(run_id=run_id, goldens=goldens, references=references, pull=dest, repo=repo)
        done.append(sid)
        time.sleep(1.1)   # each report's folder is named by the UTC second, and nothing is overwritten
    return dict(summarize(run_id=run_id, repo=repo), measured_sessions=done)
