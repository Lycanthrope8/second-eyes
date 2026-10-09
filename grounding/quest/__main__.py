"""Command line for A2.5's PC side (D98, D99, D103): `replay-bundle`, `verify-replay-bundle`, `runtime-identity`,
`replay-push`, `replay-pull`, `replay-compare`, `replay-desktop`, the diagnostic's `replay-repeat` and `repeat-compare`, and
delivery 2's `golden-build`, `golden-verify`, `golden-references`, `golden-push` and `golden-pull` (laptop or PC).

replay-bundle and verify-replay-bundle: exit 0 complete, or a clean readback; 1 a readback that found problems; 2 invalid
input, a pinned-identity, selection, token, boundary or reference mismatch, or an existing destination (nothing
published); 3 an output error or an unexpected failure (nothing published).

runtime-identity: exit 0 current deployed identity verified (A1.8c continuity linked or unverified, as printed); 1 a
check failed or continuity is contradicted; 2 incomplete (nothing failed, something could not be established); 3 not
run (an existing destination, an unreadable policy or an output error). The evidence folder is published for 0, 1 and 2.

replay-push and replay-pull: exit 0 done; 2 refused (no single device, no app folder, a bundle that does not read back,
an unknown run or results folder, or an existing destination), with nothing changed; 3 an output error.

replay-compare: exit 0 replay acceptance passes (every structural check and all D101 comparisons); 1 it fails (the folder
is written either way); 2 unreadable inputs; 3 an output error. replay-desktop: exit 0 done; 2 refused (a bundle that
does not read back, another model file, no host build); 3 a failure (kept beside the destination as .failed)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ..evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError


def _num(x) -> str:
    return f"{x:,}" if isinstance(x, int) else f"{x:,.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.quest")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("replay-bundle", help="freeze the replay requests, their float32 references and the written fixtures")
    for n in ("--requests", "--small-run", "--scores", "--tokenizer-dir", "--out"):
        b.add_argument(n, required=True)
    v = sub.add_parser("verify-replay-bundle", help="read a replay bundle back, alone or against its sources")
    v.add_argument("--bundle", required=True)
    for n in ("--requests", "--small-run", "--scores", "--tokenizer-dir"):
        v.add_argument(n)
    ri = sub.add_parser("runtime-identity", help="the installed app's native library and model file, checked on the laptop "
                                                 "with read-only adb")
    ri.add_argument("--out", required=True)
    ri.add_argument("--adb", default="adb", help="the adb executable (default: adb on PATH)")
    pu = sub.add_parser("replay-push", help="push a replay bundle's headset files to the app (never the references)")
    pu.add_argument("--bundle", required=True)
    pu.add_argument("--run", required=True, help="the run that receives the push receipt")
    pu.add_argument("--adb", default="adb")
    pl = sub.add_parser("replay-pull", help="pull a finished replay results folder and its event log into a run")
    pl.add_argument("--run", required=True)
    pl.add_argument("--results", help="a results folder name (default: the newest with done.json)")
    pl.add_argument("--adb", default="adb")
    cp = sub.add_parser("replay-compare", help="step 3: a replay's raw results against the frozen bundle (D101, five per request)")
    cp.add_argument("--bundle", required=True)
    cp.add_argument("--results", required=True, help="a pulled headset results folder or a desktop results folder")
    cp.add_argument("--run", required=True, help="the run that receives raw/compare/<UTC time>/")
    cp.add_argument("--push-receipt", help="replay-push's receipt, to tie the bundle's root manifest to the pushed files")
    dk = sub.add_parser("replay-desktop", help="the desktop diagnosis: the same bundle and schedule with the host build")
    dk.add_argument("--bundle", required=True)
    dk.add_argument("--run", required=True, help="the run that receives raw/desktop/<UTC time>/")
    dk.add_argument("--model", help="the GGUF (default: grounding/models/qwen2.5-0.5b-instruct/gguf/...q8_0.gguf)")
    dk.add_argument("--library", help="the host build of the wrapper (default: native/out/host/)")
    rr = sub.add_parser("replay-repeat", help="the fixed repeatability and context-history diagnostic (ChatGPT's proposal)")
    rr.add_argument("--bundle", required=True)
    rr.add_argument("--run", required=True, help="the run that receives raw/repeat/<UTC time>/")
    rr.add_argument("--model")
    rr.add_argument("--library")
    rc = sub.add_parser("repeat-compare", help="the diagnostic's D101 comparisons and descriptive comparisons")
    rc.add_argument("--bundle", required=True)
    rc.add_argument("--results", required=True)
    rc.add_argument("--run", required=True, help="the run that receives raw/repeat-compare/<UTC time>/")
    gb = sub.add_parser("golden-build", help="delivery 2: the PC's goldens for the headset's prompt builder (D104)")
    gb.add_argument("--bundle", required=True, help="the accepted A2.2d bundle (a22d-20261005-153413)")
    gb.add_argument("--out-root", required=True, help="a folder that receives goldens-<UTC time>/")
    gb.add_argument("--tokenizer-dir", help="default: grounding/models/qwen2.5-0.5b-instruct/hf (pinned by hash)")
    gv = sub.add_parser("golden-verify", help="read a goldens folder back, rebuilding every document and prompt")
    gv.add_argument("--goldens", required=True)
    gv.add_argument("--tokenizer-dir", help="also check every tokenization with the pinned tokenizer")
    gr = sub.add_parser("golden-references", help="fresh float32 references for every golden (A2.3a's TorchModel)")
    gr.add_argument("--goldens", required=True)
    gr.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    gr.add_argument("--model-dir", help="default: grounding/models/qwen2.5-0.5b-instruct/hf (pinned by hash)")
    gp = sub.add_parser("golden-push", help="push the goldens' headset part to the app")
    gp.add_argument("--goldens", required=True)
    gp.add_argument("--run", required=True)
    gp.add_argument("--adb", default="adb")
    gl = sub.add_parser("golden-pull", help="pull a finished golden-check results folder and its event log into a run")
    gl.add_argument("--run", required=True)
    gl.add_argument("--results")
    gl.add_argument("--adb", default="adb")
    isd = sub.add_parser("inbox-send", help="send one request to the headset's ADB inbox (delivery 4); a session must be running")
    isd.add_argument("--run", required=True)
    isd.add_argument("--goldens", required=True, help="the goldens folder: its snapshots and commands")
    isd.add_argument("--snapshot", required=True, help="an object count (3, 6 or 10) or a snapshot ID")
    src = isd.add_mutually_exclusive_group(required=True)
    src.add_argument("--command-file", help="a golden command file, e.g. commands/iref.input....json")
    src.add_argument("--text", help="a written command text (labelled written: an implementation check only)")
    src.add_argument("--dataset", type=int, help="the snapshot's N-th golden dataset command (1-based)")
    isd.add_argument("--request-id")
    isd.add_argument("--resend", action="store_true", help="deliver this request ID again on purpose (to check duplicate handling)")
    isd.add_argument("--adb", default="adb")
    opl = sub.add_parser("outbox-pull", help="pull the headset's acknowledgements, results and session log into a run")
    opl.add_argument("--run", required=True)
    opl.add_argument("--adb", default="adb")
    ip = sub.add_parser("interactive-pull", help="pull a finished interactive-check results folder (delivery 3) and its event log into a run")
    ip.add_argument("--run", required=True)
    ip.add_argument("--results")
    ip.add_argument("--adb", default="adb")
    pv = sub.add_parser("provenance-collect", help="read-only: the conversion and configuration provenance (records and metadata)")
    pv.add_argument("--model-dir", help="default: grounding/models/qwen2.5-0.5b-instruct")
    pv.add_argument("--out-root", required=True, help="the folder that receives provenance-<UTC time>/")
    pv.add_argument("--search", action="append", default=[], help="a folder to search for run 20261007_A2_r006's records (repeatable)")
    a = ap.parse_args(argv)
    if a.command == "provenance-collect":
        return _provenance(a)
    if a.command in ("inbox-send", "outbox-pull"):
        return _inbox(a)
    if a.command.startswith("golden-") or a.command == "interactive-pull":
        return _golden(a)
    if a.command in ("replay-repeat", "repeat-compare"):
        return _repeat(a)
    if a.command in ("replay-compare", "replay-desktop"):
        return _analysis(a)
    if a.command == "runtime-identity":
        return _runtime_identity(a)
    if a.command in ("replay-push", "replay-pull"):
        return _device(a)
    try:
        from . import replay_bundle as RB
        from ..inference.iref_vla_compare import design as D
        if a.command == "replay-bundle":
            m = RB.build_replay_bundle(requests=a.requests, small_run=a.small_run, scores=a.scores,
                                       tokenizer_dir=a.tokenizer_dir, out=a.out)
            s = m["selection"]
            print(f"replay bundle: {s['count']} requests and {m['fixtures']['count']} written fixtures; policy {m['policy_id']}")
            print("  sources: " + ", ".join(f"{t} {s['by_source'][t]}" for t in RB.SOURCES)
                  + f" ({s['chosen_by_several']} requests chosen by more than one source)")
            print("  longest per format: " + "; ".join(
                f"{f} {s['longest_in_format'][f]['request_id']} ({_num(s['longest_in_format'][f]['input_tokens'])} tokens)"
                for f in D.FORMATS))
            t, c = m["input_tokens"], m["tokens_after_command_boundary"]
            print(f"  input tokens: min {_num(t['min'])}, median {_num(t['median'])}, max {_num(t['max'])}, "
                  f"total {_num(t['total'])}")
            print(f"  tokens after the command-line boundary: min {_num(c['min'])}, median {_num(c['median'])}, "
                  f"max {_num(c['max'])}")
            o = m["fixtures"]["overflow"]
            print(f"  overflow fixture: {o['filler_lines']} filler lines, {_num(o['tokens'])} tokens "
                  f"(one line fewer: {_num(o['tokens_one_line_fewer'])})")
            print(f"  references: {m['inputs']['small_run_id']}, manifest {m['inputs']['small_run_manifest_sha256']}")
            print(f"  list sha256 {s['list_sha256']}")
            print(f"  manifest sha256 {hashlib.sha256((Path(a.out) / 'manifest.json').read_bytes()).hexdigest()}")
            print(f"written to {a.out}")
            return 0
        tok = None
        if a.tokenizer_dir:
            from ..preparation.iref_vla import tokens as T
            tok = T.load_pinned_tokenizer(a.tokenizer_dir)
        if (a.small_run is None) != (a.scores is None) or (a.small_run is not None and a.requests is None) \
                or (tok is not None and a.requests is None):
            print("error: --small-run and --scores go together and need --requests; --tokenizer-dir needs --requests",
                  file=sys.stderr)
            return 2
        bad = RB.verify_replay_bundle(a.bundle, requests=a.requests, small_run=a.small_run, scores=a.scores, tokenizer=tok)
        with_ = [n for n, x in (("requests", a.requests), ("reference run and scores manifest", a.small_run),
                                ("tokenizer", tok)) if x is not None]
        try:   # content, not bytes: a checkout with CRLF line endings holds the same policy
            tracked = json.loads((Path(a.bundle) / "policy.json").read_bytes()) == json.loads(RB.POLICY_PATH.read_bytes())
        except (OSError, ValueError):
            tracked = False
        print(f"readback of {a.bundle}" + (f" against the {', '.join(with_)}" if with_ else " on its own") + ": "
              + ("clean" if not bad else f"{len(bad)} problem(s)"))
        print(f"  policy content equals the tracked {RB.POLICY_PATH.name}: {'yes' if tracked else 'no'}")
        for p in bad:
            print(f"  {p}")
        return 0 if not bad else 1
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"input error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"failure, nothing published: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


def _device(a) -> int:
    from . import replay_device as RD
    from .runtime_identity import Adb
    say = lambda m: print(f"  ... {m}", flush=True)  # noqa: E731
    try:
        if a.command == "replay-push":
            r = RD.push(bundle=a.bundle, run_id=a.run, adb=Adb(a.adb), progress=say)
            print("pushed to " + RD.REMOTE_BUNDLE + ": " + ", ".join(f"{f['name']} ({f['bytes']:,} bytes)" for f in r["files"]))
            print(f"  bundle manifest {r['bundle_manifest_sha256']}; the references stayed on the laptop")
            print(f"receipt {r['receipt']}")
            return 0
        r = RD.pull(run_id=a.run, results=a.results, adb=Adb(a.adb), progress=say)
        d, c = r["done"] or {}, r["checks"]
        print(f"pulled {r['results']} into {r['folder']}: {len(r['files'])} files; missing: {', '.join(r['missing']) or 'none'}")
        print(f"  requests written {d.get('written')} of {d.get('requests')}; result lines {c['results_lines']}, "
              f"matching done.json: {'yes' if c['lines_match_done'] else 'no'}; fixtures as expected "
              f"{d.get('fixtures_as_expected')} of {d.get('fixtures')}; self-checks passed: {d.get('self_checks_passed')}")
        print("collection complete (structure and D101 are checked by replay-compare)" if r["complete"]
              else "collection INCOMPLETE: see pull.json")
        return 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"refused: {i['message']}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"output error: {e}", file=sys.stderr)
        return 3


def _inbox(a) -> int:
    from . import inbox as IB
    from .runtime_identity import Adb
    say = lambda m: print(f"  ... {m}", flush=True)  # noqa: E731
    try:
        if a.command == "inbox-send":
            r = IB.inbox_send(run_id=a.run, goldens=a.goldens, snapshot=a.snapshot, command_file=a.command_file, text=a.text,
                              dataset=a.dataset, request_id=a.request_id, resend=a.resend, adb=Adb(a.adb), progress=say)
            q = r["request"]
            print(f"sent {r['request_id']} to {r['remote']} ({q['command_kind']}: {q['command']['text']!r}; snapshot "
                  f"{q['expected_scene']['snapshot_id'][-8:]}); PC push and rename {r['pc_push_and_rename_s']} s; receipt {r['receipt']}")
            return 0
        r = IB.outbox_pull(run_id=a.run, adb=Adb(a.adb), progress=say)
        print(f"pulled {len(r['files'])} files into {r['folder']} (session {r['session']})")
        for rid, s in sorted(r["requests"].items()):
            ms = s.get("app_observed_ms")
            print(f"  {rid}: ack {s.get('ack', '-')}; {s.get('status', 'no result yet')}"
                  + (f" ({s['reason']})" if s.get("reason") else "") + (f", target {s['target']}" if s.get("target") else "")
                  + (f", path {s['execution_path']}" if s.get("execution_path") else "")
                  + (f", {ms / 1000:.1f} s app-observed" if isinstance(ms, (int, float)) else "")
                  + (f", {s['duplicates']} duplicate(s) ignored" if s.get("duplicates") else ""))
        return 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"refused: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error: {i['message']}", file=sys.stderr)
        return 3


def _provenance(a) -> int:
    import datetime
    from . import provenance as PV
    from .runtime_identity import REPO
    out = Path(a.out_root) / ("provenance-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S"))
    try:
        s = PV.collect(model_dir=a.model_dir or REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct", out=out, search=a.search)
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"refused: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error: {i['message']}", file=sys.stderr)
        return 3
    c = s["counts"]
    print(f"provenance: {c['agrees']} agree, {c['differs']} differ, {c['missing']} missing (read-only; nothing else was written)")
    for r in s["comparison"]:
        if r["status"] != "agrees":
            print(f"  {r['status']}: {r['item']}")
    for m in s["missing"]:
        print(f"  missing: {m}")
    print(f"written to {out}")
    return 0 if c["differs"] == 0 and c["missing"] == 0 else 1


def _stamped(run_id, kind) -> Path:
    import datetime
    from . import replay_device as RD
    folder = RD._run_folder(RD.REPO, run_id)
    return folder / "raw" / kind / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _golden(a) -> int:
    import datetime
    from . import prompt_goldens as PG
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        if a.command == "golden-build":
            out = Path(a.out_root) / f"goldens-{stamp}"
            r = PG.build_goldens(bundle=a.bundle, out=out, tokenizer_dir=a.tokenizer_dir)
            c = r["counts"]
            print(f"goldens: {c['snapshots']} snapshots, {c['goldens']} goldens ({c['dataset_commands']} dataset commands, "
                  f"{c['written_fixtures']} written fixtures); tokenizer {r['tokenizer']}")
            print(f"written to {out}")
            return 0
        if a.command == "golden-verify":
            from ..preparation.iref_vla import tokens as T
            tok = T.load_pinned_tokenizer(a.tokenizer_dir) if a.tokenizer_dir else None
            bad = PG.verify_goldens(a.goldens, tokenizer=tok)
            print(f"readback of {a.goldens}{' with the tokenizer' if tok else ''}: " + ("clean" if not bad else f"{len(bad)} problem(s)"))
            for b in bad[:20]:
                print(f"  {b}")
            return 0 if not bad else 1
        if a.command == "golden-references":
            out = Path(a.goldens).parent / f"{Path(a.goldens).name}-references-{stamp}"
            m = PG.golden_references(goldens=a.goldens, out=out, model_dir=a.model_dir, device=a.device)
            print(f"references: {m['references']} on {m['device_requested']} in {m['seconds']} s; canary {m['canary']}")
            print(f"written to {out}")
            return 0
        from . import golden_device as GD
        from .runtime_identity import Adb
        say = lambda m: print(f"  ... {m}", flush=True)  # noqa: E731
        if a.command == "golden-push":
            r = GD.push_goldens(goldens=a.goldens, run_id=a.run, adb=Adb(a.adb), progress=say)
            print(f"pushed {len(r['files'])} files to {r['remote']} (golden-manifest.json last); receipt {r['receipt']}")
            return 0
        if a.command == "interactive-pull":
            r = GD.pull_interactive(run_id=a.run, results=a.results, adb=Adb(a.adb), progress=say)
            d, c = r["done"] or {}, r["checks"]
            print(f"pulled {r['results']} into {r['folder']}; missing: {', '.join(r['missing']) or 'none'}")
            print(f"  outcomes without the cache: {c['off']['statuses']}; with it: {c['prefix']['statuses']}; "
                  f"lines match done.json: {'yes' if c['lines_match_done'] else 'no'}")
            print(f"  prompts equal to the goldens: {d.get('prompts_equal_goldens')} of {d.get('requests')}; same choice with and "
                  f"without the cache: {d.get('same_choice')}; same offered logits: {d.get('same_offered_logits')}; "
                  f"the cache reused tokens on {d.get('prefix_kept_some')}")
            return 0
        r = GD.pull_goldens(run_id=a.run, results=a.results, adb=Adb(a.adb), progress=say)
        d, c = r["done"] or {}, r["checks"]
        print(f"pulled {r['results']} into {r['folder']}; missing: {', '.join(r['missing']) or 'none'}")
        print(f"  goldens equal the PC's: {d.get('all_ok')} of {d.get('checked')} checked ({d.get('goldens')} shipped); "
              f"result lines match done.json: {'yes' if c['lines_match_done'] else 'no'}")
        return 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"refused: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error: {i['message']}", file=sys.stderr)
        return 3


def _repeat(a) -> int:
    from . import replay_repeat as RR
    try:
        if a.command == "replay-repeat":
            out = _stamped(a.run, "repeat")
            d = RR.run_repeat(bundle=a.bundle, out=out, model=a.model, library=a.library,
                              progress=lambda m: print(f"  ... {m}", flush=True))
            print(f"diagnostic: {d['sequences']}/{d['expected_sequences']} sequences written in {d['total_ms'] / 60000:.1f} min")
            print(f"written to {out}")
            return 0
        out = _stamped(a.run, "repeat-compare")
        s = RR.compare_repeat(bundle=a.bundle, results=a.results, out=out)
        f = s["facts"]
        print(f"D101 on each U/R/P sequence: {f['d101_failures']} of {f['d101_comparisons']} fail (diagnostic; acceptance unchanged)")
        print(f"full rows identical across contexts: {f['repeat_pairs_full_row_identical']} of {f['repeat_pairs']}; "
              f"U' identical to U: {f['uprime_full_row_identical']} of {f['uprime_pairs']}; "
              f"path differences identical across contexts: {f['path_differences_identical']} of {f['path_difference_pairs']}")
        for x in s["statements"]:
            print(f"  - {x}")
        for p in s["problems"]:
            print(f"  problem: {p}")
        print(f"written to {out}")
        return 1 if s["problems"] else 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"refused: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error: {i['message']}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"stopped: {type(e).__name__}: {e} (what was written is kept as <folder>.failed)", file=sys.stderr)
        return 3


def _analysis(a) -> int:
    try:
        if a.command == "replay-compare":
            from . import replay_compare as RC
            out = _stamped(a.run, "compare")
            s = RC.compare_replay(bundle=a.bundle, results=a.results, out=out, push_receipt=a.push_receipt)
            v = s["verdict"]
            print(f"replay acceptance: {v['replay_acceptance']}; {v['d101_failures']} of {v['comparisons']} D101 comparisons "
                  f"fail across {v['requests_with_a_failure']} requests; {v['structural_problems']} structural problem(s); "
                  f"K/V reporting: {v['k_v_reporting']}")
            for name, p in s["per_comparison"].items():
                mx = "-" if p["max_tvd"] is None else f"{p['max_tvd']:.10f}"
                print(f"  {name:16} pass {p['pass']:>2}/{p['compared']}  best {p['best_candidate_failures']}  ranking "
                      f"{p['ranking_failures']}  TVD>0.05 {p['tvd_over_0_05']}  max TVD {mx}")
            for p in s["problems"][:10]:
                print(f"  problem: {p}")
            kv = s["startup_log"].get("kv")
            if kv:
                print(f"  K ({kv['k_type']}) {kv['k_size']}, V ({kv['v_type']}) {kv['v_size']}, total {kv['size']} "
                      f"(startup log line {kv['line']})")
            print(f"written to {out}")
            return 0 if v["replay_acceptance"] == "pass" else 1
        from . import replay_desktop as DK
        out = _stamped(a.run, "desktop")
        d = DK.run_desktop(bundle=a.bundle, out=out, model=a.model, library=a.library,
                           progress=lambda m: print(f"  ... {m}", flush=True))
        print(f"desktop replay: {d['written']}/{d['requests']} requests written, {d['fixtures_as_expected']}/{d['fixtures']} "
              f"fixtures as expected, self-checks passed: {d['self_checks_passed']}; {d['total_ms'] / 60000:.1f} min")
        print(f"written to {out}")
        return 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"refused: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error: {i['message']}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"failure: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


def _runtime_identity(a) -> int:
    from . import runtime_identity as RI
    try:
        rec = RI.check_runtime_identity(out=a.out, adb=RI.Adb(a.adb), progress=lambda m: print(f"  ... {m}", flush=True))
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"not run: {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 3
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"not run: output error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 3
    for k in RI.CURRENT:
        c = rec["checks"][k]
        print(f"  {k}: {c['status']} - {c['detail']}")
    c = rec["continuity"]
    print(f"  A1.8c continuity: {c['status']} - {c['reason']}")
    for r in c["records"]:
        print(f"    {r['run']}: " + ("no build record here" if not r["found"] else
                                     f"libse_llama.so {r.get('libse_llama_sha256')}; matches: {'yes' if r.get('matches_deployed') else 'no'}"))
    if rec.get("error"):
        print(f"  unexpected error, evidence kept: {rec['error']}")
    print(rec["status"]["line"])
    print(f"written to {a.out}")
    return rec["status"]["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
