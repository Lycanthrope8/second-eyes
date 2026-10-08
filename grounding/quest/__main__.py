"""Command line for A2.5's PC side (D98, D99): `replay-bundle` and `verify-replay-bundle` (laptop).

Exit 0 complete, or a clean readback; 1 a readback that found problems; 2 invalid input, a pinned-identity, selection,
token, boundary or reference mismatch, or an existing destination (nothing published); 3 an output error or an
unexpected failure (nothing published)."""
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
    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
