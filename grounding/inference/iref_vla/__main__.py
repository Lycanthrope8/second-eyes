"""Command line for the A2.3a pilot (D81). Exit 0: published (planned exclusions included); 2: invalid input,
identity or verification failure; 3: unexpected execution or write failure (a diagnostic folder may be left)."""
from __future__ import annotations

import argparse
import sys

from ...evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.inference.iref_vla")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="freeze the pilot's exact requests from an A2.2d bundle (no model)")
    for name in ("--bundle", "--tokenizer-dir", "--model-description", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--protocol", default=None, help="defaults to the checked-in pilot-protocol.v1.json")
    r = sub.add_parser("run", help="verify the requests and the checkpoint, run the canary and the pilot")
    for name in ("--requests", "--model-dir", "--tokenizer-dir", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--device", required=True, choices=("cpu", "cuda"))
    r.add_argument("--model-description", default=None)
    r.add_argument("--expected-weights-sha256", default=None,
                   help="the weights' published SHA-256 at the pinned revision, if no download metadata is present")
    a = ap.parse_args(argv)
    try:
        if a.command == "prepare":
            from .prepare import prepare_requests
            m = prepare_requests(bundle=a.bundle, tokenizer_dir=a.tokenizer_dir, model_description=a.model_description,
                                 out=a.out, protocol=a.protocol)
            q = m["requests"]
            print(f"prepared {q['count']} requests for {q['parents']} parents ({m['selection']['eligible']} eligible; "
                  f"selected-list SHA-256 {m['selection']['selected_sha256']}): {q['within_context_limit']} within the "
                  f"context limit, {q['context_budget_exceeded']} over it, {q['empty_candidate_set']} empty scenes")
        else:
            from .run import run_pilot
            s = run_pilot(requests=a.requests, model_dir=a.model_dir, device=a.device, out=a.out,
                          tokenizer_dir=a.tokenizer_dir, model_description=a.model_description,
                          expected_weights_sha256=a.expected_weights_sha256)
            c = s["counts"]
            print(f"completed {c['completed_forwards']} forwards of {c['requests']} requests; {c['empty_scene_bypass']} "
                  f"empty-scene bypasses; {c['context_budget_exceeded']} over the context limit; device {s['device']}, {s['dtype']}")
            for k, x in s["by_view_and_format"].items():
                print(f"  {k}: objects {x['object_choices']}, ASK {x['ask_choices']}, exact ties {x['exact_score_ties']}, "
                      f"median tokens {x['prompt_tokens']['median']}")
        print(f"written to {a.out}")
        return 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"input error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"unexpected failure, no success bundle published: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
