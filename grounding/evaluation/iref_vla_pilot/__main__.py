"""Command line for A2.3b (D82): `baseline`, then `score`, as two separate commands.

Exit 0: published and technically complete, whatever the agreement with the source labels. 1: published, but the
analysis contains technical failures (resolver invalid_input or budget_exceeded) or planned exclusions; every command
is still represented. 2: invalid input, a failed identity or reference check, or an existing destination (issues
printed as path: code: reason; nothing published). 3: an unexpected failure or a write failure (nothing published).
Low agreement is a result, never an exit code.
"""
from __future__ import annotations

import argparse
import sys
import traceback

from ..iref_vla.protocol import EvaluationInputError, EvaluationOutputError
from .policy import ScoringInternalError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.evaluation.iref_vla_pilot",
                                 description="A2.3b: score the saved zero-shot pilot and a matched rules baseline.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("baseline", help="the rules system on the pilot's exact subscenes; reads no annotation or model output")
    for flag in ("--requests", "--bundle", "--relation-config", "--direction-config", "--out"):
        b.add_argument(flag, required=True)
    s = sub.add_parser("score", help="saved model and rules results against the reference annotations")
    for flag in ("--pilot", "--requests", "--bundle", "--rules", "--annotations", "--out"):
        s.add_argument(flag, required=True)
    a = ap.parse_args(argv)
    try:
        if a.command == "baseline":
            from .baseline import run_baseline
            x = run_baseline(requests=a.requests, bundle=a.bundle, relation_config=a.relation_config,
                             direction_config=a.direction_config, out=a.out)
            v = x["views"]
            print(f"{a.out}: {x['counts']['parents']} parents, {x['counts']['rules_records']} rules records; resolved: "
                  + ", ".join(f"{k} {v[k]['R']}/{v[k]['records']}" for k in v)
                  + f"; technical failures {x['technical_failures']}; semantic hash {x['semantic_rules_hash'][:16]}")
            return 1 if x["technical_failures"] else 0
        from .score import run_score
        x = run_score(pilot=a.pilot, requests=a.requests, bundle=a.bundle, rules=a.rules, annotations=a.annotations,
                      out=a.out)
        print(f"{a.out}: {x['population']['parents']} parents, {x['population']['requests']} model requests; source "
              "target selected (C/N): " + ", ".join(f"{k} {m['C']}/{m['N']}" for k, m in x["models"].items())
              + "; rules: " + ", ".join(f"{k} {r['C']}/{r['N']}" for k, r in x["rules"].items()))
        t = x["technical"]
        return 1 if any(t.values()) else 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"{i['path']}: {i['code']}: {i['message']}")
        print(f"nothing published: {len(e.issues)} issue(s)")
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"{i['path']}: {i['code']}: {i['message']}", file=sys.stderr)
        return 3
    except ScoringInternalError as e:
        print(f"internal failure, nothing published: {e}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - reported as an internal failure, never as a result
        print("internal failure: the operation failed unexpectedly; nothing was published", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
