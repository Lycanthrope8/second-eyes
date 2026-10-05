"""Command line for the A2.2b development evaluation (D75): predict, then score, as two separate commands.

Exit 0: a complete run with no technical failure (unsupported, ambiguous, insufficient and no_match results are
ordinary outcomes). 1: a complete run whose records include resolver invalid_input or budget_exceeded results; every
command is still represented. 2: invalid batch input or reference data, or an existing destination (issues printed as
path: code: reason). 3: an unexpected exception (traceback printed) or a write failure. Nothing is published unless
the whole operation succeeded.
"""
from __future__ import annotations

import argparse
import sys
import traceback

from .predict import run_predict
from .protocol import EvaluationInputError, EvaluationOutputError
from .score import run_score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grounding.evaluation.iref_vla",
                                     description="A2.2b: the text-only rules baseline on the pinned IRef-VLA sample.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="queries and resolver results; reads no annotations")
    for flag in ("--scene", "--commands", "--category-map", "--inventory-views", "--relation-config",
                 "--direction-config", "--out"):
        p.add_argument(flag, required=True)
    s = sub.add_parser("score", help="completed predictions against the separate annotations")
    for flag in ("--predictions", "--annotations", "--out"):
        s.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            summary = run_predict(scene=args.scene, commands=args.commands, category_map=args.category_map,
                                  inventory_views=args.inventory_views, relation_config=args.relation_config,
                                  direction_config=args.direction_config, out=args.out)
            v = summary["views"]
            print(f"{args.out}: {summary['totals']['parent_commands']} commands, "
                  f"{summary['totals']['prediction_records']} predictions; parsed "
                  f"{summary['parser']['parsed']}; resolved full {v['full_inventory']['bins']['resolved']}, "
                  f"source-known {v['source_known_nyu']['bins']['resolved']}; technical failures "
                  f"{summary['technical_failures']}; semantic hash {summary['semantic_prediction_hash'][:16]}")
        else:
            summary = run_score(predictions=args.predictions, annotations=args.annotations, out=args.out)
            v = summary["views"]
            print(f"{args.out}: {summary['population']['commands']} commands, "
                  f"{summary['population']['annotation_records']} annotations; source target selected: full "
                  f"{v['full_inventory']['C']}/{v['full_inventory']['N']}, source-known "
                  f"{v['source_known_nyu']['C']}/{v['source_known_nyu']['N']}; technical failures "
                  f"{summary['technical_failures']}")
        return 1 if summary["technical_failures"] else 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"{i['path']}: {i['code']}: {i['message']}")
        print(f"not {'predicted' if args.command == 'predict' else 'scored'}: {len(e.issues)} issue(s)")
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"{i['path']}: {i['code']}: {i['message']}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - reported as an internal failure, never as a result
        print("internal error: the evaluation failed unexpectedly; nothing was published", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
