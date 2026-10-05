"""Command line for the A2.2c category-complete selection audit (D77).

Exit 0: the audit completed; over-budget and unassessed rows are normal outcomes. 2: invalid input or an existing
destination (issues printed as path: code: reason; nothing published). 3: a write failure or an unexpected exception
(traceback printed); nothing published.
"""
from __future__ import annotations

import argparse
import sys
import traceback

from ...evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError
from .audit import run_audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grounding.subscenes.iref_vla",
                                     description="A2.2c: category-complete selection audit on the pinned IRef-VLA sample.")
    for flag in ("--scene", "--commands", "--category-map", "--inventory-views", "--out"):
        parser.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_audit(scene=args.scene, commands=args.commands, category_map=args.category_map,
                            inventory_views=args.inventory_views, out=args.out)
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"{i['path']}: {i['code']}: {i['message']}")
        print(f"not audited: {len(e.issues)} issue(s)")
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"{i['path']}: {i['code']}: {i['message']}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - reported as an internal failure, never as an audit
        print("internal error: the audit failed unexpectedly; nothing was published", file=sys.stderr)
        traceback.print_exc()
        return 3
    v, lim = summary["views"], summary["primary_object_limit"]
    print(f"{args.out}: {summary['population']['parent_commands']} commands; parsed {v['full_inventory']['P']}; fits at "
          f"{lim}: full {v['full_inventory']['fits']}/{v['full_inventory']['N']}, source-known "
          f"{v['source_known_nyu']['fits']}/{v['source_known_nyu']['N']}; distinct selections "
          f"{v['full_inventory']['distinct_assessed_selections']} and {v['source_known_nyu']['distinct_assessed_selections']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
