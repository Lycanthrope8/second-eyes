"""Command line for the A2.2d preparation (D78).

Exit codes: 0 the bundle was published (input ceilings may be exceeded: those are reported audit outcomes);
2 invalid input; 3 an unexpected or write failure. Nothing is published unless the whole bundle passed readback.
"""
from __future__ import annotations

import argparse
import sys

from ...evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError
from .prepare import FORMATS, VIEWS, run_preparation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.preparation.iref_vla",
                                 description="Materialize A2.2c's fitting selections and measure both formats (A2.2d).")
    for name in ("--scene", "--commands", "--category-map", "--inventory-views", "--selection-audit", "--relation-config",
                 "--direction-config", "--tokenizer-dir", "--model-description", "--out"):
        ap.add_argument(name, required=True)
    a = ap.parse_args(argv)
    try:
        s = run_preparation(scene=a.scene, commands=a.commands, category_map=a.category_map,
                            inventory_views=a.inventory_views, selection_audit=a.selection_audit,
                            relation_config=a.relation_config, direction_config=a.direction_config,
                            tokenizer_dir=a.tokenizer_dir, model_description=a.model_description, out=a.out)
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"input error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001  (unexpected: nothing was published)
        print(f"unexpected failure, nothing published: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    p = s["population"]
    print(f"prepared {p['pairs']} command/view pairs: {p['fitting_pairs']} fitting, {p['over_object_budget']} over the "
          f"object limit, {p['unassessed_parse']} unassessed; {p['scene_files']} scenes, {p['command_files']} commands, "
          f"{p['rendered_documents']} rendered documents, {p['measurement_rows']} measurement rows")
    for v in VIEWS:
        for f in FORMATS:
            x = s["views"][v][f]
            within = ", ".join(f"{c}: {x['ceilings'][c]['within']}" for c in ("1024", "2048", "4096"))
            print(f"  {v} {f}: median {x['input_tokens']['median']}, max {x['input_tokens']['max']} tokens; within {within}")
    print(f"measurement: {s['measurement_label']}")
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
