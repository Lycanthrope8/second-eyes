"""Command line for A2.3c (D94).

Exit 0: a complete, technically valid result, whatever its accuracy. 1: a verified preparation audit that contains
planned context exclusions (the run refuses it). 2: invalid input, identity or pin mismatch, or an existing
destination. 3: a failed canary or repeat control, or an unexpected runtime or write failure (a diagnostic folder
may be left; nothing is published as a result).
"""
from __future__ import annotations

import argparse
import sys

from ...evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError


def _mb(n) -> str:
    return f"{n / 1e6:.1f} MB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.inference.iref_vla_ordering")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="freeze the crossed-cyclic request bundle on the laptop (no model)")
    for name in ("--base-requests", "--bundle", "--tokenizer-dir", "--model-description", "--out"):
        p.add_argument(name, required=True)
    r = sub.add_parser("run", help="canaries, 128 repeat controls, then every other cell, on the RTX PC")
    for name in ("--requests", "--base-pilot", "--model-dir", "--tokenizer-dir", "--model-description", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--device", required=True, choices=("cuda",))
    r.add_argument("--expected-weights-sha256", default=None)
    s = sub.add_parser("score", help="score the results against the accepted A2.3b targets, on the laptop")
    for name in ("--requests", "--results", "--base-pilot", "--base-scores", "--out"):
        s.add_argument(name, required=True)
    a = ap.parse_args(argv)
    try:
        if a.command == "prepare":
            from .prepare import prepare_ordering
            m = prepare_ordering(base_requests=a.base_requests, bundle=a.bundle, tokenizer_dir=a.tokenizer_dir,
                                 model_description=a.model_description, out=a.out)
            c, t = m["counts"], m["tokens"]
            print(f"prepared {c['cells']} cells from {c['bases']} base requests ({c['identity_cells']} identity, "
                  f"{c['nonidentity_cells']} other)")
            print(f"  bases by object count {c['bases_by_object_count']}; cells by view {c['cells_by_view']}")
            print(f"  tokens {t['min']}-{t['max']} (median {t['median']}); change from the base prompt "
                  f"{t['change_from_base']['min']} to {t['change_from_base']['max']}; context exclusions "
                  f"{c['context_budget_exceeded']}")
            print(f"  disk {_mb(m['disk_bytes'])}; schedule {m['schedule_sha256']}; policy {m['policy_sha256']}")
            if c["context_budget_exceeded"]:
                import json
                from pathlib import Path
                rows = [json.loads(x) for x in (Path(a.out) / "request-index.jsonl").read_text(encoding="utf-8").splitlines()]
                over = sorted({r["base_request_id"] for r in rows if r["context_status"] != "within_context_limit"})
                print(f"  grids with cells over the limit (for review; the run refuses this bundle): {', '.join(over)}")
            print(f"written to {a.out}")
            return 1 if c["context_budget_exceeded"] else 0
        if a.command == "run":
            from .run import run_ordering
            x = run_ordering(requests=a.requests, base_pilot=a.base_pilot, model_dir=a.model_dir, device=a.device,
                             out=a.out, tokenizer_dir=a.tokenizer_dir, model_description=a.model_description,
                             expected_weights_sha256=a.expected_weights_sha256)
            c = x["counts"]
            print(f"completed {c['completed']} of {c['cells']} cells; canaries {c['canaries_accepted']}/{c['canaries']}; "
                  f"repeat controls {c['identity_controls_passed']}/{c['identity_controls']} (largest offered-logit "
                  f"difference {x['repeat_control_max_abs_logit_difference']})")
            f = x["timing_ms"]["forward"]
            print(f"  forward median {f['median']:.1f} ms, p95 {f['p95']:.1f} ms; peak GPU bytes {x['peak_gpu_bytes']}")
            print(f"written to {a.out}")
            return 0
        from .score import score_ordering
        x = score_ordering(requests=a.requests, results=a.results, base_pilot=a.base_pilot, base_scores=a.base_scores,
                           out=a.out)
        for k, b in x["by_view_format"].items():
            t = b["target_agreement"]
            print(f"  {k}: full-grid agreement macro {t['macro_mean']:.3f} (min {t['min']:.3f}, max {t['max']:.3f}); "
                  f"always-B reference {b['reference_policies']['always_B_macro']:.3f}; code-shift change "
                  f"{b['contrasts']['code_assignment']['macro_change_rate']:.3f}; list-shift change "
                  f"{b['contrasts']['list_order']['macro_change_rate']:.3f}")
        print(f"scored {x['counts']['cells']} cells of {x['counts']['commands']} commands; written to {a.out}")
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
        print(f"failure, no result published: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
