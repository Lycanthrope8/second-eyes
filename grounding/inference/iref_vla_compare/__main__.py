"""Command line for A2.3d (D95): `prepare` and `rules` (laptop). Exit 0 complete; 2 invalid input or an existing
destination; 3 an unexpected failure (nothing published)."""
from __future__ import annotations

import argparse
import sys

from ...evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.inference.iref_vla_compare")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="freeze the selection, the mappings, the prompts and both tokenizers' token IDs")
    for n in ("--bundle", "--tokenizer-small", "--tokenizer-large", "--out"):
        p.add_argument(n, required=True)
    r = sub.add_parser("rules", help="the unchanged parser and resolver on the models' subscenes")
    for n in ("--requests", "--bundle", "--relation-config", "--direction-config", "--out"):
        r.add_argument(n, required=True)
    a = ap.parse_args(argv)
    try:
        if a.command == "prepare":
            from .prepare import prepare_compare
            m = prepare_compare(bundle=a.bundle, tokenizer_small=a.tokenizer_small, tokenizer_large=a.tokenizer_large, out=a.out)
            c = m["counts"]
            print(f"prepared {c['requests']} requests: {c['parents']} parents x 2 views x 2 formats; "
                  f"selection {m['selection_sha256']}")
            print(f"  requests by object count {c['requests_by_object_count']}")
            for key, st in m["token_stats"].items():
                print(f"  {key}: " + "; ".join(f"{v}/{fm} median {s['median']} max {s['max']}" for v, d in st.items()
                                                for fm, s in d.items()))
            print(f"  over the context limit: {c['context_budget_exceeded']}; identical token IDs across the two "
                  f"tokenizers: {m['identical_token_ids_across_tokenizers']} of {c['requests']}")
            print(f"written to {a.out}")
            return 0
        from .rules import run_compare_rules
        s = run_compare_rules(requests=a.requests, bundle=a.bundle, relation_config=a.relation_config,
                              direction_config=a.direction_config, out=a.out)
        for v, d in s["views"].items():
            print(f"  {v}: {d['records']} records; outcomes {d['outcomes']}; technical failures {d['technical_failures']}")
        print(f"rules on {s['counts']['parents']} parents; written to {a.out}")
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
        print(f"failure, nothing published: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
