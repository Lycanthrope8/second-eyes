"""Command line for the offline resolver (A2.1e, D71).

    python -m grounding.resolution --scene PATH --command PATH --query PATH --resolver-config PATH
        --relation-config PATH --direction-config PATH [--category-map PATH ...] [--trace] --out DIR

Writes DIR/resolution.json (UTF-8, final LF), never over an existing one. Exit 0: processing completed, whatever the
semantic status (ambiguous, no_match, insufficient_information and unsupported included); 1: budget_exceeded;
2: invalid_input, or refusal to overwrite; 3: an unexpected error (traceback printed, no resolution.json) or a
failure to write the file. The command's text is never read to infer an action or frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from ..contract import validate as contract
from .resolve import invalid_input_record, resolve
from .validate import issue

EXIT = {"completed": 0, "budget_exceeded": 1, "invalid_input": 2}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grounding.resolution",
                                     description="Resolve a structured grounding query offline (A2.1e).")
    for flag in ("--scene", "--command", "--query", "--resolver-config", "--relation-config", "--direction-config",
                 "--out"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--category-map", nargs="+", action="extend", default=[])
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out) / "resolution.json"
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    named = [("scene", args.scene), ("command", args.command), ("query", args.query),
             ("resolver_config", args.resolver_config)] + \
        [(f"category_map[{i}]", p) for i, p in enumerate(args.category_map)]
    records, problems = {}, []
    for label, path in named:
        try:
            data = Path(path).read_bytes()
        except OSError as e:
            problems.append(issue(label, str(path), "E_INPUT_FILE", f"{type(e).__name__}: {e}"))
            continue
        record, parse_issues = contract.parse(label, data)  # duplicate keys, non-finite numbers, BOMs refused
        problems += [issue(i.file, i.path, i.code, i.message) for i in parse_issues]
        records[label] = record
    if problems:
        record = invalid_input_record(problems, validation_s=time.perf_counter() - started)
    else:
        try:
            record = resolve(records["scene"], records["command"], records["query"],
                             resolver_config=records["resolver_config"], relation_config_path=args.relation_config,
                             direction_config_path=args.direction_config,
                             category_maps=[records[f"category_map[{i}]"] for i in range(len(args.category_map))],
                             trace=args.trace)
        except Exception:  # noqa: BLE001 - reported as a failure, never as a semantic result
            traceback.print_exc()
            return 3
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    except FileExistsError:
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    except OSError as e:
        try:
            out.unlink()  # never leave a partial file that could pass for a result
        except OSError:
            pass
        print(f"could not write {out}: {type(e).__name__}: {e}; no result file was written", file=sys.stderr)
        return 3
    result = record["result"]
    summary = record["processing_status"]
    if result is not None:
        summary += f": {result['status']}" + (f" {result['target_id']}" if result["target_id"] else "")
    elif record["budget"] is not None:
        summary += f": {record['budget']['name']}"
    else:
        summary += ": " + ", ".join(sorted({i["code"] for i in record["issues"]}))
    print(f"{out}: {summary}")
    return EXIT[record["processing_status"]]


if __name__ == "__main__":
    sys.exit(main())
