"""Command line for the A2.2a IRef-VLA metadata adapter (D74).

    python -m grounding.adapters.iref_vla --objects CSV --regions CSV --vocabulary CSV
        [--statements JSON] [--graph JSON] --out NEW_FOLDER

Every supplied file must match its pin (Appendix A of the A2.2a brief). Exit 0: imported; 2: invalid, unsupported or
mismatched input, or an existing output folder (issues printed as file: path: code: reason, no traceback); 3: an
internal error (traceback printed) or a failure to write the output. Nothing is published unless everything passed.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .output import run_import
from .pinned import PINNED
from .sources import AdapterInputError, AdapterOutputError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grounding.adapters.iref_vla",
                                     description="Import the pinned IRef-VLA ScanNet sample (A2.2a).")
    for flag in ("--objects", "--regions", "--vocabulary", "--out"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--statements")
    parser.add_argument("--graph")
    args = parser.parse_args(argv)
    try:
        report = run_import(Path(args.objects), Path(args.regions), Path(args.vocabulary), Path(args.out),
                            statements=Path(args.statements) if args.statements else None,
                            graph=Path(args.graph) if args.graph else None, pins=PINNED)
    except AdapterInputError as e:
        for i in e.issues:
            print(f"{i['file']}: {i['path']}: {i['code']}: {i['message']}")
        print(f"not imported: {len(e.issues)} issue(s)")
        return 2
    except AdapterOutputError as e:
        for i in e.issues:
            print(f"{i['file']}: {i['path']}: {i['code']}: {i['message']}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - reported as an internal failure, never as an import
        print("internal error: the adapter failed unexpectedly; no output was published", file=sys.stderr)
        traceback.print_exc()
        return 3
    c = report["counts"]
    print(f"{args.out}: imported {c['objects']} objects ({c['known_categories']} known categories), "
          f"{c['commands'] if c['commands'] is not None else 0} commands, "
          f"{c['annotations'] if c['annotations'] is not None else 0} annotations; statements {report['statements']}, "
          f"graph {report['graph']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
