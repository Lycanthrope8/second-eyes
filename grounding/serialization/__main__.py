"""Offline serializer command line (A2.1d, D69).

    python -m grounding.serialization --scene PATH --command PATH
        --format coordinates_v2|coordinates_relations_v2
        --relation-config PATH --direction-config PATH
        [--category-map PATH ...] [--max-relation-work-units INTEGER] --out DIR

Records are parsed strictly by the contract's parser before rendering. On success it writes document.jsonl,
static_prefix.jsonl, dynamic_suffix.jsonl and metadata.json (token count not_checked: no tokenizer is chosen here) and
exits 0. A work-budget failure writes only metadata.json and exits 1. Rejected options, records or configuration
print their issues and exit 2 with no files written; so does an --out folder that already holds any output file, so a
failure can never leave an old success looking current. An unexpected internal error prints its traceback and exits 3.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from ..contract import validate as contract
from .constants import FORMATS
from .serializer import SerializationInputError, serialize

OUTPUTS = ("document.jsonl", "static_prefix.jsonl", "dynamic_suffix.jsonl", "metadata.json")


def _read(label, path):
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise SerializationInputError([{"label": label, "path": str(path), "code": "E_INPUT_FILE",
                                        "message": f"{type(e).__name__}: {e}"}]) from e
    record, issues = contract.parse(str(path), data)
    if issues:
        raise SerializationInputError([{"label": label, "path": i.path, "code": i.code, "message": i.message}
                                       for i in issues])
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.serialization", description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--format", required=True, choices=FORMATS)
    ap.add_argument("--relation-config", required=True)
    ap.add_argument("--direction-config", required=True)
    ap.add_argument("--category-map", nargs="*", default=[])
    ap.add_argument("--max-relation-work-units", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    existing = [name for name in OUTPUTS if (out / name).exists()]
    if existing:
        print(f"refusing to overwrite {', '.join(existing)} in {out}", file=sys.stderr)
        return 2
    try:
        scene = _read("scene", args.scene)
        command = _read("command", args.command)
        maps = [_read(f"category_map[{i}]", p) for i, p in enumerate(args.category_map)]
        result = serialize(scene, command, format=args.format, relation_config_path=args.relation_config,
                           direction_config_path=args.direction_config,
                           max_relation_work_units=args.max_relation_work_units, category_maps=maps)
    except SerializationInputError as e:
        for i in e.issues:
            print(f"{i['label']}: {i['path']}: {i['code']}: {i['message']}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001  (reported as a failure, never as a result)
        traceback.print_exc()
        return 3
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(result.metadata, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    if result.status != "ok":
        (out / "metadata.json").write_text(metadata, encoding="utf-8")
        print(f"{result.status}: see {out / 'metadata.json'}", file=sys.stderr)
        return 1
    for name, text in (("document.jsonl", result.document), ("static_prefix.jsonl", result.static_prefix),
                       ("dynamic_suffix.jsonl", result.dynamic_suffix)):
        (out / name).write_bytes(text.encode("utf-8"))
    (out / "metadata.json").write_text(metadata, encoding="utf-8")
    print(f"ok: {result.format}, {result.metadata['n']} objects, {result.metadata['text']['document']['bytes']} bytes, "
          f"token count not_checked; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
