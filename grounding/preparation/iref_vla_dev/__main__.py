"""python -m grounding.preparation.iref_vla_dev {freeze,prepare} ... (A2.6b)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import dev as D


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m grounding.preparation.iref_vla_dev")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="freeze the partition, cap, sampling, desktop subset and analysis plan")
    f.add_argument("--inventory", required=True, help="the inventory folder (inventory.json)")
    f.add_argument("--official", required=True, help="the pinned official lists (fetch-official-splits)")
    f.add_argument("--out", required=True)
    r = sub.add_parser("prepare", help="prepare the development requests under a frozen specification")
    r.add_argument("--spec", required=True, help="the frozen spec.json")
    r.add_argument("--zip", required=True, help="Scannet.zip")
    r.add_argument("--inventory", required=True)
    r.add_argument("--vocabulary", required=True, help="the pinned NYU_Object_Classes.csv")
    r.add_argument("--work", required=True, help="a new folder for the per-scene chain outputs")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--tokenizer-dir", default=str(D.TOKENIZER))
    r.add_argument("--model-description", default=str(D.MODEL_DESCRIPTION))
    r.add_argument("--review", nargs=2, action="append", metavar=("NAME", "FILE"), default=[],
                   help="a file copied beside the manifest for checking (receipt, inventory summary, proposal)")
    a = p.parse_args(argv)
    try:
        if a.command == "freeze":
            s = D.freeze(inventory_dir=a.inventory, official_dir=a.official, out=a.out)
            print((Path(s["folder"]) / "report.md").read_text(encoding="utf-8"))
            print(f"written to {s['folder']}")
        else:
            m = D.prepare(spec=a.spec, zip_path=a.zip, inventory_dir=a.inventory, vocabulary=a.vocabulary, out=a.out,
                          work=a.work, workers=a.workers, tokenizer_dir=a.tokenizer_dir,
                          model_description=a.model_description, review=dict(a.review))
            print((Path(m["folder"]) / "report.md").read_text(encoding="utf-8"))
            print(f"written to {m['folder']}")
    except (D.DevPrepError, D.R.ReleaseError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
