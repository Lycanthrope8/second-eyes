"""python -m grounding.a27 {export,bundle,desktop,compare} ... (A2.7, D115). See docs/a27-export.md."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.a27")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("export", help="the candidate's merged model to GGUF Q8_0 (pinned converter)")
    a.add_argument("--merged", required=True)
    a.add_argument("--llama", required=True, help="llama.cpp checkout at the pinned release")
    a.add_argument("--out", required=True)
    a = sub.add_parser("bundle", help="prompts, the float32 reference and the frozen headset subset")
    for k in ("--prep", "--work", "--evaluation", "--merged", "--out"):
        a.add_argument(k, required=True)
    a = sub.add_parser("desktop", help="masks and decodes through the host build")
    for k in ("--bundle", "--export", "--base-gguf", "--lib", "--llama", "--out"):
        a.add_argument(k, required=True)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--threads", type=int, default=8)
    a.add_argument("--resume", action="store_true")
    a = sub.add_parser("compare", help="judge the desktop halves of criteria 4 and 7")
    for k in ("--bundle", "--desktop", "--prep", "--work", "--out"):
        a.add_argument(k, required=True)
    x = ap.parse_args(argv)
    if x.cmd == "export":
        from . import export as X
        r = X.export(merged=x.merged, llama_dir=x.llama, out=x.out)
        print(json.dumps({"gguf": r["gguf"]["file"], "sha256": r["gguf"]["sha256"], "bytes": r["gguf"]["bytes"],
                          "file_type": r["gguf"]["metadata"]["file_type"], "tokens": r["gguf"]["metadata"]["n_tokens"],
                          "tensor_types": r["gguf"]["metadata"]["tensor_types"]}, indent=1))
    elif x.cmd == "bundle":
        from . import bundle as B
        m = B.bundle(prep=x.prep, work=x.work, evaluation=x.evaluation, merged=x.merged, out=x.out)
        print(json.dumps({"parents": m["parents"], "partitions": m["partitions"], "subset": m["subset"]["size"],
                          "allocation": m["subset"]["allocation"], "ids_sha256": m["subset"]["ids_sha256"],
                          "mask_sample": len(m["subset"]["mask_sample"])}, indent=1))
    elif x.cmd == "desktop":
        from . import desktop as D
        m = D.run(bundle=x.bundle, export=x.export, base_gguf=x.base_gguf, lib=x.lib, llama_dir=x.llama, out=x.out,
                  workers=x.workers, threads=x.threads, resume=x.resume)
        print(json.dumps({"tokenizer_equal": m["tokenizer_against_base"]["equal"], "masks": m["masks"],
                          "decodes": m["decodes"]}, indent=1))
    else:
        from . import compare as CP
        r = CP.compare(bundle=x.bundle, desktop=x.desktop, prep=x.prep, work=x.work, out=x.out)
        print(json.dumps({"criterion_4_desktop": r["criterion_4_desktop"], "masks": r["masks"],
                          "criterion_7_desktop": r["criterion_7_desktop"], "identical": r["identical"],
                          "parents": r["parents"], "agreement": round(r["agreement"], 6),
                          "differences": len(r["differences"]), "outcome_changes": r["outcome_changes"],
                          "technical_failures": len(r["technical_failures"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
