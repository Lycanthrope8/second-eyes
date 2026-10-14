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
    a = sub.add_parser("port-goldens", help="the C# port's acceptance material on real data (criteria 1 and 2)")
    for k in ("--bundle", "--prep", "--work", "--out"):
        a.add_argument(k, required=True)
    a = sub.add_parser("headset", help="the parser folder for the headset (assets, maps, startup check, acceptance)")
    for k in ("--bundle", "--desktop", "--export", "--prep", "--work", "--out"):
        a.add_argument(k, required=True)
    a.add_argument("--maps-from", action="append", required=True)
    a.add_argument("--port-out", help="also write the full acceptance export (criteria 1 and 2 on the PC) there")
    a = sub.add_parser("headset-amend", help="Delivery 2's corrections on the existing parser folder and port export (D117)")
    for k in ("--parser", "--port", "--out-parser", "--out-port"):
        a.add_argument(k, required=True)
    a = sub.add_parser("headset-judge", help="judge the headset's parser-check results (Delivery 3)")
    for k in ("--folder", "--results", "--out"):
        a.add_argument(k, required=True)
    a = sub.add_parser("goldens", help="every golden file the C# port's tests compare with, from A2.4's fixture")
    for k in ("--samples", "--official", "--out"):
        a.add_argument(k, required=True)
    a.add_argument("--lib")
    a.add_argument("--gguf")
    a = sub.add_parser("parser-push", help="push the parser folder (and the model) to the headset, the manifest last")
    a.add_argument("--folder", required=True)
    a.add_argument("--run", required=True)
    a.add_argument("--model")
    a.add_argument("--adb", default="adb")
    a = sub.add_parser("parser-pull", help="pull a finished parser-check results folder into a run")
    a.add_argument("--run", required=True)
    a.add_argument("--results")
    a.add_argument("--adb", default="adb")
    a = sub.add_parser("measure-summary", help="combine the parser sessions' measurement reports of a run (cold, warm, tokens)")
    a.add_argument("--run", required=True)
    a = sub.add_parser("measure-sessions", help="A2.5's measure-report for every session of the newest pull, then the summary")
    a.add_argument("--run", required=True)
    a.add_argument("--goldens", required=True)
    a.add_argument("--references")
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
    elif x.cmd == "headset":
        from . import headset as H
        print(json.dumps(H.build(bundle=x.bundle, desktop=x.desktop, export=x.export, prep=x.prep, work=x.work,
                                 maps_from=x.maps_from, out=x.out, port_out=x.port_out), indent=1))
    elif x.cmd == "headset-amend":
        from . import headset as H
        print(json.dumps(H.amend(parser=x.parser, port=x.port, out_parser=x.out_parser, out_port=x.out_port), indent=1))
    elif x.cmd == "headset-judge":
        from . import headset as H
        v = H.judge(folder=x.folder, results=x.results, out=x.out)
        print(json.dumps({k: v[k] for k in ("prerequisites", "criterion_1_headset", "criterion_2_headset", "criterion_4_headset",
                                            "criterion_7_headset", "decodes", "masks", "resolutions", "readers")}, indent=1))
        print("technical failures:", len(v["technical_failures"]))
    elif x.cmd == "goldens":
        from . import goldens as GD
        print(json.dumps(GD.write_all(samples=x.samples, official=x.official, out=x.out, lib=x.lib, gguf=x.gguf), indent=1))
    elif x.cmd in ("parser-push", "parser-pull"):
        from ..quest.runtime_identity import Adb
        from . import device as DV
        adb = Adb(x.adb)
        r = (DV.push_parser(folder=x.folder, run_id=x.run, model=x.model, adb=adb) if x.cmd == "parser-push"
             else DV.pull_parser(run_id=x.run, results=x.results, adb=adb))
        print(json.dumps({k: v for k, v in r.items() if k != "files"}, indent=1))
    elif x.cmd == "measure-sessions":
        from . import measure as AM
        s = AM.measure_sessions(run_id=x.run, goldens=x.goldens, references=x.references)
        print(json.dumps({k: s[k] for k in ("measured_sessions", "cold_ms", "warm_ms", "kinds", "technical_failures", "folder")}, indent=1))
    elif x.cmd == "measure-summary":
        from . import measure as AM
        s = AM.summarize(run_id=x.run)
        print(json.dumps({k: s[k] for k in ("cold_ms", "warm_ms", "kinds", "technical_failures", "folder")}, indent=1))
    elif x.cmd == "port-goldens":
        from . import acceptance as AC
        print(json.dumps(AC.build(bundle=x.bundle, prep=x.prep, work=x.work, out=x.out), indent=1))
    else:
        from . import compare as CP
        r = CP.compare(bundle=x.bundle, desktop=x.desktop, prep=x.prep, work=x.work, out=x.out)
        print(json.dumps({"criterion_4_desktop": r["criterion_4_desktop"], "masks": r["masks"],
                          "criterion_7_desktop": r["criterion_7_desktop"], "identical": r["identical"],
                          "parents": r["parents"], "agreement": round(r["agreement"], 6),
                          "differences": len(r["differences"]), "outcome_changes": r["outcome_changes"],
                          "target_changes": r["target_changes"], "candidate_changes": r["candidate_changes"],
                          "technical_failures": len(r["technical_failures"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
