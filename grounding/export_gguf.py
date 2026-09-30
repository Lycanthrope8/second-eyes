#!/usr/bin/env python3
"""Convert the downloaded model to GGUF, the format llama.cpp reads (A1.8b, D55).

    python grounding/export_gguf.py grounding/models/qwen2.5-0.5b-instruct.json [--outtype q8_0] [--force]

Runs llama.cpp's own converter, convert_hf_to_gguf.py, from third_party/llama.cpp (the release in native/llama.cpp.pin;
fetch it with: python tools/build_llama.py fetch) on grounding/models/<name>/hf, the Hugging Face download that
grounding/export_onnx.py made. It writes grounding/models/<name>/gguf/<name>-<outtype>.gguf and records the file's size
and SHA-256 in export.json under variants.gguf_<outtype>. q8_0 is 8-bit weights; f16 and f32 are for comparisons. The
converter needs torch, transformers, numpy, sentencepiece and protobuf in the active Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA = ROOT / "third_party" / "llama.cpp"
PIN = ROOT / "native" / "llama.cpp.pin"


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def pinned() -> tuple:
    tag, commit = PIN.read_text(encoding="utf-8").split()[:2]
    return tag, commit


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python grounding/export_gguf.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("description", help="e.g. grounding/models/qwen2.5-0.5b-instruct.json")
    parser.add_argument("--outtype", default="q8_0", choices=["q8_0", "f16", "f32"])
    parser.add_argument("--force", action="store_true", help="convert again even if the file is there")
    args = parser.parse_args(argv)
    try:
        desc_path = Path(args.description)
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        model_dir = desc_path.parent / desc["name"]
        hf = model_dir / "hf"
        if not (hf / "config.json").is_file():
            raise ToolError(f"{hf.as_posix()} isn't there. Download it first: python grounding/export_onnx.py "
                            f"{desc_path.as_posix()}")
        tag, commit = pinned()
        if not (LLAMA / "convert_hf_to_gguf.py").is_file():
            raise ToolError("third_party/llama.cpp isn't there. Fetch it first: python tools/build_llama.py fetch")
        head = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode == 0 and head.stdout.strip() != commit:
            raise ToolError(f"third_party/llama.cpp is at {head.stdout.strip()[:8]}, but native/llama.cpp.pin says "
                            f"{tag} ({commit[:8]}). Delete the folder and run: python tools/build_llama.py fetch")
        out = model_dir / "gguf" / f"{desc['name']}-{args.outtype}.gguf"
        if out.exists() and not args.force:
            print(f"{out.as_posix()} is already there; keeping it (--force converts again).")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"Converting {hf.as_posix()} to {args.outtype} with llama.cpp {tag} ...")
            r = subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(hf), "--outtype",
                                args.outtype, "--outfile", str(out)])
            if r.returncode != 0 or not out.exists():
                raise ToolError(f"The converter stopped with code {r.returncode}. If it names a missing package, "
                                "install it with pip and run this again.")
        print("Fingerprinting ...")
        files = {out.name: {"sha256": sha256_of(out), "bytes": out.stat().st_size}}
        export_path = model_dir / "export.json"
        export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.exists() else {"variants": {}}
        export.setdefault("variants", {})[f"gguf_{args.outtype}"] = {
            "folder": "gguf", "files": files, "llama_cpp": f"{tag} ({commit[:8]})", "converter": "convert_hf_to_gguf.py"}
        export_path.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
        f = files[out.name]
        print(f"Done: {out.as_posix()} ({f['bytes'] / 1e6:.0f} MB, SHA-256 {f['sha256'][:8]}); recorded in "
              f"{export_path.as_posix()}")
        return 0
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
