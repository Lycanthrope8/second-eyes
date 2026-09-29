#!/usr/bin/env python3
"""Download a language model and export it to ONNX the way Meta's on-device runner expects.

    python grounding/export_onnx.py grounding/models/qwen2.5-0.5b-instruct.json
    python grounding/export_onnx.py grounding/models/qwen2.5-0.5b-instruct.json --fp16-weights

The model description (grounding/models/<name>.json) says which Hugging Face model to fetch and
what shape it must have. Everything goes into grounding/models/<name>/, which Git ignores:
  hf/          the downloaded model and its tokenizer files (vocab.json, merges.txt, ...)
  onnx/        model.onnx and model.onnx_data, named the way Meta's runner reads them. optimum's
               post-processing is off: its only job here is removing Qwen's duplicate word table,
               and that step crashes on Windows at this size (D33), so the duplicate stays
  onnx_fp16w/  with --fp16-weights: the same model with its weights rounded to 16 bits and back,
               a stand-in for Unity's Float16 quantization when making a PC reference
  export.json  what was exported: model revision, package versions, file fingerprints, name check
Needs Python 3.9+ and: python -m pip install -r grounding/requirements.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

from meta_runner import check_io

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["torch", "transformers", "optimum", "optimum-onnx", "onnx", "onnxruntime", "huggingface-hub"]
FP16_MIN_ELEMENTS = 1024   # only real weight tensors are rounded; small constants stay 32-bit


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def download(desc: dict, hf_dir: Path) -> str:
    from huggingface_hub import HfApi, snapshot_download
    print(f"Downloading {desc['hf_repo']} ({desc['hf_revision']}) into {hf_dir} ...")
    snapshot_download(desc["hf_repo"], revision=desc["hf_revision"], local_dir=str(hf_dir),
                      allow_patterns=["*.json", "*.txt", "*.safetensors"])
    return HfApi().model_info(desc["hf_repo"], revision=desc["hf_revision"]).sha


def check_architecture(desc: dict, hf_dir: Path) -> None:
    """The description's shape numbers must match the model's own config (Meta's runner needs them)."""
    config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    generation = json.loads((hf_dir / "generation_config.json").read_text(encoding="utf-8"))
    eos = generation.get("eos_token_id", config.get("eos_token_id"))
    eos = eos if isinstance(eos, list) else [eos]
    actual = {"max_layers": config["num_hidden_layers"],
              "num_key_value_heads": config["num_key_value_heads"],
              "head_dim": config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"],
              "vocab_size": config["vocab_size"]}
    arch = desc["architecture"]
    wrong = [f"{k}: description {arch[k]}, model {v}" for k, v in actual.items() if arch[k] != v]
    if arch["eos_token_id"] not in eos:
        wrong.append(f"eos_token_id: description {arch['eos_token_id']}, model {eos}")
    if wrong:
        raise ToolError("The model doesn't match its description:\n  " + "\n  ".join(wrong))
    print("Shape matches the description: " + ", ".join(f"{k} {v}" for k, v in actual.items())
          + f", end token {arch['eos_token_id']}")


def export(desc: dict, hf_dir: Path, onnx_dir: Path) -> None:
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as err:
        raise ToolError(f"A model tool is missing ({err.name}). Run: python -m pip install -r grounding/requirements.txt")
    print(f"Exporting to ONNX (opset {desc['onnx_opset']}); this takes a few minutes ...")
    # no_post_process: see the docstring and D33 (the duplicate-weight removal crashes on Windows)
    main_export(str(hf_dir), output=str(onnx_dir), task="text-generation-with-past",
                opset=desc["onnx_opset"], device="cpu", no_post_process=True)


def check_names(onnx_path: Path, layers: int) -> dict:
    import onnx
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    result = check_io([i.name for i in graph.input], [o.name for o in graph.output], layers)
    if not result["ok"]:
        raise ToolError("The exported model's names don't match Meta's runner:\n"
                        f"  missing: {result['missing']}\n  extra inputs: {result['extra_inputs']}")
    print(f"Names match Meta's runner: {len(graph.input)} inputs, {len(graph.output)} outputs.")
    return result


def make_fp16_weights(onnx_dir: Path, out_dir: Path) -> int:
    """Round every weight tensor to float16 and back to float32, keeping the model otherwise identical."""
    import numpy as np
    import onnx
    from onnx import numpy_helper
    print("Making the 16-bit-weights copy (needs about twice the model's size in memory) ...")
    model = onnx.load(str(onnx_dir / "model.onnx"))
    rounded = 0
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT and int(np.prod(init.dims)) >= FP16_MIN_ELEMENTS:
            values = numpy_helper.to_array(init).astype(np.float16).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(values, init.name))
            rounded += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(out_dir / "model.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.onnx_data", size_threshold=1024)
    print(f"Rounded {rounded} weight tensors.")
    return rounded


def fingerprints(folder: Path) -> dict:
    return {p.name: {"sha256": sha256_of(p), "bytes": p.stat().st_size}
            for p in sorted(folder.glob("model.onnx*"))}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(prog="python grounding/export_onnx.py",
                                     description="Download a model and export it to ONNX for Meta's runner.")
    parser.add_argument("description", help="the model description, e.g. grounding/models/qwen2.5-0.5b-instruct.json")
    parser.add_argument("--fp16-weights", action="store_true", help="also make the 16-bit-weights copy")
    args = parser.parse_args(argv)
    try:
        desc_path = Path(args.description)
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        model_dir = desc_path.parent / desc["name"]
        hf_dir, onnx_dir, fp16_dir = model_dir / "hf", model_dir / "onnx", model_dir / "onnx_fp16w"

        commit = download(desc, hf_dir)
        check_architecture(desc, hf_dir)
        if (onnx_dir / "model.onnx").exists():
            print(f"{onnx_dir / 'model.onnx'} already exists; keeping it (delete the folder to export again).")
        else:
            export(desc, hf_dir, onnx_dir)
        names = check_names(onnx_dir / "model.onnx", desc["architecture"]["max_layers"])
        variants = {"fp32": {"folder": "onnx", "files": fingerprints(onnx_dir)}}
        if args.fp16_weights:
            rounded = make_fp16_weights(onnx_dir, fp16_dir)
            variants["fp16w"] = {"folder": "onnx_fp16w", "files": fingerprints(fp16_dir), "tensors_rounded": rounded,
                                 "rule": f"float32 tensors with at least {FP16_MIN_ELEMENTS} values, rounded via float16"}

        manifest = {"format": 1, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "description": desc_path.as_posix(), "hf_repo": desc["hf_repo"],
                    "hf_revision": desc["hf_revision"], "hf_commit": commit, "onnx_opset": desc["onnx_opset"],
                    "names_check": names, "variants": variants,
                    "packages": {p: version(p) for p in PACKAGES}}
        (model_dir / "export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name, variant in variants.items():
            biggest = max(variant["files"].items(), key=lambda kv: kv[1]["bytes"])
            print(f"  {name}: weights in {(model_dir / variant['folder'] / biggest[0]).as_posix()} "
                  f"({biggest[1]['bytes'] / 1e9:.2f} GB)")
        print(f"Done. Wrote {(model_dir / 'export.json').as_posix()}")
        return 0
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
