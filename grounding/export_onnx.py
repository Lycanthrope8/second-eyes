#!/usr/bin/env python3
"""Download a language model and export it to ONNX the way Meta's on-device runner expects.

    python grounding/export_onnx.py grounding/models/qwen2.5-0.5b-instruct.json
    python grounding/export_onnx.py grounding/models/qwen2.5-0.5b-instruct.json --fp16-weights

The model description (grounding/models/<name>.json) says which Hugging Face model to fetch and
what shape it must have. Everything goes into grounding/models/<name>/, which Git ignores:
  hf/          the downloaded model and its tokenizer files (vocab.json, merges.txt, ...)
  onnx/        model.onnx and model.onnx_data, named the way Meta's runner reads them. optimum's
               post-processing is off (it crashes on Windows at this size, D33); instead our own step
               removes the exporter's second copy of the word table (D47): the last layer reads the one
               table transposed, small constants stay inside model.onnx, and the weights file must end
               below 2 GiB, because Unity misreads what lies past it. onnxruntime checks the scores match
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


LIMIT = 2 ** 31   # Unity misreads what a weights file stores past 2 GiB (r013), so exports must end below it (D47)


def external_end(model) -> int:
    """Where the weights file's last stored value ends, in bytes; 0 if nothing is stored outside model.onnx."""
    import onnx
    tensors = list(model.graph.initializer) + [a.t for n in model.graph.node for a in n.attribute if a.name == "value"]
    end = 0
    for t in tensors:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            info = {e.key: e.value for e in t.external_data}
            end = max(end, int(info.get("offset", 0)) + int(info.get("length", 0)))
    return end


def single_word_table(onnx_dir: Path) -> dict:
    """Remove the exporter's second copy of the word table (D47), and keep the weights file below 2 GiB.

    Qwen2.5 ties its output layer to the word table, but the exporter stores the table a second time, transposed, for
    the last MatMul. That makes the weights file 2.52 GB, and Unity misreads what lies past 2 GiB (r013). This rewrites
    the last layer as Gemm(hidden states, word table, transB=1) on the one table, flattening the hidden states around
    it, keeps small constants inside model.onnx, checks with onnxruntime that the word scores stay the same, and only
    then replaces onnx/. Returns what it did, for export.json; does nothing if the model already has one table.
    """
    import gc
    import shutil
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper
    from meta_runner import NUMPY_TYPES

    graph = onnx.load(str(onnx_dir / "model.onnx"), load_external_data=False).graph
    names = {t.name for t in graph.initializer}
    last = next((n for n in graph.node if "logits" in n.output), None)
    if last is None:
        raise ToolError("The exported graph has no node producing logits.")
    if last.op_type != "MatMul" or last.input[1] not in names:
        end = external_end(onnx.load(str(onnx_dir / "model.onnx"), load_external_data=False))
        return {"state": "already one word table", "weights_end": end}
    lookup = next((n for n in graph.node if n.op_type == "Gather" and n.input[1] == "input_ids" and n.input[0] in names), None)
    if lookup is None:
        raise ToolError("The exported graph has no word-table lookup (Gather on input_ids).")
    copy_name, table_name, hidden = last.input[1], lookup.input[0], last.input[0]

    print("Removing the second copy of the word table (loads the whole model into memory) ...")
    model = onnx.load(str(onnx_dir / "model.onnx"))
    inits = {t.name: t for t in model.graph.initializer}
    table = numpy_helper.to_array(inits[table_name])
    copy = numpy_helper.to_array(inits[copy_name])
    if copy.shape != table.T.shape or not np.array_equal(copy, table.T):
        raise ToolError(f"{copy_name} isn't the word table {table_name} transposed, so the two can't be merged.")
    saved_bytes = int(copy.nbytes)
    vocab, width = table.shape
    del copy, table

    p = "/lm_head/one_table"
    model.graph.initializer.extend([numpy_helper.from_array(np.array([-1, width], np.int64), p + "/flat_shape"),
                                    numpy_helper.from_array(np.array([0], np.int64), p + "/starts"),
                                    numpy_helper.from_array(np.array([2], np.int64), p + "/ends"),
                                    numpy_helper.from_array(np.array([vocab], np.int64), p + "/vocab")])
    new_nodes = [
        helper.make_node("Reshape", [hidden, p + "/flat_shape"], [p + "/flat"], name=p + "/Reshape"),
        helper.make_node("Gemm", [p + "/flat", table_name], [p + "/scores"], name=p + "/Gemm", transB=1),
        helper.make_node("Shape", [hidden], [p + "/hidden_shape"], name=p + "/Shape"),
        helper.make_node("Slice", [p + "/hidden_shape", p + "/starts", p + "/ends"], [p + "/batch_seq"], name=p + "/Slice"),
        helper.make_node("Concat", [p + "/batch_seq", p + "/vocab"], [p + "/scores_shape"], name=p + "/Concat", axis=0),
        helper.make_node("Reshape", [p + "/scores", p + "/scores_shape"], ["logits"], name=p + "/Reshape_1"),
    ]
    at = next(i for i, n in enumerate(model.graph.node) if "logits" in n.output)
    del model.graph.node[at]
    for k, node in enumerate(new_nodes):
        model.graph.node.insert(at + k, node)
    model.graph.initializer.remove(inits[copy_name])

    new_dir = onnx_dir.with_name(onnx_dir.name + ".one_table")
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir()
    # Small tensors, the Constant nodes' values among them, stay inside model.onnx (D47).
    onnx.save_model(model, str(new_dir / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.onnx_data", size_threshold=1024, convert_attribute=False)
    del model, inits
    gc.collect()
    end = external_end(onnx.load(str(new_dir / "model.onnx"), load_external_data=False))
    if end > LIMIT:
        raise ToolError(f"Even with one word table the weights file ends at {end:,} bytes, past 2 GiB, where Unity "
                        f"misreads it. {onnx_dir.as_posix()} was left as it was.")

    print("Checking with onnxruntime that the word scores stay the same ...")
    def scores(folder: Path):
        session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
        types = {i.name: NUMPY_TYPES[i.type] for i in session.get_inputs()}
        ids = (np.arange(1, 9) * 997) % vocab
        feeds = {"input_ids": ids[None, :].astype(types["input_ids"]),
                 "attention_mask": np.ones((1, 8), dtype=types["attention_mask"]),
                 "position_ids": np.arange(8, dtype=types["position_ids"])[None, :]}
        for i in session.get_inputs():
            if i.name.startswith("past_key_values."):
                heads, head_dim = i.shape[1], i.shape[3]
                feeds[i.name] = np.zeros((1, heads, 0, head_dim), dtype=types[i.name])
        result = session.run(["logits"], feeds)[0]
        del session
        gc.collect()
        return result
    before, after = scores(onnx_dir), scores(new_dir)
    difference = float(np.max(np.abs(before - after)))
    same_top = bool(np.array_equal(before.argmax(-1), after.argmax(-1)))
    if not same_top or difference > 1e-3 * max(1.0, float(np.max(np.abs(before)))):
        raise ToolError(f"With one word table the word scores changed (largest difference {difference:.3g}, same top "
                        f"tokens: {same_top}). {onnx_dir.as_posix()} was left as it was; the rewrite is in "
                        f"{new_dir.as_posix()} for a look.")

    for f in onnx_dir.iterdir():   # other files the exporter wrote (tokenizer, config) move along; the debug graph doesn't
        if f.is_file() and f.name not in ("model.onnx", "model.onnx_data", "model_debug.onnx"):
            shutil.copy2(f, new_dir / f.name)
    old_dir = onnx_dir.with_name(onnx_dir.name + ".two_tables")
    if old_dir.exists():
        shutil.rmtree(old_dir)
    onnx_dir.rename(old_dir)
    new_dir.rename(onnx_dir)
    shutil.rmtree(old_dir)
    stale = onnx_dir.parent / "debug"
    if stale.exists():
        shutil.rmtree(stale)   # debug_values.py's values belong to the old graph
    print(f"  removed {copy_name} ({saved_bytes / 1e6:.1f} MB); the word scores now come from {table_name} "
          f"(Gemm, transB=1). Largest difference on 8 tokens: {difference:.3g}; same top tokens: {same_top}. "
          f"The weights file now ends at {end / 1e9:.2f} GB, below 2 GiB.")
    return {"state": "merged", "removed": copy_name, "bytes_saved": saved_bytes,
            "last_layer": f"Gemm(transB=1) on {table_name}", "weights_end": end,
            "check": {"tokens": 8, "largest_difference": difference, "same_top_tokens": same_top}}


def make_fp16_weights(onnx_dir: Path, out_dir: Path) -> int:
    """Round every weight tensor to float16 and back to float32, keeping the model otherwise identical."""
    import shutil
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
    if out_dir.exists():
        shutil.rmtree(out_dir)   # onnx adds to an existing weights file instead of replacing it
    out_dir.mkdir(parents=True)
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
        single = single_word_table(onnx_dir)
        if single["state"] != "merged":
            print(f"The model already has one word table; its weights file ends at {single['weights_end'] / 1e9:.2f} GB.")
        names = check_names(onnx_dir / "model.onnx", desc["architecture"]["max_layers"])
        variants = {"fp32": {"folder": "onnx", "files": fingerprints(onnx_dir)}}
        if args.fp16_weights:
            rounded = make_fp16_weights(onnx_dir, fp16_dir)
            variants["fp16w"] = {"folder": "onnx_fp16w", "files": fingerprints(fp16_dir), "tensors_rounded": rounded,
                                 "rule": f"float32 tensors with at least {FP16_MIN_ELEMENTS} values, rounded via float16"}

        manifest = {"format": 1, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "description": desc_path.as_posix(), "hf_repo": desc["hf_repo"],
                    "hf_revision": desc["hf_revision"], "hf_commit": commit, "onnx_opset": desc["onnx_opset"],
                    "names_check": names, "single_word_table": single, "variants": variants,
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
