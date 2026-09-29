#!/usr/bin/env python3
"""Save onnxruntime's intermediate values for the fixed prompt, for a layer-by-layer comparison with Unity (A1.7c-2).

    python grounding/debug_values.py grounding/models/qwen2.5-0.5b-instruct.json --reference 20260928_A1_r008

Writes, next to the export in grounding/models/<name>/:
  onnx/model_debug.onnx  the exported graph with extra outputs: every value computed outside the decoder layers (the
                         word lookup, the attention mask, the position angles, the final norm), every value inside
                         layer 0, and the output of every later layer. It reads the export's own weights file, so it
                         must sit next to it; it is small, and the other tools ignore it
  debug/index.json       those values in graph order, with their operation, inputs, type and shape
  debug/values.bin       onnxruntime's numbers for them, for one pass over the reference's prompt (as the reference ran it:
                         positions 0 to n-1, a mask of ones, empty caches)
Second Eyes > Test model on the PC, check 7, runs the same pass in Unity and names the first value that differs.
Needs Python 3.9+ and grounding/requirements.txt (onnx, onnxruntime, numpy).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_runner import NUMPY_TYPES  # noqa: E402

MAX_VALUES = 5_000_000          # skip anything bigger (20 MB); the word scores are kept only at the last position
LAYER = re.compile(r"/layers\.(\d+)/")
LAYER_ADD = re.compile(r"^/model/layers\.(\d+)/Add(_\d+)?$")   # the residual additions that end a layer's parts


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def choose(graph, types: dict) -> list:
    """The values to expose, in graph order: (value name, node). Skips constants, graph outputs and untyped values."""
    outputs = {o.name for o in graph.output}
    chosen, last_add = [], {}
    for node in graph.node:
        if node.op_type == "Constant":
            continue
        match = LAYER.search(node.name or "")
        for out in node.output:
            if not out or out in outputs or types.get(out) not in ("float", "int"):
                continue
            if match is None or match.group(1) == "0":
                chosen.append((out, node))
            elif LAYER_ADD.match(node.name or ""):
                last_add[int(match.group(1))] = (out, node)
    order = {node.name: i for i, node in enumerate(graph.node)}
    chosen += sorted(last_add.values(), key=lambda pair: order[pair[1].name])
    return sorted(chosen, key=lambda pair: order[pair[1].name])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Save onnxruntime's intermediate values for check 7 of the model test.")
    parser.add_argument("description", help="the model description, e.g. grounding/models/qwen2.5-0.5b-instruct.json")
    parser.add_argument("--reference", required=True, help="the run holding the PC reference, e.g. 20260928_A1_r008")
    parser.add_argument("--prompt", default="a17-fixed", help="the reference's prompt ID (default a17-fixed)")
    args = parser.parse_args(argv)
    try:
        import onnx
        import onnxruntime as ort
        desc_path = Path(args.description).resolve()
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        arch = desc["architecture"]
        model_dir = desc_path.parent / desc["name"]
        onnx_dir = model_dir / "onnx"
        source = onnx_dir / "model.onnx"
        if not source.is_file():
            raise ToolError(f"{source.relative_to(ROOT).as_posix()} isn't there. Export first: "
                            f"python grounding/export_onnx.py {args.description}")
        ref_path = ROOT / "runs" / args.reference / "raw" / f"reference_{args.prompt}_fp32.json"
        if not ref_path.is_file():
            raise ToolError(f"{ref_path.relative_to(ROOT).as_posix()} isn't there. Make it with grounding/reference.py")
        prompt_ids = json.loads(ref_path.read_text(encoding="utf-8"))["prompt"]["token_ids"]

        print("Adding outputs to the graph ...")
        model = onnx.load(str(source), load_external_data=False)   # the weights stay in model.onnx_data
        inferred = onnx.shape_inference.infer_shapes(model)
        kinds = {onnx.TensorProto.FLOAT: "float", onnx.TensorProto.INT64: "int", onnx.TensorProto.INT32: "int",
                 onnx.TensorProto.BOOL: "int"}
        elem_types = {info.name: info.type.tensor_type.elem_type
                      for info in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input)}
        types = {name: kinds.get(elem) for name, elem in elem_types.items()}
        chosen = choose(model.graph, types)
        if len(chosen) > 2000:
            raise ToolError(f"{len(chosen)} values would be exposed: the graph's node names don't look like the export's "
                            "(/model/layers.N/...), so nothing was written")
        for name, _ in chosen:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, elem_types[name], None))
        debug_model = onnx_dir / "model_debug.onnx"
        onnx.save_model(model, str(debug_model))
        print(f"  {len(chosen)} values exposed; wrote {debug_model.relative_to(ROOT).as_posix()}")

        print("Running onnxruntime over the prompt ...")
        session = ort.InferenceSession(str(debug_model), providers=["CPUExecutionProvider"])
        dtypes = {i.name: NUMPY_TYPES[i.type] for i in session.get_inputs()}
        n = len(prompt_ids)
        feeds = {"input_ids": np.array([prompt_ids], dtype=dtypes["input_ids"]),
                 "attention_mask": np.ones((1, n), dtype=dtypes["attention_mask"]),
                 "position_ids": np.arange(n, dtype=dtypes["position_ids"])[None, :]}
        for i in range(arch["max_layers"]):
            for part in ("key", "value"):
                name = f"past_key_values.{i}.{part}"
                feeds[name] = np.zeros((1, arch["num_key_value_heads"], 0, arch["head_dim"]), dtype=dtypes[name])
        names = [name for name, _ in chosen] + ["logits"]
        results = dict(zip(names, session.run(names, feeds)))

        debug_dir = model_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        entries, offset, skipped = [], 0, 0
        with open(debug_dir / "values.bin", "wb") as f:
            rows = [(name, node) for name, node in chosen] + [("logits", None)]
            for name, node in rows:
                value = np.asarray(results[name])
                if name == "logits":
                    value = value[:, -1, :]   # the word scores at the last position only
                if value.size > MAX_VALUES:
                    skipped += 1
                    continue
                kind = "float" if value.dtype.kind == "f" else "int"
                data = value.astype(np.float32 if kind == "float" else np.int32)
                f.write(data.tobytes(order="C"))
                entries.append({"name": name, "op": node.op_type if node is not None else "(model output)",
                                "node": node.name if node is not None else "logits", "inputs": list(node.input) if node is not None else [],
                                "type": kind, "shape": list(value.shape), "offset": offset, "count": int(value.size),
                                "last_position_only": name == "logits"})
                offset += data.nbytes
        index = {"format": 1, "written_by": "grounding/debug_values.py",
                 "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "model": desc["name"], "reference_run": args.reference, "prompt_id": args.prompt,
                 "prompt_token_ids": prompt_ids, "onnxruntime": ort.__version__, "tensors": entries}
        (debug_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
        print(f"  saved {len(entries)} values ({offset / 1e6:.0f} MB) in {debug_dir.relative_to(ROOT).as_posix()}/"
              + (f"; skipped {skipped} too big to keep" if skipped else ""))
        top = int(np.argmax(results["logits"][0, -1]))
        print(f"  onnxruntime's top token at the last position: {top}. Next, in Unity: Second Eyes > Test model on "
              "the PC, check 7.")
        return 0
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
