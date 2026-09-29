#!/usr/bin/env python3
"""Show where one of a model's stored values lives, to find why Unity reads it wrongly (A1.7c-2).

    python grounding/inspect_onnx.py grounding/models/qwen2.5-0.5b-instruct.json onnx::Expand_431 --unity-value -0.010376

For the named value it prints how it is stored: inside model.onnx, or in the weights file at which offset. It prints
its first numbers as stored. If it lies past 2 GiB in the weights file, it also prints the numbers found where a
reader that keeps only 31 bits of the offset would look, and marks them if they equal --unity-value. It then lists
every stored value that lies wholly or partly past 2 GiB, and any names that would become the same if a reader replaced
characters such as ':' '/' '.' with '_'. Reads model.onnx and a few bytes of the weights file; loads no weights.
Needs Python 3.9+ and grounding/requirements.txt (onnx, numpy).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIMIT = 2 ** 31   # 2 GiB: the largest offset a signed 32-bit number can hold, plus one


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def storage(tensor, onnx):
    """(kind, location, offset, length) of a tensor: kind is 'inline' or 'external'."""
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        info = {e.key: e.value for e in tensor.external_data}
        return "external", info.get("location"), int(info.get("offset", 0)), int(info["length"]) if "length" in info else None
    return "inline", None, None, None


def numbers_at(path: Path, offset: int, count: int, dtype) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(count * np.dtype(dtype).itemsize)
    return np.frombuffer(data, dtype=dtype)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Show where a stored value of the export lives (A1.7c-2).")
    parser.add_argument("description", help="the model description, e.g. grounding/models/qwen2.5-0.5b-instruct.json")
    parser.add_argument("name", help="the stored value's name, e.g. onnx::Expand_431")
    parser.add_argument("--unity-value", type=float, help="the first number Unity has for it, from the model test's check 7")
    args = parser.parse_args(argv)
    try:
        import onnx
        from onnx import numpy_helper
        desc_path = Path(args.description).resolve()
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        onnx_dir = desc_path.parent / desc["name"] / "onnx"
        source = onnx_dir / "model.onnx"
        if not source.is_file():
            raise ToolError(f"{source.relative_to(ROOT).as_posix()} isn't there. Export first.")
        model = onnx.load(str(source), load_external_data=False)

        # The value may be an initializer, or the output of a Constant node.
        tensor, how = next(((t, "an initializer") for t in model.graph.initializer if t.name == args.name), (None, None))
        if tensor is None:
            for node in model.graph.node:
                if node.op_type == "Constant" and args.name in node.output:
                    tensor = next((a.t for a in node.attribute if a.name == "value"), None)
                    how = f"the output of Constant node {node.name or '(unnamed)'}"
        if tensor is None:
            close = [t.name for t in model.graph.initializer if args.name.split("_")[0] in t.name][:10]
            raise ToolError(f"No stored value named {args.name}. Some that look alike: {close}")
        dtype = numpy_helper.helper.tensor_dtype_to_np_dtype(tensor.data_type)
        count = int(np.prod(tensor.dims)) if len(tensor.dims) else 1
        kind, location, offset, length = storage(tensor, onnx)
        print(f"{args.name}: {how}, {onnx.TensorProto.DataType.Name(tensor.data_type)}, shape {list(tensor.dims)} ({count} numbers)")

        if kind == "inline":
            values = numpy_helper.to_array(tensor).reshape(-1)
            print(f"  stored inside model.onnx; first numbers: {np.array2string(values[:5], precision=6)}")
        else:
            weights = onnx_dir / location
            size = weights.stat().st_size
            print(f"  stored in {location} ({size:,} bytes) at offset {offset:,}, {length if length is not None else '?'} bytes"
                  + (f": PAST 2 GiB ({LIMIT:,})" if offset + (length or 0) > LIMIT else ": below 2 GiB"))
            values = numbers_at(weights, offset, count, dtype)
            print(f"  first numbers as stored: {np.array2string(values[:5], precision=6)}")
            if offset >= LIMIT:
                wrapped = offset & (LIMIT - 1)   # what a reader keeping only 31 bits of the offset would use
                guess = numbers_at(weights, wrapped, min(count, 5), dtype)
                mark = ""
                if args.unity_value is not None and len(guess) and abs(float(guess[0]) - args.unity_value) <= 1e-4 * max(1.0, abs(args.unity_value)):
                    mark = f"   <- equals Unity's {args.unity_value}"
                print(f"  at offset {wrapped:,} (the offset without its top bit): {np.array2string(guess, precision=6)}{mark}")

        # Everything the weights file holds past 2 GiB.
        past = []
        for t in model.graph.initializer:
            k, loc, off, ln = storage(t, onnx)
            if k == "external" and ln is not None and off + ln > LIMIT:
                past.append((off, ln, t.name))
        for node in model.graph.node:
            for a in node.attribute:
                if a.name == "value" and a.t is not None and a.t.data_location == onnx.TensorProto.EXTERNAL:
                    k, loc, off, ln = storage(a.t, onnx)
                    if ln is not None and off + ln > LIMIT:
                        past.append((off, ln, f"{node.output[0]} (Constant)"))
        past.sort()
        print(f"\nStored values lying wholly or partly past 2 GiB in the weights file: {len(past)}")
        for off, ln, name in past[:20]:
            print(f"  {name}: offset {off:,}, {ln / 1e6:.1f} MB" + ("  (crosses 2 GiB)" if off < LIMIT else ""))
        if len(past) > 20:
            print(f"  ... and {len(past) - 20} more")

        # Names that would collide under a simple renaming.
        groups = {}
        for t in model.graph.initializer:
            groups.setdefault(re.sub(r"[^A-Za-z0-9_]", "_", t.name), []).append(t.name)
        clashes = [names for names in groups.values() if len(names) > 1]
        print(f"\nStored-value names that become the same when ':' '/' '.' and the like turn into '_': {len(clashes)}")
        for names in clashes[:10]:
            print(f"  {names}")
        return 0
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
