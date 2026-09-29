#!/usr/bin/env python3
"""Copy a model's files into the Unity app, for the convert and fill menus (A1.7c).

    python grounding/copy_to_unity.py grounding/models/qwen2.5-0.5b-instruct.json

Reads the model description and grounding/models/<name>/export.json, and writes into quest-app/Assets/:
  SecondEyes/Models/<name>/  in Git: vocab.json, merges.txt and tokenizer_config.json (the tokenizer files Meta's
                             provider reads, copied byte for byte from hf/), LICENSE.txt, README.md, and model.json.
                             model.json names the .sentis file and records the ONNX export's folder and fingerprints
                             for Second Eyes > Convert model, and the tokenizer files' fingerprints for Second Eyes >
                             Fill chat provider
  StreamingAssets/           created if missing; Second Eyes > Convert model saves the .sentis file there
The ONNX files stay in grounding/models/<name>/onnx/: inside the Unity project, Unity would import them, which runs
out of memory for a model this size (D38). The .sentis name carries the start of the ONNX weights' fingerprint (D34),
because Meta's runner copies the file out of the app once and then reuses any copy with the same name.
The whole procedure: docs/setup/quest-model.md.
Needs Python 3.9+ and: python -m pip install -r grounding/requirements.txt (huggingface-hub, for the licence file)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "quest-app" / "Assets"
MODELS = ASSETS / "SecondEyes" / "Models"
TOKENIZER_FILES = ["vocab.json", "merges.txt", "tokenizer_config.json"]   # what Meta's provider reads
QUANTIZATIONS = {"Float16": "f16", "Uint8": "u8", "None": "f32"}          # how weights are rounded: name tag

README = """# {name}

Files for Meta's on-device chat provider, written by `grounding/copy_to_unity.py` (A1.7c). Don't edit them by hand; run the tool again (`docs/setup/quest-model.md`).

- `vocab.json`, `merges.txt`, `tokenizer_config.json`: the tokenizer of {repo} at commit `{commit}`, byte for byte (`.gitattributes`).
- `LICENSE.txt`: that model's licence, which covers these files.
- `model.json`: which `.sentis` file the provider loads, and the fingerprints that Second Eyes → Fill chat provider checks.
- The provider asset in this folder is filled by Second Eyes → Fill chat provider.

The model itself, `{sentis}`, is made by Second Eyes → Convert model in `Assets/StreamingAssets/`, which Git ignores (D34, D35, D38).
"""


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, hint: str = "") -> dict:
    if not path.is_file():
        raise ToolError(f"{path.as_posix()} isn't there." + (f" {hint}" if hint else ""))
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_license(repo: str, commit: str, target: Path) -> bool:
    """Copy the model's LICENSE into the Unity folder. A failure only warns: nothing else depends on it."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, "LICENSE", revision=commit)
    except Exception as err:   # offline, or the model has no LICENSE file
        kept = " The existing copy stays." if target.is_file() else ""
        print(f"Warning: couldn't fetch {repo}'s LICENSE ({type(err).__name__}).{kept} To add it by hand, save "
              f"https://huggingface.co/{repo}/blob/{commit}/LICENSE as {target.relative_to(ROOT).as_posix()}")
        return False
    shutil.copyfile(path, target)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Copy a model's files into the Unity app (docs/setup/quest-model.md).")
    parser.add_argument("description", help="the model description, e.g. grounding/models/qwen2.5-0.5b-instruct.json")
    parser.add_argument("--quantization", choices=list(QUANTIZATIONS), default="Float16",
                        help="how Convert model rounds the weights; goes into the .sentis name (default Float16)")
    args = parser.parse_args(argv)
    try:
        # Everything that can fail runs before anything is written.
        desc_path = Path(args.description)
        desc = load_json(desc_path)
        name = desc["name"]
        model_dir = desc_path.parent / name
        export = load_json(model_dir / "export.json",
                           f"Export first: python grounding/export_onnx.py {desc_path.as_posix()}")
        if not ASSETS.is_dir():
            raise ToolError(f"No Unity project at {ASSETS.as_posix()}")
        files = export["variants"]["fp32"]["files"]
        weights = max(files.items(), key=lambda kv: kv[1]["bytes"])[0]
        sentis = f"{name}-{files[weights]['sha256'][:8]}-{QUANTIZATIONS[args.quantization]}.sentis"

        hf_dir = model_dir / "hf"
        missing = [f for f in TOKENIZER_FILES if not (hf_dir / f).is_file()]
        if missing:
            raise ToolError(f"{hf_dir.as_posix()} has no {', '.join(missing)}. Export again: "
                            f"python grounding/export_onnx.py {desc_path.as_posix()}")
        onnx_dir = model_dir / "onnx"
        onnx_files = sorted({"model.onnx", weights})
        for f in onnx_files:   # sizes only: Convert model checks the fingerprints right before converting
            path = onnx_dir / f
            if f not in files or not path.is_file() or path.stat().st_size != files[f]["bytes"]:
                raise ToolError(f"{path.as_posix()} isn't the file export.json describes. Export again: "
                                f"python grounding/export_onnx.py {desc_path.as_posix()}")
        try:
            onnx_folder = onnx_dir.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            raise ToolError(f"The model folder must be inside the repository ({ROOT.as_posix()}), so Unity can find it")

        unity_dir = MODELS / name
        unity_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = []
        for f in TOKENIZER_FILES:
            shutil.copyfile(hf_dir / f, unity_dir / f)
            tokenizer.append({"file": f, "sha256": sha256_of(unity_dir / f), "bytes": (unity_dir / f).stat().st_size})
        written = TOKENIZER_FILES[:]
        if fetch_license(export["hf_repo"], export["hf_commit"], unity_dir / "LICENSE.txt"):
            written.append("LICENSE.txt")
        (unity_dir / "README.md").write_text(README.format(name=name, repo=export["hf_repo"], commit=export["hf_commit"],
                                                            sentis=sentis), encoding="utf-8")
        record = {"format": 1, "name": name, "hf_repo": export["hf_repo"], "hf_commit": export["hf_commit"],
                  "quantization": args.quantization, "sentis_file": sentis, "onnx_folder": onnx_folder,
                  "onnx_weights_file": weights,
                  "onnx_files": [{"file": f, "sha256": files[f]["sha256"], "bytes": files[f]["bytes"]} for f in onnx_files],
                  "tokenizer_files": tokenizer, "written_by": "grounding/copy_to_unity.py"}
        (unity_dir / "model.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (ASSETS / "StreamingAssets").mkdir(exist_ok=True)
        print(f"Wrote {unity_dir.relative_to(ROOT).as_posix()}/: {', '.join(written + ['README.md', 'model.json'])}")

        folder = unity_dir.relative_to(ASSETS.parent).as_posix()
        if (ASSETS / "StreamingAssets" / sentis).is_file():
            print(f"The converted model is there: Assets/StreamingAssets/{sentis}")
        else:
            print(f"Next, in Unity: select {folder} and run Second Eyes > Convert model. It converts "
                  f"{onnx_folder}/model.onnx ({files[weights]['bytes'] / 1e9:.2f} GB of weights) into:")
            print(f"  Assets/StreamingAssets/{sentis}")
        return 0
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
