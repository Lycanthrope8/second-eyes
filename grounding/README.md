# grounding

Language grounding (proposal §6): the scene schema, the relation library for user, drone and object frames, synthetic scenes and commands, the rule-based parser, and fine-tuning and evaluation of the language models. Everything here runs on a PC; on the headset, the model runs inside Meta's provider (docs/meta-ai.md).

| Path | What it is | Since |
|---|---|---|
| `models/<name>.json` | a model description: which model, its shape, chat template, length limits and the headset provider's run settings (`provider`); Second Eyes → Fill chat provider copies them into the provider asset | A1.7b |
| `models/<name>/` | the downloaded and exported model files (ignored by Git; recreate with `export_onnx.py`) | A1.7b |
| `prompts/<id>.json` | a fixed prompt: system message, user text and the expected target. `a17-fixed` and the presets `a17-front`, `a17-left` and `a17-table` share one scene and system message and differ in their command (D49). The headset's test panel reads a copy in `quest-app/Assets/SecondEyes/Grounding/Prompts/`; `check_headset.py` compares token IDs, so a stale copy can't pass | A1.7b |
| `meta_runner.py` | a Python mirror of Meta's on-device runner: the model's input and output names and its generation loop | A1.7b |
| `export_onnx.py` | downloads a model and exports it to ONNX the way Meta's runner reads it, with one word table and the weights file below 2 GiB (D47) | A1.7b |
| `reference.py` | makes the PC reference answer that the headset must reproduce; with `--candidates`, each listed object's log-probability instead (PyTorch, 32-bit) | A1.7b, A1.8b |
| `scene.py` | what a prompt's scene and answer look like, shared by the references and the headset's jobs: the objects to score, the answer's start, where the cacheable scene ends | A1.8b |
| `export_gguf.py` | converts the downloaded model to GGUF for llama.cpp (8-bit Q8_0 by default) with llama.cpp's own converter, and fingerprints it | A1.8b |
| `llama_headset.py` | llama.cpp on the headset from the command line: prepares a job, pushes the test program and model, runs it, runs llama-bench, and checks the results against the PC references | A1.8b |
| `check_headset.py` | compares the sends in a run's logs (headset, or the editor's Play mode) with the PC reference: the prompt's token IDs, and the answer text token by token, even if the answer didn't finish | A1.7c |
| `debug_values.py` | saves onnxruntime's intermediate values for the fixed prompt, and a copy of the export that exposes them (`onnx/model_debug.onnx`), for check 7 of Second Eyes → Test model on the PC | A1.7c |
| `copy_to_unity.py` | copies a model's tokenizer files into the Unity app and writes `model.json`, which names the converted file and records the ONNX export for Second Eyes → Convert model (`docs/setup/quest-model.md`) | A1.7c |

## Setup

Python 3.9 or newer, and a few GB of downloads the first time:

```
python -m pip install -r grounding/requirements.txt
```

To use a different model, add a description in `models/` and export it; the tools and the headset's provider asset read everything model-specific from there.
