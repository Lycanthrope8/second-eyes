# grounding

Language grounding (proposal §6): the scene schema, the relation library for user, drone and object frames, synthetic scenes and commands, the rule-based parser, and fine-tuning and evaluation of the language models. Everything here runs on a PC; on the headset, the model runs inside Meta's provider (docs/meta-ai.md).

| Path | What it is | Since |
|---|---|---|
| `models/<name>.json` | a model description: which model, its shape, chat template and length limits; the headset's provider asset gets the same values | A1.7b |
| `models/<name>/` | the downloaded and exported model files (ignored by Git; recreate with `export_onnx.py`) | A1.7b |
| `prompts/<id>.json` | a fixed prompt: system message, user text and the expected target | A1.7b |
| `meta_runner.py` | a Python mirror of Meta's on-device runner: the model's input and output names and its generation loop | A1.7b |
| `export_onnx.py` | downloads a model and exports it to ONNX the way Meta's runner reads it | A1.7b |
| `reference.py` | makes the PC reference answer that the headset must reproduce | A1.7b |

## Setup

Python 3.9 or newer, and a few GB of downloads the first time:

```
python -m pip install -r grounding/requirements.txt
```

To use a different model, add a description in `models/` and export it; the tools and the headset's provider asset read everything model-specific from there.
