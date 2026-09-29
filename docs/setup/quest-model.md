# Putting a language model into the Quest app

How a model gets from `grounding/models/` into the app, for Meta's on-device chat provider (A1.7c). Repeat it whenever the model, its export or its conversion changes. Meta's and Unity's side of it is in `docs/meta-ai.md`.

## What goes where

| Where | What | In Git |
|---|---|---|
| `grounding/models/<model>/onnx/` | the ONNX export, read by the convert menu from there | no |
| `quest-app/Assets/SecondEyes/Models/<model>/` | the tokenizer files (`vocab.json`, `merges.txt`, `tokenizer_config.json`), `LICENSE.txt`, `README.md` and `model.json`, all written by `grounding/copy_to_unity.py`; and the provider asset, filled by Second Eyes → Fill chat provider | yes; the tokenizer files byte for byte (`.gitattributes`, D35) |
| `quest-app/Assets/StreamingAssets/<model>-<8 characters>-<quantization>.sentis` | the converted model, made by Second Eyes → Convert model, which every build carries | no (D35) |

ONNX files never go into `Assets/`. Unity would import them at once, and its importer runs out of memory for a model this size (D38).

The 8 characters are the start of the ONNX weights' fingerprint, and the name must change whenever the model does (D34). The first time the app uses the model on the headset, Meta's runner copies the `.sentis` file out of the app into the app's data folder. After that, it loads any copy with that name without checking it.

## Steps

1. **Export** the model, if that isn't done: `python grounding/export_onnx.py grounding/models/<model>.json`.
2. **Copy** the small files into the Unity project: `python grounding/copy_to_unity.py grounding/models/<model>.json`. It writes `model.json`, which names the `.sentis` file and records the ONNX export's fingerprints.
3. **Convert.** In Unity, select `Assets/SecondEyes/Models/<model>/` in the Project window and run Second Eyes → Convert model. It checks the ONNX files against `model.json`, converts them with Unity's ONNX converter, rounds the weights as `model.json` says (Float16), and only then saves the `.sentis` file. It takes a few minutes and several GB of memory, and Unity doesn't respond meanwhile, so close big programs such as a web browser first. The Console gets a report with the Inference Engine version and the time each step took.
4. **Fill the provider.** The first time, create the asset in `Assets/SecondEyes/Models/<model>/`: Create → Meta → AI → Provider Assets → On-Device → Unity Inference Engine. Select it, then run Second Eyes → Fill chat provider. It takes every value from the model description (including its `provider` section), the prompt file and `model.json`, checks everything before writing, and lists every setting in the Console. Run it again after any change to those files.

## On the headset

- The copy is at `/sdcard/Android/data/com.secondeyes.quest/files/<name>.sentis`, next to the `logs/` folder. Old copies stay there after the name changes, about 1.3 GB each. List them with `adb shell ls -l /sdcard/Android/data/com.secondeyes.quest/files/`, and delete one with `adb shell rm /sdcard/Android/data/com.secondeyes.quest/files/<name>.sentis`.
- If the app is stopped during that first copy, the copy can be left incomplete, and the next start loads it anyway. Delete it with the same `adb shell rm` and start the app again.
