# quest-app

The Unity app that runs on the Meta Quest 3. At deployment, all AI inference happens here.

## Setup record

| Item | Value |
|---|---|
| Unity | 6000.3.25f1 |
| Meta XR Core SDK | 207.0.0 |
| Unity Inference Engine (`com.unity.ai.inference`) | 2.2.1 (D36) |
| XR plug-in provider | OpenXR 1.18.0 |
| Render pipeline | URP (D14) |
| Package name | `com.secondeyes.quest` (D14) |
| Target frame rate | 72 Hz (D13) |

Update this table whenever a version changes, and log the change in `notes/decisions.md`.

## Where code goes

- Our app-level code goes in `Assets/SecondEyes/` (D14).
- Where each component's C# code lives (O6) is decided when the first component has C# code.
- Editor-only tools go in `Assets/SecondEyes/Editor/`: Second Eyes → Convert model (D38) and Second Eyes → Fill chat provider (D37), which share `ModelFiles.cs`.
- Each language model has a folder `Assets/SecondEyes/Models/<model>/` with its tokenizer files and provider asset. The model itself goes in `Assets/StreamingAssets/`, which Git ignores (D35). ONNX files never go into `Assets/`: Unity would import them, which runs out of memory for a model this size (D38). How: `docs/setup/quest-model.md`.
- Commit `Assets/`, `Packages/` and `ProjectSettings/`. Unity's generated folders (`Library/`, `Temp/`, `Logs/`, `Builds/` and similar) are ignored by the root `.gitignore`.

## Build and install

With the headset connected: File → Build Profiles → Android → Build And Run. APKs go in `Builds/`, which Git ignores. While the model is in `StreamingAssets/`, every build carries it (about 1.3 GB), so building and installing take longer (D31).
