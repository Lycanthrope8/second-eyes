# quest-app

The Unity app that runs on the Meta Quest 3. At deployment, all AI inference happens here.

## Setup record

| Item | Value |
|---|---|
| Unity | 6000.3.25f1 |
| Meta XR Core SDK | 207.0.0 |
| Unity Inference Engine (`com.unity.ai.inference`) | 2.2.1 (D36) |
| System keyboard | on: OVRManager → Quest Features → General → Requires System Keyboard (D42) |
| XR plug-in provider | OpenXR 1.18.0 |
| Render pipeline | URP (D14) |
| Package name | `com.secondeyes.quest` (D14) |
| Target frame rate | 72 Hz (D13) |

Update this table whenever a version changes, and log the change in `notes/decisions.md`.

## Where code goes

- Our app-level code goes in `Assets/SecondEyes/` (D14).
- Each component's headset code goes in `Assets/SecondEyes/<Component>/`, for example `Grounding/` (D39).
- The scene has a `ChatPanel` object with the `ChatPanel` and `UiPointer` components (A1.7c-2, D40, D41). Its text box opens the system keyboard itself (D44). In the editor's Play mode, right-click the Chat Panel component and choose Send or Stop to run the model on the PC (D43).
- Editor-only tools go in `Assets/SecondEyes/Editor/`: Second Eyes → Convert model (D38) and Second Eyes → Fill chat provider (D37), which share `ModelFiles.cs`.
- Each language model has a folder `Assets/SecondEyes/Models/<model>/` with its tokenizer files and provider asset. The model itself goes in `Assets/StreamingAssets/`, which Git ignores (D35). ONNX files never go into `Assets/`: Unity would import them, which runs out of memory for a model this size (D38). How: `docs/setup/quest-model.md`.
- Commit `Assets/`, `Packages/` and `ProjectSettings/`. Unity's generated folders (`Library/`, `Temp/`, `Logs/`, `Builds/` and similar) are ignored by the root `.gitignore`.

## Build and install

With the headset connected: File → Build Profiles → Android → Build And Run. APKs go in `Builds/`, which Git ignores. While the model is in `StreamingAssets/`, every build carries it (about 1.3 GB), so building and installing take longer (D31).

## Android manifest

`Assets/Plugins/Android/AndroidManifest.xml` is written by Meta's tools from the project's settings and committed whenever it changes. Its changes up to A1.7c-1 declare hand tracking (the feature, the `com.oculus.permission.HAND_TRACKING` permission and update rate `LOW`), passthrough (the feature, and a launch screen showing passthrough), and the supported devices Quest 2, Quest Pro, Quest 3 and 3S. Turning on the system keyboard adds one more feature (D42).

Building adds further permissions from Unity and its packages. The installed app of `20260929_A1_r009` (`adb shell dumpsys package com.secondeyes.quest`) requests:

- **Hand tracking:** `com.oculus.permission.HAND_TRACKING`, `horizonos.permission.HAND_TRACKING`, and `horizonos.permission.internal.ACCESS_HAND_TRACKING` (not granted).
- **Network:** `android.permission.INTERNET`.
- **Audio and Bluetooth:** `android.permission.RECORD_AUDIO` (asked at run time; not granted), `android.permission.MODIFY_AUDIO_SETTINGS` and `android.permission.BLUETOOTH`. We asked for none of these. Where they come from, and whether to drop them until speech recognition needs the microphone, is open item O16.
- **Android's own:** `com.secondeyes.quest.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`, which the app declares for its internal messages.

Record the list again after every change that could affect it. The paper's ethics section and the IRB protocol need it.
