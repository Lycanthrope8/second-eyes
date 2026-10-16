# perception

Pre-scan perception (proposal §5): object detection on drone keyframes, bearings, size-based ranging, cross-view association and triangulation, scored against Vicon ground truth. The proposal plans a Python prototype first and a C# port for the headset; the headset's C# code lives in `quest-app/Assets/SecondEyes/Perception/` (D39, which closed O6).

| Folder | What it holds | Since |
|---|---|---|
| `detector/` | the PC side of the on-device detector: building the detector package from YOLOX's release, the fixed test images and PC reference, and the headset-PC parity check (`docs/detector.md`) | A1.10b |

Ranging, association and triangulation start in A4 (pre-scan without a drone). A4's frames, transforms and pinhole camera
are a Unity-free C# library in `quest-app/Assets/SecondEyes/Perception/Frames/` (`docs/frames.md`), checked by
`tools/csharp-tests/Frames`.

From A4 on, `localization/` holds the localization's default configuration; the core itself is in the app
(`quest-app/Assets/SecondEyes/Perception/Localization/`), and `docs/localization.md` describes it.

`scan/` holds the automatic scan's default settings (`docs/scan.md`).
