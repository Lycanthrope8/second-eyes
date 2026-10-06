# perception

Pre-scan perception (proposal §5): object detection on drone keyframes, bearings, size-based ranging, cross-view association and triangulation, scored against Vicon ground truth. The proposal plans a Python prototype first and a C# port for the headset; where the C# code lives is open item O6.

| Folder | What it holds | Since |
|---|---|---|
| `detector/` | the PC side of the on-device detector: building the detector package from YOLOX's release, the fixed test images and PC reference, and the headset-PC parity check (`docs/detector.md`) | A1.10b |

Ranging, association and triangulation start in A4 (pre-scan without a drone).
