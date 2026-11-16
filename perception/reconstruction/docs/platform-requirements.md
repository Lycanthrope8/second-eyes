# Platform requirements

What each piece needs, kept apart from the work itself. "Workstation today" means the RTX PC as it is:
Windows, Python, PyTorch 2.11 with CUDA 12.8, no WSL2 or Linux.

| Piece | Workstation today | Needs |
|---|---|---|
| Harness, self-check, tests | Yes | Python venv with numpy, scipy, OpenCV, PyYAML |
| Depth Anything V2 Small | Yes, expected | PyTorch (present); the repository and one checkpoint file |
| DA3-Small with supplied poses | Yes, expected | PyTorch; the package's Python dependencies. `xformers` is optional in its code and is left out so it can't replace the installed PyTorch. `pycolmap` needs a Windows wheel for your Python version |
| stella_vslam on the desktop | **No** | A C++ build: Linux, WSL2 or Docker; or Visual Studio build tools on Windows. Not required if trajectories come from the Quest build |
| stella_vslam on the Quest | Laptop | `tools/build_tracking.py`: Unity's NDK (as for llama.cpp), CMake and Ninja (`pip install cmake ninja`), Git; it fetches and cross-builds Eigen, SQLite, yaml-cpp, g2o and stella, and uses OpenCV's prebuilt Android SDK |
| MASt3R-SLAM comparison | **No** | Linux with CUDA. Blocked until Linux or WSL2 is authorized. Not needed for the screening decision |
| SpatialLM on the reconstructed clouds | Not established | Its sparse-convolution dependencies are Linux-first. Needed only for the semantic evaluation after the geometry comparison |
| Reconstruction program for the headset | Laptop | Built by the same `tools/build_tracking.py`, with ONNX Runtime's prebuilt Android package (1.29.0) |
| Depth model export | Workstation | `scripts/export_dav2_onnx.py` with the pinned repository and checkpoint; `pip install onnx onnxscript onnxruntime==1.29.0 "ml_dtypes==0.5.4"` |
| Decoding a recorded video into frames | Laptop | `scripts/video_tools.py frames` (OpenCV); with the drone, frames come from the CM4 instead |
| Tracking on the headset | Laptop and headset | adb; `tools/track_on_quest.py` runs the tracker as a command-line program, with the Second Eyes app in the foreground for its normal load |
| Depth resource test | Laptop and headset | The Second Eyes Unity project; a model export the headset runtime can load (next delivery) |

Nothing in this delivery installs system software or changes the workstation outside one project folder and its venv.
