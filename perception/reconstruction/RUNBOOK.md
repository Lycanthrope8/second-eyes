# Runbook · testing the Quest-only pipeline end to end (R1–R3)

For testing, follow the doc "Quest-only reconstruction: test guide": it has the same steps with expected outputs,
pass/fail checks and a results sheet. This file is the short reference.

One ordered procedure for testing everything together. Three machines:
- **Workstation (RTX PC):** the depth comparison on public data, and the depth model's export. Its existing Windows
  Python and CUDA PyTorch only.
- **Laptop:** builds the headset programs, prepares the house video, and drives the headset over USB.
- **Headset:** runs tracking, anchoring, depth, calibration, filtering and fusion.

Commands are PowerShell, run from this folder (`quest_recon`). The parts I couldn't run here are flagged. When
something fails, send me the output named in that step rather than changing versions or thresholds.

## 0. Put it in place
1. Copy `quest_recon` into the Second Eyes repository on both PCs; `perception/reconstruction/` is the proposed
   place (O37).
2. Paste `docs/decision-record.md` into `notes/decisions.md`.
3. If you choose a run phase (O36), set it in `configs/registry.yaml`.

## 1. Workstation: environment and plumbing check
```
py -3 -m venv .venv-qrecon
.\.venv-qrecon\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts\qrecon.py selfcheck
python -m pytest -q tests
```
Expected: `SELFCHECK PASS`, then `17 passed, 4 skipped`. The four skipped tests need a Linux build of the headset
program; I ran them here (`21 passed`).

## 2. Workstation: public sequences and models
```
mkdir data\tum
curl.exe -L -o fr3.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
curl.exe -L -o fr1.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
tar -xzf fr3.tgz -C data\tum
tar -xzf fr1.tgz -C data\tum
git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party\Depth-Anything-V2
git -C third_party\Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 third_party\Depth-Anything-3
git -C third_party\Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
pip install -r requirements-da3-windows.txt
pip install --no-deps -e third_party\Depth-Anything-3
mkdir models
hf download depth-anything/Depth-Anything-V2-Small depth_anything_v2_vits.pth --local-dir models
python -c "from huggingface_hub import model_info as m; print('DAV2', m('depth-anything/Depth-Anything-V2-Small').sha); print('DA3', m('depth-anything/DA3-SMALL').sha)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
- Put the printed DA3 revision into `configs\qrecon.yaml` as `arms.da3s.revision`, and note both revisions.
- The last line must still show the CUDA build and `True`. If PyTorch was replaced, stop.

## 3. Workstation: development runs, freeze, validation
```
python scripts\qrecon.py reference --seq fr3_long_office
python scripts\qrecon.py dense --seq fr3_long_office --arm oracle  --reference <A>
python scripts\qrecon.py dense --seq fr3_long_office --arm dav2s   --reference <A>
python scripts\qrecon.py dense --seq fr3_long_office --arm da3s_w4 --reference <A>
python scripts\qrecon.py dense --seq fr3_long_office --arm da3s_w8 --reference <A>
python scripts\qrecon.py evaluate --run <each run> --reference <A>
```
- `<A>` is the run ID the first command prints.
- The oracle run must pass easily. In the DA3 runs' `keyframes.json`, the landmark `scale` should sit near 1. If
  `agree_share_of_checked` is near zero, stop and send me the files.
- Tune `configs\qrecon.yaml` on this sequence only, if at all.

Then freeze and run validation:
```
python scripts\qrecon.py freeze
python scripts\qrecon.py reference --seq fr1_room
python scripts\qrecon.py dense --seq fr1_room --arm dav2s   --reference <A2>
python scripts\qrecon.py dense --seq fr1_room --arm da3s_w4 --reference <A2>
python scripts\qrecon.py dense --seq fr1_room --arm da3s_w8 --reference <A2>
python scripts\qrecon.py evaluate --run <each run> --reference <A2>
python scripts\qrecon.py compare --baseline <dav2s run> --challenger <da3 run>
```

## 4. Workstation: export the depth model for the headset
```
pip install onnx onnxscript onnxruntime==1.29.0 "ml_dtypes==0.5.4"
python scripts\video_tools.py frames --video 20261110_191353.mp4 --start 30 --end 30.1 --scale 0.5 --out data\one_frame
python scripts\export_dav2_onnx.py --frame-size 960 540 --image data\one_frame\rgb\000900.jpg --out models\dav2s_960x540.onnx
python scripts\export_dav2_onnx.py --frame-size 640 480 --image data\tum\rgbd_dataset_freiburg3_long_office_household\rgb\<any>.png --out models\dav2s_640x480.onnx
```
- Each run must end with `"ok": true`. Its JSON records two checks: the ONNX model against PyTorch, and the
  headset's preprocessing and resizing against Depth Anything V2's own inference.
- If the frame name differs, use any file in `data\one_frame\rgb\`.
- Copy `models\*.onnx` to the laptop. They are about 100 MB each.

## 5. Laptop: build the headset programs (not yet run anywhere)
```
py -3 -m venv .venv-qrecon
.\.venv-qrecon\Scripts\Activate.ps1
pip install -r requirements.txt
pip install "cmake==3.31.10" ninja certifi
$env:SSL_CERT_FILE = (python -m certifi)
if (-not (Test-Path "$HOME\android-ndk-unity")) { New-Item -ItemType Junction -Path "$HOME\android-ndk-unity" -Target "C:\Program Files\Unity\Hub\Editor\6000.3.25f1\Editor\Data\PlaybackEngines\AndroidPlayer\NDK" }
python tools\build_tracking.py --ndk "$HOME\android-ndk-unity"
```

CMake 3.31, not 4: stella and yaml-cpp declare CMake minimums that CMake 4 refuses. `SSL_CERT_FILE`: Python on Windows can lack the root certificate sqlite.org uses. The junction: with a space in the NDK path, Ninja calls clang++ by its 8.3 short name and clang then links as C, without the C++ library. Remove the junction only with `cmd /c rmdir`, never `Remove-Item -Recurse`.
- It finds Unity's NDK as your llama.cpp build does, or takes `--ndk <folder>`.
- It fetches about 700 MB of sources and prebuilt packages, then builds for 20–60 minutes.
- Expected end: `done -> ...tracking\out\android-arm64`, with `bin\qtrack`, `bin\qrecon_quest`, `lib\*.so`,
  `share\orb_vocab.fbow` and `build.json`.
- If it stops, send me the last 60 lines.

## 6. Laptop: prepare the house video
```
python scripts\video_tools.py charuco-calibrate --video 20261110_191353.mp4 --board ref01 --start 0 --end 20 --out calib_s26.json
python scripts\video_tools.py frames --video 20261110_191353.mp4 --every 2 --scale 0.5 --out data\house_15fps
python scripts\video_tools.py frames --video 20261110_191353.mp4 --recording house_20261110_191353 --every 2 --out data\house_boards
```
- **Calibration:** check `rms_px` (under 1), `focal_std_fraction` and any warning. It estimates focal length and k1
  only (`profile: weak`), because a board at about 1 m covers little of the image.
- **Tracking frames:** about 5,100 frames at 960×540, 15 frames a second.
- **Board frames:** about 550 full-resolution frames from the board windows in `configs\references.yaml`. Boards
  are detected at full resolution: at half resolution the markers are only a few pixels across.

## 7. Laptop + headset: reconstruct the house on the Quest
Start the Second Eyes app with its detector running (the normal load), keep the headset on USB, then:
```
python tools\reconstruct_on_quest.py --name house --frames data\house_15fps --board-frames data\house_boards --recording house_20261110_191353 --calib calib_s26.json --frame-scale 0.5 --fps 15 --model models\dav2s_960x540.onnx
```
- **First try:** add `--max-keyframes 20` for a short trial, then run without it. The first push of the frames takes
  several minutes.
- **What it does:** tracks every frame on the headset, anchors with board 01, measures board 02 and the return to
  board 01, predicts depth for each keyframe, and fuses one cloud.
- **Results** in `runs\<run>\`:

| File | What it holds |
|---|---|
| `recon\anchor.json` | Scale, corner fit and held-out check; board 02's scale drift and position; the closure at board 01 |
| `recon\cloud.ply` | The fused cloud, in metres, in board 01's frame. Open it in CloudCompare or MeshLab |
| `recon\stats.json` | Stage times, keyframes, peak memory |
| `tracking\stats.json` | Tracking time per frame, frames tracked, lost events |
| `device.json` | The app in the foreground and the thermal state before and after |

- **To redo only the reconstruction:** add `--tracking-run <run>`.

## 8. Laptop + headset: public sequences on the Quest
```
python tools\reconstruct_on_quest.py --name tum_fr3 --frames data\tum\rgbd_dataset_freiburg3_long_office_household --seq fr3_long_office --model models\dav2s_640x480.onnx
```
Copy that run folder to the workstation, then score it there:
```
python scripts\qrecon.py evaluate-cloud --cloud runs\<run>\recon\cloud.ply --reference <A> --seq fr3_long_office --trajectory runs\<run>\tracking\frame_trajectory.txt --align diagnostic_start
python scripts\qrecon.py dense --seq fr3_long_office --arm dav2s --reference <A> --poses estimated --trajectory runs\<run>\tracking\frame_trajectory.txt --align diagnostic_start
```
- The first command scores the headset's own cloud. The second gives the desktop tracker-pose run (run C) for the
  same trajectory.
- Both are diagnostic: a public sequence has no board, so scale is fitted.
- fr1_room has lens distortion that the reconstruction program doesn't model. Run it, but read its edges with care.
  fr3 has none.
- The validation sequence follows the freeze, as in step 3.

## 9. Send back
- **Workstation:** `manifest.json`, `summary.json` and `evaluation.json` from each run; the export JSONs; the model
  revisions.
- **Laptop:** `tracking\out\android-arm64\build.json` and `calib_s26.json`.
- **Each headset run:** `summary.json`, `device.json`, `recon\anchor.json`, `recon\stats.json`,
  `tracking\stats.json`, and every `stdout.txt`.
- **For the house:** a screenshot of `cloud.ply`.

## Troubleshooting
| Message | Meaning |
|---|---|
| `REFUSED: ... not frozen` | A validation run before the freeze, or a config change after it |
| `anchor refused: ...` | Board 01 seen in too few frames or from too few positions; send `anchor.json` |
| `runtime not built` | Step 5 hasn't finished |
| `expected exactly one headset over adb` | Check `adb devices` |
| Few keyframes calibrated | Read the `reason` fields in `recon\keyframes.json` before changing any threshold |

## Whole recordings (R3.14)

One command per recording; nothing recording-specific to enter (D138):

    python tools/process_recording.py --video 20261111_121118.mp4 --calib calib_s26_lines.json --name h121118
    python tools/merge_clouds.py --recordings runs/h121118_recording.json runs/h120450_recording.json --out house

It finds the boards, extracts frames (with the video's own timestamps), tracks on the headset with a new map after
2 s of lost tracking, prints the coverage report (D135), reconstructs every map that saw a board, and lists the maps
that saw none as unplaced. The merge joins all boards' frames through the best links and reports each room's Level 5
status separately (D137). `python tools/coverage.py --run RUN` reports coverage for any tracking run, old or new;
`video_tools.py source-times` adds source timestamps to a frame folder extracted before R3.14.

R3.15: best-effort anchoring fits every window of every board, and each board's windows together, and takes the
passing fit of the most preferred board (the tightest corner fit within it), or the tightest fit of all when none
passes; board windows that overlap are merged. The merge uses a link from a run whose own anchor failed its check only
when nothing better joins, and reports how far apart several links between the same two boards are. To redo only the
reconstructions of a processed recording: `process_recording.py ... --tracking-run <its tracking run>`.

R3.16: `tools/place_maps.py --recordings runs/<name>_recording.json [...]` places the maps that saw no board by the views
they share with maps that a board placed, in any of the given recordings (D139), reconstructs them on the headset in
that board's frame, and updates the summaries; then merge as before. `--dry-run` only locates and fits.
`--include-failed-anchors` also locates against maps whose anchor failed its check but kept half their keyframes.
