"""On-device object detection, PC side (A1.10b, D83 to D85).

The detector package (quest-app/Assets/SecondEyes/Models/yolox-nano/) wraps YOLOX-Nano's released ONNX file so the
headset and the PC run the same graph: an RGB image in [0, 1] goes in; boxes, scores and class indices come out.
`python -m perception.detector build` makes that file from the release, `reference` makes the PC reference on the fixed
test images, and `parity` compares a headset run with it. See docs/detector.md.
"""
