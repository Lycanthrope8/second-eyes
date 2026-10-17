"""A scan's PC side (A4 Delivery 1, stage 3; grounding/quest/scan_device.py), with a labelled test double for adb.

Pulls a scan: the newest when none is named, its four files and the keyframe images its facts name, never over an
existing folder, refusing a scan with no facts; and writes the view page with each box scaled into the model's input.

    python grounding/tests/test_scan_device.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from grounding.quest import scan_device as SD  # noqa: E402

FAILED, PASSED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


class FakeDevice:
    """Labelled test double for adb: a dictionary of remote files and one device."""

    def __init__(self, files):
        self.files = dict(files)

    def run(self, *args):
        if args == ("devices",):
            return 0, "List of devices attached\nSERIAL1\tdevice\n", ""
        if args[0] == "pull":
            if args[1] not in self.files:
                return 1, "", "failed to stat remote object"
            Path(args[2]).write_bytes(self.files[args[1]])
            return 0, "1 file pulled\n", ""
        if args[:2] == ("shell", "ls"):
            path = args[-1]
            kids = sorted({k[len(path) + 1:].split("/")[0] for k in self.files if k.startswith(path + "/")})
            return (0, "".join(k + "\n" for k in kids), "") if kids else (1, "", "No such file or directory")
        return 1, "", "unsupported in the fake"

    def shell(self, *args):
        return self.run("shell", *args)


def refuses(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    except Exception:  # noqa: BLE001
        return True
    return False


def main() -> int:
    capture = json.loads((REPO / "grounding" / "tests" / "fixtures" / "capture" / "valid" / "capture.bedroom.json").read_text(encoding="utf-8"))
    facts = {"capture_id": "scan.s1.2", "stamp_kind": "DateTime", "sdk_unit": "100ns", "kept": 2, "failed": 0,
             "keyframes": [{"kept": True, "index": 0, "snapshot": "s1_001", "frame_age_us": 0},
                           {"kept": True, "index": 1, "snapshot": "s1_002", "frame_age_us": 12000}]}
    remote = {f"{SD.REMOTE_SCANS}/scan.s1.1/facts.json": b"{}",
              f"{SD.REMOTE_SCANS}/scan.s1.2/facts.json": json.dumps(facts).encode(),
              f"{SD.REMOTE_SCANS}/scan.s1.2/capture.json": json.dumps(capture).encode(),
              f"{SD.REMOTE_SCANS}/scan.s1.2/scene.json": b"{}",
              f"{SD.REMOTE_SCANS}/scan.s1.10/notes.txt": b"x",
              f"{SD.REMOTE_SNAPSHOTS}/s1_001.png": b"png1", f"{SD.REMOTE_SNAPSHOTS}/s1_002.png": b"png2"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dev = FakeDevice(remote)
        print("Which scan")
        check("the newest is the highest number in the newest session (10 after 2, not before)", SD.newest(dev) == "scan.s1.10")
        print("Pulling")
        r = SD.pull(out=tmp, capture="scan.s1.2", adb=dev, progress=lambda _: None)
        d = tmp / "scan.s1.2"
        check("the files it has come, and no others", r["files"] == ["facts.json", "capture.json", "scene.json"] and not (d / "trace.json").exists())
        check("each keyframe image its facts name comes", (d / "snapshots" / "s1_001.png").read_bytes() == b"png1" and r["snapshots"] == 2)
        check("pull.json records what came", json.loads((d / "pull.json").read_text(encoding="utf-8"))["capture_id"] == "scan.s1.2")
        check("pulling over an existing folder is refused", refuses(lambda: SD.pull(out=tmp, capture="scan.s1.2", adb=dev, progress=lambda _: None)))
        check("a scan with no facts is refused", refuses(lambda: SD.pull(out=tmp / "x", capture="scan.s1.10", adb=dev, progress=lambda _: None)))
        print("The view page")
        page = SD.view(scan=d).read_text(encoding="utf-8")
        ratio = min(416 / 960, 416 / 1280)
        x1 = 412.5 * ratio
        check("it shows each kept keyframe's image", page.count("<img ") == 2 and "snapshots/s1_001.png" in page)
        check("each box is scaled into the model's input by the letterbox ratio", f'x="{x1:.1f}"' in page, f"{x1:.1f}")
        check("it names the timestamp kind and each frame's age", "DateTime" in page and "frame age 12 ms" in page)
        check("without the trace's fates, the boxes keep one style", "<rect class=" not in page and "<rect " in page)
        print("The view page with each detection's fate")
        (d / "trace.json").write_text(json.dumps({"record_type": "localization_trace", "detections": [
            {"keyframe": 0, "detection": 0, "label": "chair", "score": 0.87, "fate": "object", "object_id": "obj_001", "sighting": 0, "joined_by_ray": False},
            {"keyframe": 0, "detection": 1, "label": "dining table", "score": 0.79, "fate": "cut_by_the_edge"},
            {"keyframe": 1, "detection": 0, "label": "dining table", "score": 0.79, "fate": "object", "object_id": "obj_003", "sighting": 1, "joined_by_ray": True}]}),
            encoding="utf-8")
        (d / "scene.json").write_text(json.dumps({"objects": [
            {"object_id": "obj_001", "observation": {"association": {"status": "possible_duplicate", "may_duplicate": ["obj_002"]}}},
            {"object_id": "obj_003", "observation": {"association": {"status": "multi_position", "may_duplicate": []}}}]}), encoding="utf-8")
        page = SD.view(scan=d).read_text(encoding="utf-8")
        check("a box that went into an object is green, with its object's ID", '<rect class="obj"' in page and "dining table 0.79 \u2192 obj_003 (joined by its ray)" in page)
        check("a box left out is grey and dashed, with the reason", '<rect class="out"' in page and "dining table 0.79: cut by the picture&#x27;s edge" in page)
        check("a box whose object may be another is amber, naming the other", '<rect class="dup"' in page and "chair 0.87 \u2192 obj_001, maybe the same as obj_002" in page)
        check("the page counts every detection's fate", "3 detections: 2 in objects (1 joined by their ray), 1 left out: 0 not a category in this room&#x27;s map, 0 score below the minimum, 1 cut by the picture&#x27;s edge." in page)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
