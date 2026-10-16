"""A4 Delivery 1, stage 3: a scan's files from the headset, and a page to look at its keyframes, over adb.

`pull` copies one scan's folder, files/scans/<capture ID>/ (the newest when none is named), into a new local folder:
facts.json, capture.json, scene.json and trace.json, as many as the scan wrote, and each saved keyframe image its facts
name (files/snapshots/<name>.png). It never writes over an existing folder; pull.json records what came.
`view` writes view.html beside a pulled scan: each kept keyframe's image, the model's own input, with that keyframe's
detections drawn over it as the model saw them (boxes in source pixels times the letterbox ratio), so a person can see
whether the boxes sit on the objects and whether the image is upright. Without saved images it says so.

    python -m grounding.quest.scan_device pull --out <folder> [--capture <capture ID>]
    python -m grounding.quest.scan_device view --scan <pulled scan folder>
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "grounding.quest"

from ..evaluation.iref_vla.protocol import encode_json  # noqa: E402
from .replay_device import APP_DIR, _fail, _one_device, _stamp  # noqa: E402
from .runtime_identity import Adb  # noqa: E402

REMOTE_SCANS = APP_DIR + "/scans"
REMOTE_SNAPSHOTS = APP_DIR + "/snapshots"
FILES = ("facts.json", "capture.json", "scene.json", "trace.json")


def _order(name: str):
    """scan.<session>.<n>: newest session last, then the highest n."""
    parts = name.split(".")
    n = int(parts[-1]) if parts[-1].isdigit() else -1
    return (".".join(parts[1:-1]), n)


def newest(adb) -> str | None:
    code, out, _ = adb.shell("ls", REMOTE_SCANS)
    if code != 0:
        return None
    names = [x.strip() for x in out.splitlines() if x.strip().startswith("scan.")]
    return max(names, key=_order) if names else None


def pull(*, out, capture=None, adb=None, progress=print) -> dict:
    adb = adb if adb is not None else Adb()
    serial = _one_device(adb)
    cid = capture or newest(adb)
    if not cid:
        _fail(f"no scan on the headset ({REMOTE_SCANS} is empty or missing)", "scan")
    dest = Path(out) / cid
    if dest.exists():
        _fail(f"{dest} already exists; scans are pulled into a new folder", "scan")
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        _fail(f"{tmp} is left from an earlier pull; remove it first", "scan")
    tmp.mkdir(parents=True)
    got = []
    for name in FILES:
        if adb.run("pull", f"{REMOTE_SCANS}/{cid}/{name}", str(tmp / name))[0] == 0:
            got.append(name)
            progress(f"pulled {name}")
    if "facts.json" not in got:
        _fail(f"{cid} has no facts.json on the headset: not a finished scan", "scan")
    facts = json.loads((tmp / "facts.json").read_text(encoding="utf-8"))
    snaps = [k["snapshot"] for k in facts.get("keyframes", []) if k.get("snapshot")]
    missing = []
    if snaps:
        (tmp / "snapshots").mkdir()
    for s in snaps:
        if adb.run("pull", f"{REMOTE_SNAPSHOTS}/{s}.png", str(tmp / "snapshots" / f"{s}.png"))[0] != 0:
            missing.append(s)
    receipt = {"format_version": 1, "record_type": "a4_scan_pull", "pulled_utc": _stamp(), "device": serial, "capture_id": cid,
               "files": got, "snapshots": len(snaps) - len(missing), "missing_snapshots": missing}
    (tmp / "pull.json").write_bytes(encode_json(receipt))
    tmp.rename(dest)
    receipt["folder"] = str(dest)
    return receipt


def view(*, scan) -> Path:
    scan = Path(scan)
    facts = json.loads((scan / "facts.json").read_text(encoding="utf-8"))
    capture = json.loads((scan / "capture.json").read_text(encoding="utf-8")) if (scan / "capture.json").is_file() else None
    parts = [f"<h1>Scan {html.escape(facts.get('capture_id', ''))}</h1>",
             f"<p>Timestamps: {html.escape(str(facts.get('stamp_kind')))} ({html.escape(str(facts.get('sdk_unit')))}); "
             f"{facts.get('kept', 0)} keyframes kept, {facts.get('failed', 0)} left out.</p>"]
    if capture is None:
        parts.append("<p>No capture.json: the scan stopped before it made one.</p>")
    else:
        cam, size = capture["camera"], capture["detector"]["input_px"]
        ratio = min(size / cam["height_px"], size / cam["width_px"])
        frames = {k["index"]: k for k in capture["keyframes"]}
        kept = [k for k in facts.get("keyframes", []) if k.get("kept")]
        if not any(k.get("snapshot") for k in kept):
            parts.append("<p>No keyframe images were saved: files/scans/save-images was absent on the headset.</p>")
        for k in kept:
            f = frames.get(k["index"])
            if f is None or not k.get("snapshot"):
                continue
            boxes = []
            for d in f["detections"]:
                x1, y1, x2, y2 = (v * ratio for v in d["box_px"])
                label = html.escape(f"{d['label']} {d['score']:.2f}")
                boxes.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{max(x2 - x1, 0):.1f}" height="{max(y2 - y1, 0):.1f}"/>'
                             f'<text x="{x1 + 2:.1f}" y="{max(y1 - 3, 10):.1f}">{label}</text>')
            parts.append(f'<figure><div class="frame"><img src="snapshots/{html.escape(k["snapshot"])}.png" width="{size}" height="{size}">'
                         f'<svg width="{size}" height="{size}">{"".join(boxes)}</svg></div>'
                         f'<figcaption>{html.escape(f["frame_id"])}: spot {f["position_index"] + 1}, step {f["step_index"]}; '
                         f'{len(f["detections"])} detections; frame age {k.get("frame_age_us", 0) / 1000:.0f} ms</figcaption></figure>')
    page = ("<!doctype html><meta charset=\"utf-8\"><title>Scan view</title><style>"
            "body{font-family:system-ui,sans-serif;margin:24px;background:#f6f6f6;color:#222}"
            "figure{display:inline-block;margin:0 16px 16px 0}.frame{position:relative}"
            "svg{position:absolute;left:0;top:0}rect{fill:none;stroke:#e4002b;stroke-width:2}"
            "text{fill:#e4002b;font-size:12px;font-weight:600}</style>" + "".join(parts))
    out = scan / "view.html"
    out.write_text(page, encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.quest.scan_device", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pull", help="copy a scan from the headset")
    p.add_argument("--out", required=True)
    p.add_argument("--capture")
    v = sub.add_parser("view", help="write view.html for a pulled scan")
    v.add_argument("--scan", required=True)
    a = ap.parse_args(argv)
    if a.command == "pull":
        r = pull(out=a.out, capture=a.capture)
        print(f"pulled {r['capture_id']} into {r['folder']}: {', '.join(r['files'])}; {r['snapshots']} keyframe images"
              + (f", {len(r['missing_snapshots'])} missing" if r["missing_snapshots"] else ""))
    else:
        print(f"wrote {view(scan=a.scan)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
