"""Puts the clouds of several reconstruction runs into board 01's frame and writes them as one cloud.

Each run's anchor.json says which frame its cloud is in: ref01, ref02, or tracker (unscaled; cannot be placed). A run
anchored by board 01 that also saw board 02 gives board 02's pose in board 01's frame, and a run anchored by board 02
that also saw board 01 gives the reverse. Both are similarities, so they also correct board 02's assumed print size.
Clouds in board 02's frame are moved into board 01's with the link whose board fit is tightest.

    python tools/merge_clouds.py --runs RUN [RUN ...] --out merged_name

writes runs/<out>.ply and runs/<out>.json (the runs, their frames, points, and the link used).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])


def read_ply(path):
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii", "replace")
    if "binary_little_endian" not in head or "property uchar red" not in head:
        sys.exit(f"{path}: not a cloud written by qrecon_quest")
    v = np.frombuffer(raw[end:], dtype=VERTEX)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), np.stack([v["r"], v["g"], v["b"]], 1)


def write_ply(path, P, Cc):
    v = np.empty(len(P), dtype=VERTEX)
    v["x"], v["y"], v["z"] = P[:, 0], P[:, 1], P[:, 2]
    v["r"], v["g"], v["b"] = Cc[:, 0], Cc[:, 1], Cc[:, 2]
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
            f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    Path(path).write_bytes(head.encode("ascii") + v.tobytes())


def frame_of(anchor):
    if anchor.get("frame"):
        return anchor["frame"]
    return "ref01" if anchor.get("ref01", {}).get("ok") else "tracker"       # runs from before best-effort anchoring


def links(runs):
    """Candidate similarities taking board 02's frame to board 01's, as (corner fit, s, R, t, source run)."""
    out = []
    for run, anchor in runs:
        if frame_of(anchor) == "ref01" and "sim3_board_in_frame" in anchor.get("ref02", {}):
            L = anchor["ref02"]["sim3_board_in_frame"]
            out.append((anchor["ref02"].get("corner_fit_rms_m", 1.0), L["s"], np.array(L["R"]), np.array(L["t"]), run))
        if frame_of(anchor) == "ref02" and "sim3_board_in_frame" in anchor.get("ref01_link", {}):
            L = anchor["ref01_link"]["sim3_board_in_frame"]                  # board 01 in board 02's frame: invert it
            s, R, t = L["s"], np.array(L["R"]), np.array(L["t"])
            out.append((anchor["ref01_link"].get("corner_fit_rms_m", 1.0), 1.0 / s, R.T, -R.T @ t / s, run))
    return sorted(out, key=lambda x: x[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = []
    for run in a.runs:
        rec = HERE / "runs" / run / "recon"
        if not (rec / "cloud.ply").is_file() or not (rec / "anchor.json").is_file():
            sys.exit(f"{rec}: no cloud.ply and anchor.json; did that reconstruction finish?")
        runs.append((run, json.loads((rec / "anchor.json").read_text())))
    cand = links(runs)
    parts, report = [], {"runs": [], "link": None}
    if cand:
        fit, s, R, t, src = cand[0]
        report["link"] = {"from_run": src, "corner_fit_rms_m": fit, "scale": s, "R": R.tolist(), "t": t.tolist()}
    for run, anchor in runs:
        P, Cc = read_ply(HERE / "runs" / run / "recon" / "cloud.ply")
        fr = frame_of(anchor)
        entry = {"run": run, "frame": fr, "points": int(len(P)), "warnings": anchor.get("warnings", [])}
        if fr == "ref02":
            if not cand:
                entry["placed"] = False; entry["why"] = "in board 02's frame and no run links board 02 to board 01"
                report["runs"].append(entry); continue
            P = s * P @ R.T + t
        elif fr != "ref01":
            entry["placed"] = False; entry["why"] = "no board anchored it (tracker units)"
            report["runs"].append(entry); continue
        entry["placed"] = True
        parts.append((P, Cc)); report["runs"].append(entry)
    if not parts:
        sys.exit("no cloud could be placed in board 01's frame")
    P = np.concatenate([p for p, _ in parts]); Cc = np.concatenate([c for _, c in parts])
    out = HERE / "runs" / f"{a.out}.ply"
    write_ply(out, P, Cc)
    report["points"] = int(len(P)); report["extent_m"] = (P.max(0) - P.min(0)).round(2).tolist()
    (HERE / "runs" / f"{a.out}.json").write_text(json.dumps(report, indent=1))
    for e in report["runs"]:
        print(f"  {e['run']}: {e['points']} points in {e['frame']}'s frame" + ("" if e["placed"] else f" - left out: {e['why']}"))
    if report["link"]:
        print(f"  board 02 -> board 01 from {report['link']['from_run']} (board fit {1000 * report['link']['corner_fit_rms_m']:.1f} mm, "
              f"scale {report['link']['scale']:.3f})")
    print(f"{report['points']} points, extent {report['extent_m']} m -> {out}")


if __name__ == "__main__":
    main()
