#!/usr/bin/env python3
"""Shows the maps in a merged cloud, all together or one at a time, and says which map is which.

    python tools/view_maps.py house_all4                     all maps, one colour per map, in place
    python tools/view_maps.py house_all4 --one-by-one        each map alone in its own image colours;
                                                             close the window to see the next
    python tools/view_maps.py house_all4 --one-by-one --context   the same, with the rest of the house in grey
    python tools/view_maps.py house_all4 --map qrecon_20261118_r009

Reads runs/NAME.json, runs/NAME.ply and runs/NAME_by_run.ply (written by merge_clouds.py --color-by-run) and the
recording summaries in runs/ (which recording and map each reconstruction came from). Needs numpy and open3d.
Viewer: drag to rotate, scroll to zoom, Ctrl+drag to pan, + and - change the point size, Q closes the window.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
NAMES = {(230, 25, 75): "red", (60, 180, 75): "green", (255, 225, 25): "yellow", (0, 130, 200): "blue",
         (245, 130, 48): "orange", (145, 30, 180): "purple", (70, 240, 240): "cyan", (240, 50, 230): "magenta",
         (210, 245, 60): "lime", (250, 190, 190): "pink", (0, 128, 128): "teal", (170, 110, 40): "brown",
         (128, 0, 0): "maroon", (0, 0, 128): "navy", (128, 128, 0): "olive", (255, 255, 255): "white"}


def read_ply(path):
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    v = np.frombuffer(raw[end:], dtype=VERTEX)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), np.stack([v["r"], v["g"], v["b"]], 1)


def map_info(runs_dir):
    """{run id: description} for every reconstructed map, and a list of the maps not in any cloud."""
    placed, missing = {}, []
    for p in sorted(glob.glob(str(Path(runs_dir) / "*_recording.json"))):
        S = json.loads(Path(p).read_text())
        for e in S.get("maps", []):
            span = f"{e.get('first_t', 0):.0f}-{e.get('last_t', 0):.0f} s"
            run = e.get("recon_run")
            if not run:
                missing.append(f"{S['name']} map {e['map']} ({span}): {e.get('skipped', 'not reconstructed')}")
                continue
            pb = e.get("placed_by")
            if pb:
                how = f"placed by shared views with {', '.join(pb.get('with_maps', []))}"
                if pb.get("kept") == "trimmed":
                    span = f"{pb['span_s'][0]:.0f}-{pb['span_s'][1]:.0f} s kept of {span}"
            else:
                how = f"anchored by board {e.get('frame', '?').replace('ref', '')}"
            share = e.get("valid_keyframe_share")
            placed[run] = {"title": f"{S['name']} map {e['map']}", "span": span, "how": how, "frame": e.get("frame"),
                           "usable": f"{100 * share:.0f}% of keyframes usable" if share is not None else ""}
    return placed, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="the merge's name, for instance house_all4")
    ap.add_argument("--one-by-one", action="store_true")
    ap.add_argument("--context", action="store_true", help="with --one-by-one or --map: show the rest of the house in grey")
    ap.add_argument("--map", nargs="+", metavar="RUN", help="show only these reconstruction runs")
    a = ap.parse_args()
    runs = HERE / "runs"
    J = json.loads((runs / f"{a.name}.json").read_text())
    if "run_colors" not in J:
        sys.exit(f"runs/{a.name}.json has no run colours: run merge_clouds.py again with --color-by-run")
    P, C = read_ply(runs / f"{a.name}.ply")
    _, R = read_ply(runs / f"{a.name}_by_run.ply")
    info, missing = map_info(runs)
    rows = []
    print(f"{a.name}: {len(P)} points, {len(J['run_colors'])} maps")
    for run, col in J["run_colors"].items():
        mask = np.all(R == np.array(col, np.uint8), axis=1)
        d = info.get(run, {"title": run, "span": "", "how": "", "usable": ""})
        rows.append((run, col, mask, d))
        print(f"  {NAMES.get(tuple(col), str(tuple(col))):8s} {d['title']:16s} {d['span']:28s} {d['how']}; {d['usable']}  [{run}, {mask.sum()} points]")
    if missing:
        print("not in the cloud (no board seen and not placed, or thrown away by the tracker):")
        for m in missing:
            print(f"  {m}")
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        sys.exit("open3d is not installed in this Python: run this with the viewer environment, "
                 "for instance  & \"$HOME\\viewer-venv\\Scripts\\python.exe\" tools\\view_maps.py " + a.name)

    def cloud(points, colours):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colours.astype(np.float64) / 255.0)
        return pc
    if not a.one_by_one and not a.map:
        print("showing all maps, one colour per map; close the window to finish")
        o3d.visualization.draw_geometries([cloud(P, R)], window_name=f"{a.name}: one colour per map", width=1280, height=800)
        return
    chosen = [r for r in rows if not a.map or r[0] in a.map]
    for k, (run, col, mask, d) in enumerate(chosen, 1):
        title = f"{d['title']} ({d['span']}), {d['how']} [{run}]"
        print(f"map {k} of {len(chosen)}: {title}; close the window for the next")
        if a.context:
            colours = np.where(mask[:, None], C, 190).astype(np.uint8)
            geo = cloud(P, colours)
        else:
            geo = cloud(P[mask], C[mask])
        o3d.visualization.draw_geometries([geo], window_name=title, width=1280, height=800)


if __name__ == "__main__":
    main()
