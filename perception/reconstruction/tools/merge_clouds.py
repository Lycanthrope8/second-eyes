#!/usr/bin/env python3
"""Puts the clouds of several reconstruction runs into one board's frame and writes them as one cloud.

Each run's anchor.json says which board's frame its cloud is in (or "tracker": unscaled, so it cannot be placed). A
run that also saw another board reports that board's pose in its frame, as a similarity (the scale also absorbs any
print-size difference between the boards). These links join the boards' frames. Every cloud is moved into the target
frame (board 01's unless --frame says otherwise) along the chain of best links: the tightest board fit, or for a link
taken from single views (camera standing still), the smallest spread of those views; a link from a run whose own anchor
failed its check is used only when nothing better joins. Where several runs link the same two boards, the report says
how far apart their answers are. Because the boards are fixed to
the walls, a link measured in one recording holds for every recording of the same place.

    python tools/merge_clouds.py --runs RUN [RUN ...] --out NAME [--frame ref01]
    python tools/merge_clouds.py --recordings runs/<name>_recording.json [...] --out NAME

A run without a cloud (made with --anchor-only) is used only for the links it reports.
Writes runs/<NAME>.ply and runs/<NAME>.json: the runs, their frames, points, the links used, and per room (the board
that anchors it) a Level 5 status in which every metric assessment stays pending until reference measurements and an
evaluation procedure exist (D137); a combined figure never stands in for a room.
"""
import argparse, heapq, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
MIN_VALID_SHARE = 0.8          # Level 5's keyframe criterion


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


def run_links(run, anchor):
    """Links reported by one run, as edges {frm: board, to: the run's frame, s, R, t, quality_m, how, run}: a point in
    board frm's frame maps to s R x + t in the run's frame."""
    F = frame_of(anchor)
    if F == "tracker":
        return []
    L = dict(anchor.get("links") or {})
    if not L:                                                                # anchor.json from R3.12 and R3.13
        if F == "ref01" and "ref02" in anchor:
            L["ref02"] = anchor["ref02"]
        if F == "ref02" and "ref01_link" in anchor:
            L["ref01"] = anchor["ref01_link"]
    out = []
    for b, r in L.items():
        if "sim3_board_in_frame" not in r or b == F:
            continue
        S = r["sim3_board_in_frame"]
        single = "position_spread_m" in r
        q = r["position_spread_m"] if single else r.get("corner_fit_rms_m")
        out.append({"frm": b, "to": F, "s": float(S["s"]), "R": np.array(S["R"], float), "t": np.array(S["t"], float),
                    "quality_m": float(q) if q is not None else 1.0, "how": "single views" if single else "board fit", "run": run,
                    "anchor_ok": bool(anchor.get(F, {}).get("ok"))})
    return out


FAILED_ANCHOR_PENALTY_M = 1.0   # a link from a run whose own anchor failed its check is used only when nothing better joins


def link_cost(e):
    return e["quality_m"] + (0.0 if e.get("anchor_ok", True) else FAILED_ANCHOR_PENALTY_M)


def link_checks(edges, max_m=0.10, max_deg=5.0):
    """Where several runs link the same two boards: how far apart their answers are (board B's pose in A's frame)."""
    groups = {}
    for e in edges:
        a, b = sorted((e["frm"], e["to"]))
        g = (e["s"], e["R"], e["t"])
        if e["to"] != a:                       # make every edge map b's frame into a's
            g = _invert(g)
        groups.setdefault((a, b), []).append((g, e))
    out = []
    for (a, b), items in groups.items():
        if len(items) < 2:
            continue
        dpos = max(np.linalg.norm(g1[2] - g2[2]) for (g1, _), (g2, _) in __import__("itertools").combinations(items, 2))
        drot = max(np.degrees(np.arccos(np.clip((np.trace(g1[1].T @ g2[1]) - 1) / 2, -1, 1)))
                   for (g1, _), (g2, _) in __import__("itertools").combinations(items, 2))
        out.append({"boards": f"{a}-{b}", "links": len(items), "runs": [e["run"] for _, e in items],
                    "max_position_difference_m": round(float(dpos), 3), "max_rotation_difference_deg": round(float(drot), 2),
                    "agree": bool(dpos <= max_m and drot <= max_deg)})
    return out


def _compose(g2, g1):        # apply g1, then g2
    s2, R2, t2 = g2
    s1, R1, t1 = g1
    return s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2


def _invert(g):
    s, R, t = g
    return 1.0 / s, R.T, -R.T @ t / s


def best_paths(edges, target):
    """For every frame that can reach the target: the similarity into the target frame and the edges used (Dijkstra on
    the sum of link quality; both directions of every link)."""
    adj = {}
    for e in edges:
        g = (e["s"], e["R"], e["t"])
        adj.setdefault(e["frm"], []).append((e["to"], g, e))                  # frm -> to
        adj.setdefault(e["to"], []).append((e["frm"], _invert(g), e))         # to -> frm
    I = (1.0, np.eye(3), np.zeros(3))
    best = {target: (0.0, I, [])}                                             # frame -> (cost, frame->target, edges)
    heap = [(0.0, target)]
    while heap:
        cost, f = heapq.heappop(heap)
        if cost > best[f][0]:
            continue
        for nb, g, e in adj.get(f, []):
            # g maps f -> nb, so nb -> target is (f -> target) after (nb -> f)
            c = cost + link_cost(e)
            if nb not in best or c < best[nb][0]:
                best[nb] = (c, _compose(best[f][1], _invert(g)), best[f][2] + [e])
                heapq.heappush(heap, (c, nb))
    return best


def _desc(e):
    q = f"spread {1000 * e['quality_m']:.1f} mm" if e["how"] == "single views" else f"board fit {1000 * e['quality_m']:.1f} mm"
    return (f"{e['frm']} -> {e['to']} from {e['run']} ({e['how']}, {q}, scale {e['s']:.3f}"
            + ("" if e.get("anchor_ok", True) else ", its run's own anchor failed its check") + ")")


def room_status(runs_info):
    """Per room (the board that anchors a run): Level 5 as far as it can be judged now (D137)."""
    rooms = {}
    for e in runs_info:
        F = e["frame"]
        if F == "tracker" or e.get("keyframes") is None:
            continue
        r = rooms.setdefault(F, {"runs": [], "keyframes": 0, "valid_keyframes": 0, "anchors_ok": 0, "anchors": 0})
        r["runs"].append(e["run"]); r["keyframes"] += e["keyframes"]; r["valid_keyframes"] += e["valid_keyframes"]
        r["anchors"] += 1; r["anchors_ok"] += int(bool(e.get("anchor_ok")))
    for F, r in rooms.items():
        share = r["valid_keyframes"] / r["keyframes"] if r["keyframes"] else 0.0
        r["valid_keyframe_share"] = round(share, 3)
        r["keyframe_criterion"] = "met" if share >= MIN_VALID_SHARE else "failed"
        r["anchor"] = "valid" if r["anchors_ok"] else "failed check"
        r["tape_distances"] = "pending: no reference measurements"
        r["walls_appearing_twice"] = "pending: evaluation procedure not defined"
        r["metric_assessment"] = "pending" + ("" if r["anchors_ok"] else ": no valid anchor")
        failing = [f"valid keyframe share {share:.2f} < {MIN_VALID_SHARE}"] if share < MIN_VALID_SHARE else []
        r["level5"] = ("not passed: " + "; ".join(failing) + "; " if failing else "") + "pending (reference measurements and procedure)"
    return rooms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--recordings", nargs="*", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--frame", default="ref01", help="target board frame")
    a = ap.parse_args()
    names = list(a.runs)
    for rec in a.recordings:
        names += [r for r in json.loads(Path(rec).read_text()).get("recon_runs", []) if r not in names]
    if not names:
        sys.exit("give --runs or --recordings")
    runs, link_only = [], []
    for run in names:
        rec = HERE / "runs" / run / "recon"
        if not (rec / "anchor.json").is_file():
            sys.exit(f"{rec}: no anchor.json; did that reconstruction finish?")
        anchor = json.loads((rec / "anchor.json").read_text())
        (runs if (rec / "cloud.ply").is_file() else link_only).append((run, anchor))
    edges = [e for run, anchor in runs + link_only for e in run_links(run, anchor)]
    target = a.frame
    paths = best_paths(edges, target)
    parts, report = [], {"target_frame": target, "runs": [], "links_used": {}, "link": None,
                         "link_only_runs": [run for run, _ in link_only]}
    for run, anchor in runs:
        F = frame_of(anchor)
        P, Cc = read_ply(HERE / "runs" / run / "recon" / "cloud.ply")
        st = HERE / "runs" / run / "recon" / "stats.json"
        stats = json.loads(st.read_text()) if st.is_file() else {}
        entry = {"run": run, "frame": F, "points": len(P), "warnings": anchor.get("warnings", []),
                 "anchor_ok": bool(anchor.get(F, {}).get("ok")) if F != "tracker" else False,
                 "keyframes": stats.get("keyframes"), "valid_keyframes": stats.get("valid_keyframes")}
        if F == "tracker":
            entry.update(placed=False, why="no board anchored it (tracker units)")
        elif F not in paths:
            entry.update(placed=False, why=f"in {F}'s frame and no run links {F} to {target}")
        else:
            s, R, t = paths[F][1]
            P = s * P @ R.T + t
            entry["placed"] = True
            if paths[F][2]:
                report["links_used"][F] = [_desc(e) for e in paths[F][2]]
                if report["link"] is None:
                    e = paths[F][2][0]
                    report["link"] = {"from_run": e["run"], "how": e["how"], "quality_m": e["quality_m"],
                                      "corner_fit_rms_m": e["quality_m"] if e["how"] == "board fit" else None,
                                      "scale": e["s"], "R": e["R"].tolist(), "t": e["t"].tolist()}
            parts.append((P, Cc))
        report["runs"].append(entry)
    report["rooms"] = room_status(report["runs"])
    report["link_checks"] = link_checks(edges)
    for e in report["runs"]:
        print(f"  {e['run']}: {e['points']} points in {e['frame']}'s frame" + ("" if e["placed"] else f" - left out: {e['why']}"))
    for F, used in report["links_used"].items():
        for d in used:
            print(f"  link: {d}")
    for c in report["link_checks"]:
        print(f"  {'' if c['agree'] else 'WARNING: '}{c['links']} links between {c['boards']} differ by up to "
              f"{c['max_position_difference_m']:.2f} m and {c['max_rotation_difference_deg']:.1f} deg ({', '.join(c['runs'])})")
    for F, r in report["rooms"].items():
        print(f"  room of {F}: valid keyframes {r['valid_keyframe_share']:.2f} ({r['keyframe_criterion']}), anchor {r['anchor']}; "
              f"Level 5 {r['level5']}")
    if not parts:
        sys.exit(f"no cloud could be placed in {target}'s frame")
    P = np.concatenate([p for p, _ in parts]); Cc = np.concatenate([c for _, c in parts])
    report["points"] = int(len(P))
    report["extent_m"] = [round(float(x), 2) for x in (P.max(0) - P.min(0))] if len(P) else [0, 0, 0]
    write_ply(HERE / "runs" / f"{a.out}.ply", P, Cc)
    (HERE / "runs" / f"{a.out}.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"{len(P)} points, extent {report['extent_m']} m -> {HERE / 'runs' / (a.out + '.ply')}")


if __name__ == "__main__":
    main()
