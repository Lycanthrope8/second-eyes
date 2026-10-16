"""The perception folder's PC side (A4 Delivery 1, stage 3; grounding/quest/perception_device.py).

Builds a folder from the repository and checks it: every file is the repository's, byte for byte; the manifest lists
each with its size and SHA-256 and names the four roles; it reads back; a changed, missing or extra file is found; a
second build is byte-identical; and the build refuses to write over a folder or to start from a broken source.

    python grounding/tests/test_perception_folder.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from grounding.quest import perception_device as PD  # noqa: E402

FAILED, PASSED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def refuses(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    except Exception:  # noqa: BLE001
        return True
    return False


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("Building from the repository")
        check("the repository's files make a perception folder", PD.check_sources(REPO) == [], str(PD.check_sources(REPO)[:2]))
        r = PD.build(out=tmp / "a")
        folder = tmp / "a"
        m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        check("the manifest is a perception folder's", m["record_type"] == "a4_perception_folder" and m["format_version"] == 1)
        check("it lists every file, nine schemas and four role files", len(m["files"]) == 13
              and sum(k.startswith("schemas/") for k in m["files"]) == 9, str(len(m["files"])))
        same = all((folder / dest).read_bytes() == src.read_bytes() for dest, src in PD.files(REPO).items())
        check("every file is the repository's, byte for byte", same)
        check("each listed size and SHA-256 is the file's", all(f["bytes"] == (folder / n).stat().st_size
              and f["sha256"] == hashlib.sha256((folder / n).read_bytes()).hexdigest() for n, f in m["files"].items()))
        check("the four roles name listed files", all(m[k] in m["files"] for k in ("scan_config", "localization_config", "size_priors", "category_map")))
        check("the manifest's SHA-256 is reported", r["manifest_sha256"] == hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest())
        check("it reads back", PD.verify(folder) == [])
        PD.build(out=tmp / "b")
        check("a second build is byte-identical", all((tmp / "a" / n).read_bytes() == (tmp / "b" / n).read_bytes()
                                                     for n in list(m["files"]) + ["manifest.json"]))
        print("What verify finds")
        c = tmp / "c"; shutil.copytree(folder, c)
        (c / "configs" / "scan.default.v1.json").write_text("{}", encoding="utf-8")
        check("a changed file", any("differs" in x for x in PD.verify(c)))
        d = tmp / "d"; shutil.copytree(folder, d)
        (d / "maps" / "category-map.bedroom.v1.json").unlink()
        check("a missing file", any("missing" in x for x in PD.verify(d)))
        e = tmp / "e"; shutil.copytree(folder, e)
        (e / "stray.json").write_text("{}", encoding="utf-8")
        check("a file the manifest doesn't list", any("not in the manifest" for x in PD.verify(e) if "stray" in x))
        print("What build refuses")
        check("writing over an existing folder", refuses(lambda: PD.build(out=folder)))
        broken = tmp / "repo"
        shutil.copytree(REPO / "schemas", broken / "schemas")
        shutil.copytree(REPO / "perception", broken / "perception", ignore=shutil.ignore_patterns("detector"))
        pri = broken / "perception" / "bedroom" / "size-priors.bedroom.v1.json"
        rec = json.loads(pri.read_text(encoding="utf-8")); rec["entries"][0]["size_m"] = [0.1, 0.1]
        pri.write_text(json.dumps(rec), encoding="utf-8")
        check("a size prior that fails its schema", any("size-priors" in x for x in PD.check_sources(broken)))
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
