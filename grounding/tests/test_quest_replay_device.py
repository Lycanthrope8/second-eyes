"""A2.5 delivery 1, step 2 (D103): replay-push and replay-pull.

Run directly (python grounding/tests/test_quest_replay_device.py) or as a module. A real replay bundle is built on A2.3b's
fixture chain (labelled doubles, as in test_quest_replay_bundle); a labelled fake device stores pushed files, serves
results folders and records every adb command. Expectations are written by hand.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
APP = "/sdcard/Android/data/com.secondeyes.quest/files"
ALLOWED = {("devices",), ("push",), ("pull",), ("shell", "ls"), ("shell", "mkdir"), ("shell", "stat")}


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def refused(fn):
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
    try:
        fn()
    except EI as e:
        return " | ".join(i["message"] for i in e.issues)
    return None


class FakeDevice:
    """Labelled test double for adb: a dictionary of remote files, one or more devices, an app folder or none."""

    def __init__(self, files=None, devices=("SERIAL1",), app=True, truncate=False):
        self.files, self.devices, self.app, self.truncate, self.calls = dict(files or {}), list(devices), app, truncate, []

    def run(self, *args):
        self.calls.append(args)
        if args == ("devices",):
            return 0, "List of devices attached\n" + "".join(f"{d}\tdevice\n" for d in self.devices), ""
        if args[0] == "push":
            data = Path(args[1]).read_bytes()
            self.files[args[2]] = data[:-1] if self.truncate else data
            return 0, "1 file pushed\n", ""
        if args[0] == "pull":
            if args[1] not in self.files:
                return 1, "", f"adb: error: failed to stat remote object '{args[1]}'"
            Path(args[2]).write_bytes(self.files[args[1]])
            return 0, "1 file pulled\n", ""
        if args[:2] == ("shell", "ls"):
            path = args[-1]
            if path == APP:
                return (0, "logs\n", "") if self.app else (1, "", "No such file or directory")
            kids = sorted({k[len(path) + 1:].split("/")[0] for k in self.files if k.startswith(path + "/")})
            return (0, "".join(k + "\n" for k in kids), "") if kids else (1, "", "No such file or directory")
        if args[:2] == ("shell", "mkdir"):
            return 0, "", ""
        if args[:3] == ("shell", "stat", "-c"):
            return (0, f"{len(self.files[args[4]])}\n", "") if args[4] in self.files else (1, "", "No such file")
        return 1, "", "unsupported in the fake"

    def shell(self, *args):
        return self.run("shell", *args)

    def pull(self, remote, local):
        return self.run("pull", remote, str(local))


def build_bundle(tmp):
    TQ = importlib.import_module("grounding.tests.test_quest_replay_bundle")
    D = importlib.import_module("grounding.inference.iref_vla_compare.design")
    P = importlib.import_module("grounding.inference.iref_vla_compare.prepare")
    RUN = importlib.import_module("grounding.inference.iref_vla_compare.run")
    RB = importlib.import_module("grounding.quest.replay_bundle")
    A, _, _ = TQ.fixture_chain(tmp, D, P, RUN)
    req, run, run7 = tmp / "req", tmp / f"run-{TQ.KEYS[0]}", tmp / f"run-{TQ.KEYS[1]}"
    sc = TQ.scores_folder(tmp / "scores", req, run, run7)
    pol = TQ.replay_policy(RB, tmp, "pol", req, run, sc, TQ.jsonl(req / "request-index.jsonl"))
    RB.build_replay_bundle(requests=req, small_run=run, scores=tmp / "scores", tokenizer_dir=None, out=tmp / "replay",
                           policy=pol, tokenizer=A.OffsetCharTokenizer())
    return tmp / "replay"


def make_run(repo, run_id):
    (repo / "runs" / run_id).mkdir(parents=True)
    (repo / "runs" / run_id / "config.yaml").write_text("run_id: " + run_id + "\n", encoding="utf-8")


def results_folder(device, name, written, done=True, log="/sdcard/Android/data/com.secondeyes.quest/files/logs/s1.jsonl"):
    base = f"{APP}/replay/results/{name}"
    lines = "".join(json.dumps({"request_id": f"r{k:04d}"}) + "\n" for k in range(1, written + 1))
    device.files.update({f"{base}/identity.json": json.dumps({"app": {"event_log": log}}).encode(),
                         f"{base}/startup-log.txt": b"llama_kv_cache: K (f16): 48.00 MiB\n",
                         f"{base}/selfchecks.json": b"{}\n", f"{base}/fixtures.jsonl": b"{}\n",
                         f"{base}/results.jsonl": lines.encode(), log: b'{"ev":"replay.end"}\n'})
    if done:
        device.files[f"{base}/done.json"] = json.dumps({"requests": written, "written": written, "fixtures": 5,
                                                        "fixtures_as_expected": 5, "self_checks_passed": True,
                                                        "total_ms": 1.0}).encode()


def main() -> int:
    RD = importlib.import_module("grounding.quest.replay_device")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = build_bundle(tmp)
        repo = tmp / "repo"
        make_run(repo, "20261008_A2_r009")
        print("-- replay-push (labelled fake device)")
        dev = FakeDevice()
        r = RD.push(bundle=bundle, run_id="20261008_A2_r009", adb=dev, repo=repo, progress=lambda m: None)
        pushed = sorted(k for k in dev.files)
        check("exactly the three headset files reach the app's replay/bundle folder, byte for byte",
              pushed == sorted(f"{APP}/replay/bundle/{n}" for n in ("fixtures-written.jsonl", "replay-manifest.json", "requests.jsonl"))
              and all(dev.files[f"{APP}/replay/bundle/{n}"] == (bundle / "headset" / n).read_bytes() for n in RD.HEADSET_FILES))
        check("no reference file leaves the laptop", not any("reference" in k for k in dev.files))
        rec = json.loads(Path(r["receipt"]).read_text(encoding="utf-8"))
        check("the receipt sits in the run's raw/replay with every file's SHA-256 and the bundle manifest's",
              Path(r["receipt"]).parent == repo / "runs" / "20261008_A2_r009" / "raw" / "replay"
              and [f["sha256"] for f in rec["files"]] == [hashlib.sha256((bundle / "headset" / n).read_bytes()).hexdigest()
                                                         for n in RD.HEADSET_FILES]
              and rec["bundle_manifest_sha256"] == hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest())
        check("push uses only devices, ls, mkdir, stat and push: no install, launch or rerun",
              all(c[:1] in ALLOWED or c[:2] in ALLOWED for c in dev.calls), str(dev.calls[:3]))
        cases = [("an unknown run", dict(run_id="20261008_A2_r099"), FakeDevice(), "does not exist"),
                 ("a malformed run ID", dict(run_id="2026MMDD_A2_r009"), FakeDevice(), "is not a run ID"),
                 ("two devices", dict(), FakeDevice(devices=("A", "B")), "2 devices ready"),
                 ("no app folder yet", dict(), FakeDevice(app=False), "start the app once"),
                 ("a file that arrives shorter", dict(), FakeDevice(truncate=True), "bytes on the headset")]
        for label, kw, d, why in cases:
            msg = refused(lambda: RD.push(**{"bundle": bundle, "run_id": "20261008_A2_r009", "adb": d, "repo": repo,
                                             "progress": lambda m: None, **kw}))
            check(f"push refused: {label}", msg is not None and why in msg, str(msg))
        broken = tmp / "broken"
        import shutil
        shutil.copytree(bundle, broken)
        with open(broken / "headset" / "requests.jsonl", "ab") as fh:
            fh.write(b" ")
        d = FakeDevice()
        msg = refused(lambda: RD.push(bundle=broken, run_id="20261008_A2_r009", adb=d, repo=repo, progress=lambda m: None))
        check("push refused: a bundle that does not read back, before anything is pushed",
              msg is not None and "does not read back" in msg and not d.files)

        print("-- replay-pull (labelled fake device)")
        dev = FakeDevice()
        results_folder(dev, "20261008-100000", written=3)
        results_folder(dev, "20261008-110000", written=1, done=False)
        r = RD.pull(run_id="20261008_A2_r009", adb=dev, repo=repo, progress=lambda m: None)
        dest = repo / "runs" / "20261008_A2_r009" / "raw" / "replay" / "20261008-100000"
        check("the newest finished folder is chosen; the newer one without done.json is not", r["results"] == "20261008-100000")
        got = sorted(p.name for p in dest.iterdir())
        check("its six files, the session's event log and the receipt arrive", got == sorted(list(RD.RESULT_FILES) +
              ["events.jsonl", "pull.json"]), str(got))
        check("the receipt says complete: nothing missing, three result lines matching done.json",
              r["complete"] and r["missing"] == [] and r["checks"]["results_lines"] == 3 and r["checks"]["lines_match_done"])
        check("every pulled file is hashed in the receipt",
              all(r["files"][n]["sha256"] == hashlib.sha256((dest / n).read_bytes()).hexdigest() for n in r["files"]))
        check("pull uses only devices, ls, stat and pull", all(c[:1] in ALLOWED or c[:2] in ALLOWED for c in dev.calls))
        msg = refused(lambda: RD.pull(run_id="20261008_A2_r009", adb=dev, repo=repo, progress=lambda m: None))
        check("pulling the same folder again is refused (nothing is overwritten)", msg is not None and "already pulled" in msg)
        r = RD.pull(run_id="20261008_A2_r009", results="20261008-110000", adb=dev, repo=repo, progress=lambda m: None)
        check("an unfinished folder can be pulled by name, and is marked incomplete with done.json missing",
              not r["complete"] and "done.json" in r["missing"])
        msg = refused(lambda: RD.pull(run_id="20261008_A2_r009", results="20991231-000000", adb=dev, repo=repo,
                                      progress=lambda m: None))
        check("an unknown results folder is refused", msg is not None and "no results folder" in msg)
        d = FakeDevice()
        results_folder(d, "20261009-090000", written=1, done=False)
        msg = refused(lambda: RD.pull(run_id="20261008_A2_r009", adb=d, repo=repo, progress=lambda m: None))
        check("with no finished folder, a default pull is refused", msg is not None and "no finished results folder" in msg)
        check("no staging folder is left behind", not any(p.name.startswith(".") for p in dest.parent.iterdir()))

        print("-- the command line")
        CLI = importlib.import_module("grounding.quest.__main__")
        RI = importlib.import_module("grounding.quest.runtime_identity")
        real_adb, real_repo = RI.Adb, RD.REPO
        try:
            dev = FakeDevice()
            RI.Adb = lambda binary="adb": dev
            RD.REPO = repo
            check("replay-push exits 0, its receipt never overwriting the earlier one from the same second",
                  CLI.main(["replay-push", "--bundle", str(bundle), "--run", "20261008_A2_r009"]) == 0
                  and len(list((repo / "runs" / "20261008_A2_r009" / "raw" / "replay").glob("push-*.json"))) == 2)
            check("replay-push refuses an unknown run with exit 2",
                  CLI.main(["replay-push", "--bundle", str(bundle), "--run", "20261008_A2_r098"]) == 2)
            results_folder(dev, "20261008-120000", written=2)
            check("replay-pull exits 0", CLI.main(["replay-pull", "--run", "20261008_A2_r009"]) == 0)
        finally:
            RI.Adb, RD.REPO = real_adb, real_repo
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
