"""A2.5 delivery 4: the laptop's side of the ADB inbox (labelled fake device, fixture goldens).

Run directly (python grounding/tests/test_quest_inbox.py) or as a module. Expectations are written by hand.
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


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def refused(fn, EI):
    try:
        fn()
    except EI as e:
        return "; ".join(i["message"] for i in e.issues)
    return None


def main() -> int:
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    PG = importlib.import_module("grounding.quest.prompt_goldens")
    TD = importlib.import_module("grounding.tests.test_quest_replay_device")
    IB = importlib.import_module("grounding.quest.inbox")
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError

    class Device(TD.FakeDevice):
        """Labelled double: the replay helpers' fake device, plus an atomic rename (shell mv)."""
        def run(self, *args):
            if args[:2] == ("shell", "mv"):
                self.calls.append(args)
                if args[2] not in self.files:
                    return 1, "", "No such file"
                self.files[args[3]] = self.files.pop(args[2])
                return 0, "", ""
            return super().run(*args)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        A = SB.pilot_helpers()
        bundle = SB.scoring_bundle(tmp, A)
        PG.build_goldens(bundle=bundle, out=tmp / "g", tokenizer=A.OffsetCharTokenizer(), sizes=(4, 5, 6), index_sha256_prefix=None)
        head = tmp / "g" / "headset"
        rows = [json.loads(x) for x in (head / "goldens.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        repo = tmp / "repo"
        TD.make_run(repo, "20261010_A2_r017")
        print("-- inbox-send (labelled fake device)")
        dev = Device(files={f"{IB.INBOX}/processed/old.json": b"{}"})
        r4 = next(r for r in rows if r["kind"] == "dataset_command" and len(json.loads((head / r["scene_file"]).read_text())["objects"]) == 4)
        rc = IB.inbox_send(run_id="20261010_A2_r017", goldens=tmp / "g", snapshot="4", command_file=r4["command_file"], request_id="q-001",
                           adb=dev, repo=repo, progress=lambda m: None)
        q = rc["request"]
        scene = (head / r4["scene_file"]).read_bytes()
        check("the request binds the golden snapshot by ID, scene, revision and file SHA-256",
              q["expected_scene"] == {"snapshot_id": r4["snapshot_id"], "scene_id": json.loads(scene)["scene_id"],
                                      "scene_revision": json.loads(scene)["scene_revision"], "snapshot_sha256": hashlib.sha256(scene).hexdigest()})
        check("a golden dataset command is sent as it is, labelled dataset_command",
              q["command"] == json.loads((head / r4["command_file"]).read_text(encoding="utf-8")) and q["command_kind"] == "dataset_command")
        pushes = [c for c in dev.calls if c[0] == "push"]
        moves = [c for c in dev.calls if c[:2] == ("shell", "mv")]
        check("it is pushed under a temporary name and renamed when complete, leaving no temporary file",
              len(pushes) == 1 and pushes[0][2] == f"{IB.INBOX}/q-001.json.tmp" and moves == [("shell", "mv", f"{IB.INBOX}/q-001.json.tmp", f"{IB.INBOX}/q-001.json")]
              and f"{IB.INBOX}/q-001.json" in dev.files and f"{IB.INBOX}/q-001.json.tmp" not in dev.files)
        check("the delivered bytes are the receipt's, and the receipt keeps the PC's push time apart",
              hashlib.sha256(dev.files[f"{IB.INBOX}/q-001.json"]).hexdigest() == rc["request_sha256"]
              and "never combined" in rc["pc_clock_note"] and Path(rc["receipt"]).parent == repo / "runs" / "20261010_A2_r017" / "raw" / "inbox")
        rw = IB.inbox_send(run_id="20261010_A2_r017", goldens=tmp / "g", snapshot="4", text="the one by the door", request_id="q-002",
                           adb=dev, repo=repo, progress=lambda m: None)["request"]
        check("a written text is labelled written, on the snapshot's dataset command with a suffixed command ID",
              rw["command_kind"] == "written" and rw["command"]["text"] == "the one by the door" and "-w" in rw["command"]["command_id"]
              and rw["command"]["scene_id"] == q["expected_scene"]["scene_id"])
        for label, kw, want in (("a request ID used before in the run", dict(snapshot="4", command_file=r4["command_file"], request_id="q-001"), "already sent"),
                                ("a request ID with a path", dict(snapshot="4", command_file=r4["command_file"], request_id="../x"), "must match"),
                                ("an object count with no snapshot", dict(snapshot="9", command_file=r4["command_file"], request_id="q-003"), "names 0"),
                                ("a command of another snapshot", dict(snapshot="5", command_file=r4["command_file"], request_id="q-004"), "not a golden command")):
            msg = refused(lambda: IB.inbox_send(run_id="20261010_A2_r017", goldens=tmp / "g", adb=dev, repo=repo, progress=lambda m: None, **kw), EI) or ""
            check(f"refused: {label}", want in msg, msg[:100])
        msg = refused(lambda: IB.inbox_send(run_id="20261010_A2_r017", goldens=tmp / "g", snapshot="4", command_file=r4["command_file"], request_id="q-005",
                                            adb=Device(), repo=repo, progress=lambda m: None), EI) or ""
        check("refused: no inbox on the headset (no session started); this helper never launches the app", "start a session" in msg, msg[:100])
        rd = IB.inbox_send(run_id="20261010_A2_r017", goldens=tmp / "g", snapshot="4", dataset=1, request_id="q-006", adb=dev, repo=repo,
                           progress=lambda m: None)["request"]
        first4 = next(r for r in rows if r["snapshot_id"] == r4["snapshot_id"] and r["kind"] == "dataset_command")
        check("--dataset 1 sends the snapshot's first golden dataset command",
              rd["command"] == json.loads((head / first4["command_file"]).read_text(encoding="utf-8")) and rd["command_kind"] == "dataset_command")
        rr = IB.inbox_send(run_id="20261010_A2_r017", goldens=tmp / "g", snapshot="4", dataset=1, request_id="q-006", resend=True, adb=dev,
                           repo=repo, progress=lambda m: None)
        check("--resend delivers the same request ID again on purpose, with its own receipt",
              rr["resend"] and Path(rr["receipt"]).name == "send-q-006-resend2.json" and rr["request"]["request_id"] == "q-006")
        print("-- outbox-pull (labelled fake device)")
        out = lambda n, d: (f"{IB.OUTBOX}/{n}", json.dumps(d).encode())  # noqa: E731
        dev.files.update(dict([
            out("q-001.ack.json", {"request_id": "q-001", "status": "accepted", "reason": None}),
            out("q-001.result.json", {"request_id": "q-001", "outcome": {"status": "completed", "reason": "max_offered_logit", "target_object_id": "obj_2",
                                                                          "execution_path": "U"}, "times": {"app_observed_ms": 17250.5}}),
            out("q-002.ack.json", {"request_id": "q-002", "status": "rejected", "reason": "stale_scene_binding"}),
            out("q-002.result.json", {"request_id": "q-002", "outcome": {"status": "refused", "reason": "stale_scene_binding", "execution_path": "none"},
                                      "times": {"app_observed_ms": 3.0}}),
            out("q-001.duplicate-101500123.json", {"request_id": "q-001"}),
            (f"{IB.SESSIONS}/20261010-101000/session.json", b'{"session":"20261010-101000"}'),
            (f"{IB.SESSIONS}/20261010-101000/outcomes.jsonl", b'{"request_id":"q-001"}\n')]))
        n0 = len(dev.calls)
        rp = IB.outbox_pull(run_id="20261010_A2_r017", adb=dev, repo=repo, progress=lambda m: None)
        s = rp["requests"]
        check("acknowledgements, results and duplicate notices arrive, with the newest session's log",
              Path(rp["folder"]).parent == repo / "runs" / "20261010_A2_r017" / "raw" / "outbox" and rp["session"] == "20261010-101000"
              and (Path(rp["folder"]) / "session" / "outcomes.jsonl").is_file() and len([k for k in rp["files"] if k.startswith("outbox/")]) == 5)
        check("the summary keeps each request's acknowledgement, outcome, path and app-observed time, and counts duplicates",
              s["q-001"]["ack"] == "accepted" and s["q-001"]["status"] == "completed" and s["q-001"]["target"] == "obj_2"
              and s["q-001"]["app_observed_ms"] == 17250.5 and s["q-001"]["duplicates"] == 1
              and s["q-002"]["ack"] == "rejected (stale_scene_binding)" and s["q-002"]["status"] == "refused")
        check("every pulled file is hashed in the receipt",
              all(hashlib.sha256((Path(rp["folder"]) / k).read_bytes()).hexdigest() == v for k, v in rp["files"].items()))
        check("pull is read-only on the headset: only devices, ls and pull",
              len(dev.calls) > n0 and all(c[0] in ("devices", "pull") or c[:2] == ("shell", "ls") for c in dev.calls[n0:]))
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
