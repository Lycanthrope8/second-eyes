#!/usr/bin/env python3
"""llama.cpp on the headset, from the command line (A1.8b, D55).

    python grounding/llama_headset.py prepare RUN_ID grounding/models/qwen2.5-0.5b-instruct.json
    python grounding/llama_headset.py push RUN_ID
    python grounding/llama_headset.py run RUN_ID
    python grounding/llama_headset.py bench RUN_ID
    python grounding/llama_headset.py check RUN_ID --reference REFERENCE_RUN [MORE_REFERENCE_RUNS]

prepare writes runs/RUN_ID/raw/job.json: the fixed prompt and the presets, formatted with the model description's
template, where each one's scene ends (grounding/scene.py), and the objects to score. push copies the test program,
libse_llama.so and llama-bench (tools/build_llama.py android) and the GGUF model (grounding/export_gguf.py) to
/data/local/tmp/se on the headset; the model only if the headset's copy differs in size. run runs the job there and
pulls cli_results.json; bench runs llama.cpp's llama-bench for standard speeds (llama_bench.json). check compares the
results with the PC references in the reference runs (the first that has a file wins): token IDs, the greedy answer, cached against uncached, and each
candidate's log-probability against reference_<prompt>_candidates.json (grounding/reference.py --candidates); then it
summarizes the timings. With --local BIN_DIR, prepare and run use the PC and a local build instead of the headset.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import scene

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DEVICE_DIR = "/data/local/tmp/se"
BINARIES = ("se_llama_cli", "libse_llama.so", "llama-bench")
DEFAULT_PROMPTS = ("a17-fixed", "a17-front", "a17-left", "a17-table")
# A candidate's log-probability may differ from the 32-bit PC reference by this much (8-bit weights round), and the
# candidates must keep their order.
SCORE_TOLERANCE = 0.15
CACHE_TOLERANCE = 0.05   # cached scene against uncached, on the same device


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def raw_dir(run_id: str) -> Path:
    run = RUNS / run_id
    if not (run / "config.yaml").is_file():
        raise ToolError(f"No run {run_id} in runs/. Create it first with: python tools/runs.py new ...")
    (run / "raw").mkdir(exist_ok=True)
    return run / "raw"


def adb(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    if shutil.which("adb") is None:
        raise ToolError("adb was not found on your PATH.")
    r = subprocess.run(["adb", *args], text=True, capture_output=capture, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise ToolError(f"adb {' '.join(args)} failed: {(r.stderr or r.stdout or '').strip()}")
    return r


def gguf_of(desc_path: Path, desc: dict, quant: str) -> Path:
    return desc_path.parent / desc["name"] / "gguf" / f"{desc['name']}-{quant}.gguf"


def cmd_prepare(args) -> int:
    raw = raw_dir(args.run_id)
    desc_path = Path(args.description)
    desc = json.loads(desc_path.read_text(encoding="utf-8"))
    gguf = gguf_of(desc_path, desc, args.quant)
    if not gguf.is_file():
        raise ToolError(f"{gguf.as_posix()} isn't there. Convert first: python grounding/export_gguf.py "
                        f"{desc_path.as_posix()}")
    prompts = []
    for pid in args.prompts:
        p = json.loads((ROOT / "grounding" / "prompts" / f"{pid}.json").read_text(encoding="utf-8"))
        text = scene.formatted(desc["chat_template"], p["system"], p["user"])
        prompts.append({"id": p["id"], "formatted": text, "prefix_chars": scene.scene_chars(text),
                        "answer_prefix": scene.ANSWER_PREFIX, "candidates": scene.candidates(p["user"]),
                        "candidate_suffix": scene.CANDIDATE_SUFFIX, "expected_target": p.get("expected_target")})
    model = str(gguf.resolve()) if args.local else f"{DEVICE_DIR}/{gguf.name}"
    job = {"model": model, "n_ctx": 1024, "threads": args.threads, "repeats": args.repeats,
           "max_new_tokens": desc["max_new_tokens"], "prompts": prompts,
           "gguf": {"file": gguf.name, "bytes": gguf.stat().st_size}}
    (raw / "job.json").write_text(json.dumps(job, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote runs/{args.run_id}/raw/job.json: {len(prompts)} prompts, threads {args.threads}, "
          f"{args.repeats} repeats, model {gguf.name} ({gguf.stat().st_size / 1e6:.0f} MB)"
          + (" on this PC" if args.local else " on the headset"))
    return 0


def cmd_push(args) -> int:
    raw = raw_dir(args.run_id)
    job = json.loads((raw / "job.json").read_text(encoding="utf-8"))
    bins = ROOT / "native" / "out" / "android-arm64"
    missing = [b for b in BINARIES if not (bins / b).is_file()]
    if missing:
        raise ToolError(f"{', '.join(missing)} missing in native/out/android-arm64/. Build first: "
                        "python tools/build_llama.py android")
    adb("shell", "mkdir", "-p", DEVICE_DIR)
    for b in BINARIES:
        adb("push", str(bins / b), f"{DEVICE_DIR}/{b}")
    adb("shell", "chmod", "755", f"{DEVICE_DIR}/se_llama_cli", f"{DEVICE_DIR}/llama-bench")
    if (bins / "build.json").is_file():   # which release, NDK and flags this run's programs came from
        shutil.copy2(bins / "build.json", raw / "build.json")
    gguf = next((p for p in (ROOT / "grounding" / "models").glob(f"*/gguf/{job['gguf']['file']}")), None)
    if gguf is None:
        raise ToolError(f"{job['gguf']['file']} not found under grounding/models/*/gguf/.")
    remote = f"{DEVICE_DIR}/{gguf.name}"
    size = adb("shell", "stat", "-c", "%s", remote, check=False).stdout.strip()
    if size == str(gguf.stat().st_size):
        print(f"{gguf.name} is already on the headset.")
    else:
        print(f"Pushing {gguf.name} ({gguf.stat().st_size / 1e6:.0f} MB) ...")
        adb("push", str(gguf), remote, capture=False)
    adb("push", str(raw / "job.json"), f"{DEVICE_DIR}/job.json")
    print(f"Pushed to {DEVICE_DIR}. Next: python grounding/llama_headset.py run {args.run_id}")
    return 0


def run_streamed(command: list, log: Path) -> int:
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print("  " + line.rstrip())
            out.write(line)
        return proc.wait()


def cmd_run(args) -> int:
    raw = raw_dir(args.run_id)
    if args.local:
        cli = Path(args.local) / "se_llama_cli"
        code = run_streamed([str(cli), str(raw / "job.json"), str(raw / "cli_results.json")], raw / "cli_output.txt")
    else:
        code = run_streamed(["adb", "shell", f"cd {DEVICE_DIR} && LD_LIBRARY_PATH=. ./se_llama_cli job.json "
                                             "results.json"], raw / "cli_output.txt")
        if code == 0:
            adb("pull", f"{DEVICE_DIR}/results.json", str(raw / "cli_results.json"))
    if code != 0:
        raise ToolError(f"The test program stopped with code {code}; its output is in runs/{args.run_id}/raw/"
                        "cli_output.txt")
    print(f"Wrote runs/{args.run_id}/raw/cli_results.json. Next: python grounding/llama_headset.py check "
          f"{args.run_id} --reference <reference run>")
    return 0


def cmd_bench(args) -> int:
    raw = raw_dir(args.run_id)
    job = json.loads((raw / "job.json").read_text(encoding="utf-8"))
    threads = ",".join(str(t) for t in args.threads)
    print(f"llama-bench: prompt of {args.prompt_tokens} tokens and {args.gen_tokens} generated, threads {threads} ...")
    r = adb("shell", f"cd {DEVICE_DIR} && LD_LIBRARY_PATH=. ./llama-bench -m {job['model']} -t {threads} "
                     f"-p {args.prompt_tokens} -n {args.gen_tokens} -r 3 -o json")
    (raw / "llama_bench.json").write_text(r.stdout, encoding="utf-8")
    print_bench(raw / "llama_bench.json")
    return 0


def print_bench(path: Path) -> None:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for r in rows:
        kind = f"prompt of {r['n_prompt']}" if r.get("n_prompt") else f"{r['n_gen']} generated"
        print(f"  llama-bench, {r.get('n_threads')} threads, {kind}: {r.get('avg_ts', 0):.1f} tokens/s "
              f"(± {r.get('stddev_ts', 0):.1f})")


def med(values) -> float:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def first_difference(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"differs from token {i}"
    return "same" if len(a) == len(b) else f"same start, {len(a)} tokens against {len(b)}"


def cmd_check(args) -> int:
    raw = raw_dir(args.run_id)
    res = json.loads((raw / "cli_results.json").read_text(encoding="utf-8"))
    ref_dirs = [raw_dir(r) for r in args.reference]
    find = lambda name: next((d / name for d in ref_dirs if (d / name).exists()), None)
    print(f"Run {args.run_id}: llama.cpp {res['llama_cpp']}; model loaded in {res['load_ms'] / 1000:.2f} s; "
          f"memory {res['memory_kb_after_load'] / 1024:.0f} MB after loading, peak {res['memory_kb_peak'] / 1024:.0f} MB")
    print(f"  {res['system_info'].strip()}")
    problems, timing = [], {}
    for r in res["runs"]:
        pid, t = r["prompt_id"], r["threads"]
        head = f"threads {t}, {pid}:"
        ref_path = find(f"reference_{pid}_fp32.json")
        ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path else None
        lines = []
        if ref:
            same_ids = r["prompt_token_ids"] == ref["prompt"]["token_ids"]
            lines.append(f"prompt tokens {'same as' if same_ids else 'DIFFERENT from'} the reference "
                         f"({len(r['prompt_token_ids'])})")
            if not same_ids:
                problems.append(f"{head} prompt tokens differ")
            answer = r["uncached"][0]["answer"]["token_ids"]
            lines.append(f"answer {r['uncached'][0]['answer']['text']!r}: "
                         f"{first_difference(answer, ref['answer']['token_ids'])} against the reference")
        cached, uncached = r["cached"], r["uncached"]
        if cached:
            same = cached[0]["answer"]["token_ids"] == uncached[0]["answer"]["token_ids"]
            gap = max(abs(cached[0]["scores"]["logprobs"][c] - uncached[0]["scores"]["logprobs"][c])
                      for c in uncached[0]["scores"]["logprobs"])
            lines.append(f"cached scene: answer {'same' if same else 'DIFFERENT'}, scores within {gap:.4f}")
            if not same or gap > CACHE_TOLERANCE:
                problems.append(f"{head} the cached scene changes the result")
        else:
            problems.append(f"{head} no cached runs (the scene isn't a token prefix of the prompt)")
        cref_path = find(f"reference_{pid}_candidates.json")
        scores = uncached[0]["scores"]["logprobs"]
        if cref_path:
            cref = json.loads(cref_path.read_text(encoding="utf-8"))
            want = {c: v["logprob"] for c, v in cref["candidates"].items()}
            ids_ok = all(r["candidate_ids"][c] == cref["candidates"][c]["token_ids"] for c in want)
            gap = max(abs(scores[c] - want[c]) for c in want)
            order = sorted(want, key=want.get, reverse=True) == sorted(scores, key=scores.get, reverse=True)
            lines.append(f"candidates: best {max(scores, key=scores.get)} (reference {max(want, key=want.get)}), "
                         f"order {'same' if order else 'DIFFERENT'}, largest difference {gap:.3f}"
                         + ("" if ids_ok else ", candidate tokens DIFFER"))
            if not (order and ids_ok and gap <= SCORE_TOLERANCE):
                problems.append(f"{head} candidate scores don't match the reference")
        else:
            lines.append("candidates: " + ", ".join(f"{c} {v:.2f}" for c, v in sorted(scores.items(), key=lambda kv: -kv[1]))
                         + f" (no reference_{pid}_candidates.json in {', '.join(args.reference)})")
        print(f"\n{head}\n  " + "\n  ".join(lines))
        tm = timing.setdefault(t, {"prompt": [], "step": [], "scene": [], "suffix": [], "score": [], "suffix_tokens": []})
        tm["prompt"] += [u["prompt_ms"] for u in uncached]
        tm["step"] += [s for u in uncached for s in u["answer"]["step_ms"]]
        tm["score"] += [u["scores"]["ms"] for u in uncached + cached]
        tm["suffix"] += [c["suffix_ms"] for c in cached]
        tm["suffix_tokens"] += [c["suffix_tokens"] for c in cached]
        if "scene_evaluated_ms" in r:
            tm["scene"].append(r["scene_evaluated_ms"])
    print("\nTimings (medians):")
    for t, tm in timing.items():
        print(f"  threads {t}: whole prompt {med(tm['prompt']):.0f} ms; then {med(tm['step']):.0f} ms per token; "
              f"scene alone {med(tm['scene']):.0f} ms, once; with it cached, the rest "
              f"({med(tm['suffix_tokens']):.0f} tokens) {med(tm['suffix']):.0f} ms; "
              f"scoring {len(res['runs'][0]['candidate_ids'])} candidates {med(tm['score']):.0f} ms")
    if (raw / "llama_bench.json").exists():
        print("\nllama-bench:")
        print_bench(raw / "llama_bench.json")
    if problems:
        print("\nResult: FAIL.\n  " + "\n  ".join(problems))
        return 1
    print("\nResult: PASS. Same tokens as the PC, the cached scene changes nothing, and the candidate scores match "
          "the reference where it exists.")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(prog="python grounding/llama_headset.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="write the job")
    p.add_argument("run_id", metavar="RUN_ID")
    p.add_argument("description", help="e.g. grounding/models/qwen2.5-0.5b-instruct.json")
    p.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    p.add_argument("--threads", nargs="+", type=int, default=[4, 2])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--quant", default="q8_0", help="the GGUF's weight type, as grounding/export_gguf.py named it")
    p.add_argument("--local", metavar="BIN_DIR", help="run on this PC with a local build")
    for name in ("push", "run", "bench"):
        q = sub.add_parser(name)
        q.add_argument("run_id", metavar="RUN_ID")
        if name == "run":
            q.add_argument("--local", metavar="BIN_DIR", help="run on this PC with the build in BIN_DIR")
        if name == "bench":
            q.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
            q.add_argument("--prompt-tokens", type=int, default=230)
            q.add_argument("--gen-tokens", type=int, default=32)
    c = sub.add_parser("check")
    c.add_argument("run_id", metavar="RUN_ID")
    c.add_argument("--reference", required=True, nargs="+", metavar="RUN", help="the runs holding the PC references")
    args = parser.parse_args(argv)
    try:
        return {"prepare": cmd_prepare, "push": cmd_push, "run": cmd_run, "bench": cmd_bench,
                "check": cmd_check}[args.command](args)
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
