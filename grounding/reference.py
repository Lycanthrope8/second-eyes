#!/usr/bin/env python3
"""Make the PC reference answer for a fixed prompt, following Meta's on-device runner step by step.

    python grounding/reference.py <run ID> grounding/models/qwen2.5-0.5b-instruct.json grounding/prompts/a17-fixed.json
    python grounding/reference.py <run ID> grounding/models/qwen2.5-0.5b-instruct.json grounding/prompts/a17-fixed.json --variant fp16w

It builds the prompt exactly as Meta's runner does (chat template, {0} = user text, {1} = system
message), tokenizes it with the model's own tokenizer, then generates with onnxruntime using the
runner's loop (grounding/meta_runner.py). It does this twice, to show the answer is the same every time.
For the full-precision model it also asks PyTorch for the same answer (greedy, no penalties), which
checks the export itself. The result goes to runs/<run ID>/raw/reference_<prompt id>_<variant>.json.
Export the model first: python grounding/export_onnx.py <description> [--fp16-weights]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from meta_runner import apply_chat_template, check_io, generate

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FOLDERS = {"fp32": "onnx", "fp16w": "onnx_fp16w"}


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def pytorch_answer(hf_dir: Path, prompt_ids: list, arch: dict, max_new_tokens: int) -> list:
    """transformers' greedy answer: most likely token, no sampling, no repetition penalty."""
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig
    model = AutoModelForCausalLM.from_pretrained(str(hf_dir), torch_dtype=torch.float32)
    model.eval()
    config = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                              eos_token_id=arch["eos_token_id"], pad_token_id=arch["eos_token_id"])
    with torch.no_grad():
        ids = torch.tensor([prompt_ids])
        out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=config)
    answer = out[0, len(prompt_ids):].tolist()
    return answer[:-1] if answer and answer[-1] == arch["eos_token_id"] else answer


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(prog="python grounding/reference.py",
                                     description="Make the PC reference answer that the headset must reproduce.")
    parser.add_argument("run_id", metavar="RUN_ID")
    parser.add_argument("description", help="e.g. grounding/models/qwen2.5-0.5b-instruct.json")
    parser.add_argument("prompt", help="e.g. grounding/prompts/a17-fixed.json")
    parser.add_argument("--variant", choices=sorted(FOLDERS), default="fp32",
                        help="fp32 = the exported model; fp16w = its 16-bit-weights copy")
    parser.add_argument("--skip-pytorch", action="store_true", help="don't run the PyTorch check (fp32 only)")
    args = parser.parse_args(argv)
    try:
        run = RUNS / args.run_id
        if not (run / "config.yaml").is_file():
            raise ToolError(f"No run {args.run_id} in runs/. Create it first with: python tools/runs.py new ...")
        desc_path = Path(args.description)
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        prompt = json.loads(Path(args.prompt).read_text(encoding="utf-8"))
        model_dir = desc_path.parent / desc["name"]
        onnx_path = model_dir / FOLDERS[args.variant] / "model.onnx"
        export_path = model_dir / "export.json"
        if not onnx_path.exists() or not export_path.exists():
            raise ToolError(f"{onnx_path.as_posix()} isn't there yet. Export first: python grounding/export_onnx.py "
                            f"{desc_path.as_posix()}" + (" --fp16-weights" if args.variant == "fp16w" else ""))
        export = json.loads(export_path.read_text(encoding="utf-8"))
        arch = desc["architecture"]

        from transformers import AutoTokenizer
        import onnxruntime as ort
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "hf"))
        formatted = apply_chat_template(desc["chat_template"], prompt["user"], prompt["system"])
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > desc["max_prompt_length"]:
            raise ToolError(f"The prompt has {len(prompt_ids)} tokens, over max_prompt_length "
                            f"{desc['max_prompt_length']}; Meta's runner would cut its beginning.")

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        names = check_io([i.name for i in session.get_inputs()], [o.name for o in session.get_outputs()],
                         arch["max_layers"])
        if not names["ok"]:
            raise ToolError(f"The model's names don't match Meta's runner: {names}")

        print(f"Prompt: {len(prompt_ids)} tokens. Generating ({args.variant}) twice ...")
        first = generate(session, prompt_ids, arch, desc["max_new_tokens"])
        second = generate(session, prompt_ids, arch, desc["max_new_tokens"])
        repeatable = first["tokens"] == second["tokens"]

        pytorch_match = None
        if args.variant == "fp32" and not args.skip_pytorch:
            print("Checking against PyTorch ...")
            pytorch_match = pytorch_answer(model_dir / "hf", prompt_ids, arch, desc["max_new_tokens"]) == first["tokens"]

        tokens = first["tokens"]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        text_meta_style = "".join(tokenizer.decode([t], skip_special_tokens=False) for t in tokens)
        try:
            target = json.loads(text.strip()).get("target")
        except (ValueError, AttributeError):
            target = None

        result = {"format": 1, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                  "run_id": args.run_id, "variant": args.variant,
                  "model": {"description": desc_path.as_posix(), "name": desc["name"], "hf_repo": export["hf_repo"],
                            "hf_commit": export["hf_commit"],
                            "files": export["variants"].get(args.variant, {}).get("files")},
                  "settings": {"max_new_tokens": desc["max_new_tokens"], "max_prompt_length": desc["max_prompt_length"],
                               "eos_token_id": arch["eos_token_id"], "rule": "greedy, as Meta's TextOnlyLlmRunner"},
                  "prompt": {"id": prompt["id"], "system": prompt["system"], "user": prompt["user"],
                             "formatted": formatted, "token_ids": prompt_ids, "tokens": len(prompt_ids)},
                  "answer": {"token_ids": tokens, "tokens": len(tokens), "stop": first["stop"], "text": text,
                             "text_meta_style": text_meta_style, "json_target": target,
                             "expected_target": prompt.get("expected_target")},
                  "checks": {"repeatable": repeatable, "pytorch_match": pytorch_match},
                  "timing_pc_ms": {"prefill": first["prefill_ms"], "per_token_mean": first["step_ms_mean"]},
                  "packages": export["packages"]}
        out = run / "raw" / f"reference_{prompt['id']}_{args.variant}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(f"Answer ({len(tokens)} tokens, stopped by {first['stop']}): {text}")
        print(f"  target {target} (expected {prompt.get('expected_target')})")
        print(f"  repeatable: {'yes' if repeatable else 'NO'}; PyTorch gives the same tokens: "
              f"{'-' if pytorch_match is None else ('yes' if pytorch_match else 'NO')}")
        print(f"Wrote {out.relative_to(ROOT).as_posix()}")
        return 0 if repeatable and pytorch_match is not False else 1
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
