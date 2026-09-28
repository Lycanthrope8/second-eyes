"""A Python mirror of Meta's on-device runner (TextOnlyLlmRunner, Meta XR SDK v207), for the PC side.

It holds the two things our PC tools must match exactly:
- the model's input and output names that the runner feeds and reads, and
- the runner's generation loop: the whole prompt at once, then one token at a time with the cache,
  always the most likely token, stopping at the end token or after max_new_tokens.

Source of truth: TextOnlyLLMRunner.cs in quest-app/Library/PackageCache/com.meta.xr.sdk.core@*/Scripts/
BuildingBlocks/AIBlocks/. If Meta changes the runner, change this file to match (see docs/meta-ai.md).
"""

from __future__ import annotations

import time

import numpy as np

NUMPY_TYPES = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
               "tensor(float)": np.float32, "tensor(float16)": np.float16}


def expected_inputs(layers: int) -> list:
    names = ["input_ids", "attention_mask", "position_ids"]
    for i in range(layers):
        names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
    return names


def expected_outputs(layers: int) -> list:
    names = ["logits"]
    for i in range(layers):
        names += [f"present.{i}.key", f"present.{i}.value"]
    return names


def check_io(input_names, output_names, layers: int) -> dict:
    """Compare a model's input and output names with what Meta's runner uses."""
    want_in, want_out = expected_inputs(layers), expected_outputs(layers)
    missing = [n for n in want_in if n not in input_names] + [n for n in want_out if n not in output_names]
    extra = [n for n in input_names if n not in want_in]   # an extra input the runner never feeds breaks it
    return {"ok": not missing and not extra, "missing": missing, "extra_inputs": extra}


def apply_chat_template(template: str, user: str, system: str) -> str:
    """Same as OnDeviceLlmConfig.ApplyChatTemplate: {0} is the user text, {1} the system message."""
    return template.format(user, system)


def generate(session, prompt_ids: list, arch: dict, max_new_tokens: int) -> dict:
    """Run Meta's generation loop with an onnxruntime session. Returns the answer's token IDs and timings."""
    types = {i.name: NUMPY_TYPES[i.type] for i in session.get_inputs()}
    layers, eos = arch["max_layers"], arch["eos_token_id"]
    out_names = [o.name for o in session.get_outputs()]
    n = len(prompt_ids)

    feeds = {"input_ids": np.array([prompt_ids], dtype=types["input_ids"]),
             "attention_mask": np.ones((1, n), dtype=types["attention_mask"]),
             "position_ids": np.arange(n, dtype=types["position_ids"])[None, :]}
    for i in range(layers):
        for part in ("key", "value"):
            name = f"past_key_values.{i}.{part}"
            feeds[name] = np.zeros((1, arch["num_key_value_heads"], 0, arch["head_dim"]), dtype=types[name])

    started = time.perf_counter()
    outs = dict(zip(out_names, session.run(out_names, feeds)))
    prefill_ms = (time.perf_counter() - started) * 1000
    token = int(np.argmax(outs["logits"][0, n - 1]))       # first maximum wins, like FindArgMaxCpu
    tokens = [token] if token != eos else []

    step, step_ms = 1, []
    while step < max_new_tokens and token != eos:
        length = n + step
        feeds = {"input_ids": np.array([[token]], dtype=types["input_ids"]),
                 "attention_mask": np.ones((1, length), dtype=types["attention_mask"]),
                 "position_ids": np.array([[length - 1]], dtype=types["position_ids"])}
        for i in range(layers):
            feeds[f"past_key_values.{i}.key"] = outs[f"present.{i}.key"]
            feeds[f"past_key_values.{i}.value"] = outs[f"present.{i}.value"]
        started = time.perf_counter()
        outs = dict(zip(out_names, session.run(out_names, feeds)))
        step_ms.append((time.perf_counter() - started) * 1000)
        token = int(np.argmax(outs["logits"][0, 0]))
        if token != eos:
            tokens.append(token)
        step += 1

    return {"tokens": tokens, "stop": "eos" if token == eos else "max_new_tokens",
            "prefill_ms": round(prefill_ms, 1),
            "step_ms_mean": round(sum(step_ms) / len(step_ms), 1) if step_ms else None}
