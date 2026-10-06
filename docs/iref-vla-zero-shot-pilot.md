# IRef-VLA zero-shot direct-selection pilot (A2.3a)

The pilot (D81) checks that model loading, exact inference requests, bounded candidate scoring and result records work
together on real imported scenes. A successful run means technically valid execution, not correct grounding: it reads
no annotation, scores nothing as correct, chooses no format and evaluates no gate.

## Commands

```
python -m grounding.inference.iref_vla prepare --bundle A22D_BUNDLE --tokenizer-dir quest-app/Assets/SecondEyes/Models/qwen2.5-0.5b-instruct --model-description grounding/models/qwen2.5-0.5b-instruct.json --out NEW_REQUESTS
python -m grounding.inference.iref_vla run --requests REQUESTS --model-dir grounding/models/qwen2.5-0.5b-instruct/hf --device cuda --tokenizer-dir quest-app/Assets/SecondEyes/Models/qwen2.5-0.5b-instruct --out NEW_RESULTS
```

Exit 0: published, planned exclusions included; 2: invalid input, identity or verification failure; 3: an unexpected
execution or write failure. Existing output folders are refused. `prepare` never imports PyTorch; `run` needs PyTorch
and the pinned transformers 4.57.6 and tokenizers 0.22.2, whatever the environment is called. Tests:
`python grounding/tests/test_iref_vla_pilot.py --unit-only` (fakes), without the flag (also the pinned tokenizer), or
`--requests DIR` (also verifies a prepared request folder). No real model runs in the tests.

## The choice interface

For a scene of n objects (0 to 10) in their serialized order, codes A..J name the first n objects and K is ASK at every
size; unused codes are not offered and every retained object stays eligible, unknown categories included. The prompt is
Qwen's chat wrapper around the protocol's system message, then a user turn holding the compact choices line
(`{"choices":[["A","obj_..."],...,["K","ASK"]]}` and LF) followed by the A2.2d document's exact bytes, then the
assistant prefix. The command text is data. With the pinned tokenizer every offered code must be one non-special token
with its pinned ID (A..K are 32..42) at that exact boundary: `encode(prompt + code) == encode(prompt) + [id]`, checked
for every request when preparing and again when running.

## Requests

`prepare` runs A2.2d's bundle verifier first, then checks roles, plain-integer versions, duplicates, four-way
membership, path containment, document hashes, scene and command consistency and object order. It takes the parents
whose inputs rendered in both views and both formats, orders them by the SHA-256 of the protocol's salt plus the parent
ID, and keeps the first 32, refusing to continue unless the eligible count and the list's hash equal the protocol's
literal values. Each parent gets four requests in a fixed view and format order. A request whose prompt plus its one
continuation token exceeds 8,192 tokens is kept and marked `context_budget_exceeded`, never truncated or replaced. The
folder holds the protocol, a closed index, each request's exact prompt bytes and token-ID array, and a manifest with
every file's hash and the source bundle's provenance; it carries nothing answer-bearing.

## Running

`run` verifies every request file and identity, ties the local weights to the pinned revision (Hugging Face's download
metadata in `.cache/huggingface/download/`, or `--expected-weights-sha256` with the published value) and re-tokenizes
every prompt for exact agreement before loading anything. The model loads locally in float32 with eager attention, TF32
off and deterministic algorithms requested, on the device named, with no CPU fallback. A canary first compares the
first request's offered logits from the production path (`logits_to_keep=1`) with an independent extraction (the last
hidden state through `lm_head`): agreement within 1e-5 absolute plus 1e-5 relative and the same choice, or the run
stops. Then each request gets one forward without a cache: the largest offered logit chooses, an exact maximal tie gives
K with `exact_score_tie`, and an empty scene gives K without a model call. A non-finite output stops the run; no success
bundle is published, and a folder named `<out>.failed-<time>` holds the diagnostic.

## Results

`results.jsonl` (one closed row per request: the key, identities, mapping, every offered logit, its full-vocabulary log
probability and its restricted share, the choice and why, timings), `summary.json`, `report.md` and `manifest.json`
(checkpoint evidence, software and device, the canary, counts, file hashes). Shares are conditional ranking shares, not
calibrated probabilities. Agreement between formats is not accuracy. Timings are PC timings, not Quest timings.
