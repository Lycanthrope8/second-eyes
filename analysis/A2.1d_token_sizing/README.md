# A2.1d token-sizing audit (serializer proposal v0.2, section 12)

A temporary offline probe, not the serializer. It renders the two v0.2 formats for the four inputs of §12.4, wraps them
with A1's verified chat template and the §12.5 audit system message, and measures exact token counts. It changes no
production module and no fixture.

**Status.** Sandbox measurements complete: the pinned tokenizer files and llama.cpp b11277's `se_tokenize`, with the
release's Qwen2 vocabulary, give identical token IDs on every audited input. **Pending:** reproduction on JH's
current repository, and the check that the deployed Q8_0 GGUF's own tokenizer is equivalent. Until that check passes,
the counts are **not labelled runtime-exact for the deployed model**. See *Pending* below for the Windows command.

## Inputs (§12.4)

| Case | Scene record | Command | Objects | Profile |
|---|---|---|---|---|
| `a17.annotated` | `grounding/tests/fixtures/contract/valid/scene.a17.annotated.json` | `command.a17.c001.json` (`fx.a17.c001`) | all 6 | annotated |
| `a17.restricted` | `…/contract/valid/scene.a17.restricted.json` | same | all 6 | restricted |
| `dirh10.annotated` | `grounding/tests/fixtures/directions/scene.h.annotated.json` | `command.h.base.json` (`fx.dir.h.base`) | first ten IDs, `obj_001`–`obj_010` | annotated |
| `dirh10.restricted` | `…/directions/scene.h.restricted.json` | same | the same ten IDs | restricted |

The matched profiles hold the same IDs, which the script checks. The ten-object selection is the §12.4 ID-based stress
check: its boxes sit centimetres apart. It is not a natural scene. Fixtures are read, never changed.

Every input is rendered in `coordinates_v1` and `coordinates_relations_v1` as v0.2 specifies:
- mirror removal: only right and in_front_of are serialized;
- losslessly compressed truth and conditional maps;
- full-precision distances;
- full user-to-anchor coverage;
- configuration hashes in metadata only;
- the §12.5 header, semantics and system message, verbatim.

## Tokenizer and wrapper identity

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` (model description and tokenizer files agree) |
| Tokenizer files | tracked in `quest-app/Assets/SecondEyes/Models/qwen2.5-0.5b-instruct/`, byte-identical to that revision per its README |
| Tokenizer used | `Hugging Face Qwen2TokenizerFast from quest-app/Assets/SecondEyes/Models/qwen2.5-0.5b-instruct`; transformers 4.57.6 (A1's recorded version), tokenizers 0.22.2 |
| Runtime path | `llama.cpp b11277 (eae11d22) se_tokenize with tiny-qwen2-tokenizer.gguf` |
| llama.cpp | `b11277 eae11d2217fe9225d1aaba48773b6cca45ae4de9` (`native/llama.cpp.pin`), fetched and host-built with `tools/build_llama.py`; `libse_llama.so` SHA-256 `35d20d1d…cf161` |
| Runtime vocabulary | llama.cpp's `models/ggml-vocab-qwen2.gguf` at that commit (SHA-256 `44c2f46b…d154d8c`), carried by A1's `native/tests/make_tiny_model.py` model (`dff77b1a…bb5497`). This is the stand-in A1 used for its tokenizer parity tests |
| Deployed model | `qwen2.5-0.5b-instruct-q8_0.gguf`, SHA-256 `dd753cd6…c83eb18` (runs r027, r032): **not in the sandbox** |
| Chat template | `"<|im_start|>system\n{1}<|im_end|>\n<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n"`, from `grounding/models/qwen2.5-0.5b-instruct.json` (SHA-256 `1274542348d33a77…`) |
| Applied as | `grounding/scene.formatted()`, as A1's headset jobs did. The script checks that it equals `template.format(user, system)`, the form Meta's runner and the app's chat provider use |
| Special tokens | Special-token text is parsed, and nothing is added: no BOS or EOS (`se_tokenize`: add_special=false, parse_special=true; Hugging Face: `add_special_tokens=False`). `add_bos_token` is false and `bos_token` is null in `tokenizer_config.json` |
| Assistant prefix | `<|im_start|>assistant\n`, once, from the template. No answer prefix and no output reserve |
| Wrapper tokens | head 49 = `<|im_start|>`, `system`, `\n`, the system message (41), `<|im_end|>`, `\n`, `<|im_start|>`, `user`, `\n`; tail 5 = `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n` |
| Environment | Python 3.12.3, Linux x86_64 sandbox; jsonschema 4.26.0; CMake 4.4.3, Ninja 1.13.2, g++ 13.3.0 |

| Tokenizer file | SHA-256 | Equals `model.json`'s fingerprint |
|---|---|---|
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | True |
| `merges.txt` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` | True |
| `tokenizer_config.json` | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` | True |

## Calibration against A1's records

A1 recorded, in `docs/llama-cpp.md` and A1.8b's note: the four A1 prompts as 230, 232, 230 and 227 tokens; their
cached scene as 191; and token-ID equality between Hugging Face and llama.cpp. Here each prompt is formatted from
`grounding/prompts/*.json` with the model description's template, as `grounding/llama_headset.py prepare` did.

| A1 prompt | Recorded tokens | Pinned files | Runtime path | Recorded scene | Measured scene |
|---|---|---|---|---|---|
| a17-fixed | 230 | 230 | 230 | 191 | 191, 191 |
| a17-front | 232 | 232 | 232 | 191 | 191, 191 |
| a17-left | 230 | 230 | 230 | 191 | 191, 191 |
| a17-table | 227 | 227 | 227 | 191 | 191, 191 |

All four counts, and the scene prefix, reproduce the record exactly with both tokenizers, and the scene is a token
prefix of the whole prompt.
- `a17-fixed.json` is the earlier five-object prompt with category-bearing IDs. It is not one of the audit inputs.
- The raw reference files with A1's token IDs (`runs/…/raw/`) aren't tracked, so exact-ID calibration against them is
  part of the pending Windows run.
- A1's 1.39 s is the median uncached whole-prompt time over all four prompts (227–232 tokens, so the label "230
  tokens" is their median). It was measured from `adb shell` at 2 threads (run r027). `grounding/llama_headset.py` takes
  that median, and its `llama-bench` default is a 230-token prompt (170 tokens/s at 2 threads).

## Parity

51 of 53 texts give identical token IDs from the pinned files and the runtime path. The texts are:
- the eight wrapped prompts;
- the eight documents alone;
- the eight static-prefix inputs;
- the four A1 prompts;
- 25 representative strings: object IDs including `obj_1000`, signed and long decimals, `1e-06`, JSON punctuation, NFC Unicode, quotes and an escaped newline.

The two differences are deliberately decomposed (non-NFC) strings, `cafe\u0301` and `A\u030angstro\u0308m`.
Hugging Face's Qwen2 tokenizer normalizes to NFC and llama.cpp b11277 doesn't. All eight audit inputs are NFC, so no
count is affected. It is a real PC-versus-headset difference for any future text that isn't NFC; this audit doesn't
resolve it.

The release vocabulary equals the pinned files in all 151,643 BPE tokens and 151,387 merges, and in
`<|endoftext|>`, `<|im_start|>` and `<|im_end|>`. It lacks Qwen2.5's 19 further added tokens (IDs 151646–151664,
such as `<tool_call>`), which the deployed GGUF should carry. None of them occurs in any audited text; the script
scans for them.

## Results

| Input | Objects | Profile | Wrapped bytes | **Wrapped tokens** | Document alone | Static lines | Dynamic lines | After reusable prefix | Cold s (extrap.) | Warm best case s (extrap.) | Calls + distances |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `a17.annotated.coordinates_v1` | 6 | annotated | 4,309 | **1,218** | 1,164 | 1,096 | 68 | 73 | 7.4 | 0.4 | 0 |
| `a17.annotated.coordinates_relations_v1` | 6 | annotated | 8,789 | **3,209** | 3,155 | 2,634 | 521 | 526 | 19.4 | 3.2 | 342 |
| `a17.restricted.coordinates_v1` | 6 | restricted | 4,249 | **1,147** | 1,093 | 1,025 | 68 | 73 | 6.9 | 0.4 | 0 |
| `a17.restricted.coordinates_relations_v1` | 6 | restricted | 7,876 | **2,659** | 2,605 | 2,084 | 521 | 526 | 16.1 | 3.2 | 342 |
| `dirh10.annotated.coordinates_v1` | 10 | annotated | 4,572 | **1,318** | 1,264 | 1,211 | 53 | 58 | 8.0 | 0.4 | 0 |
| `dirh10.annotated.coordinates_relations_v1` | 10 | annotated | 13,332 | **5,849** | 5,795 | 4,715 | 1,080 | 1,085 | 35.3 | 6.6 | 1,190 |
| `dirh10.restricted.coordinates_v1` | 10 | restricted | 4,532 | **1,288** | 1,234 | 1,181 | 53 | 58 | 7.8 | 0.4 | 0 |
| `dirh10.restricted.coordinates_relations_v1` | 10 | restricted | 17,868 | **8,401** | 8,347 | 7,267 | 1,080 | 1,085 | 50.8 | 6.6 | 1,190 |

*Seconds are a **linear extrapolation from A1, NOT measured Quest latency**: tokens ÷ 165.5 tokens/s (230 tokens in
1.39 s; context above). Every input here is well beyond the 227–232 tokens A1 measured, and nothing is assumed about
how speed changes with length. "Warm best case" counts the tokens after the longest common token-ID prefix of two
tokenizations: the candidate cached prefix (wrapper head plus static lines) and the complete input. It is an upper
bound on reuse, not a measured cache hit. In every input that common prefix is the whole candidate prefix, so the line
boundary before the pose doesn't merge tokens.*

Every input carries the same fixed cost: the 49-token wrapper head, the 137–138-token header, the 613-token semantics
line and the 5-token tail, 804 or 805 tokens in all.

| Block (marginal tokens) | a17.annotated coord | a17.annotated aug | a17.restricted coord | a17.restricted aug | dirh10.annotated coord | dirh10.annotated aug | dirh10.restricted coord | dirh10.restricted aug |
|---|---|---|---|---|---|---|---|---|
| wrapper_head | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 |
| header | 137 | 138 | 137 | 138 | 137 | 138 | 137 | 138 |
| semantics | 613 | 613 | 613 | 613 | 613 | 613 | 613 | 613 |
| objects | 346 | 346 | 275 | 275 | 461 | 461 | 431 | 431 |
| pose | 57 | 57 | 57 | 57 | 38 | 38 | 38 | 38 |
| command | 11 | 11 | 11 | 11 | 15 | 15 | 15 | 15 |
| wrapper_tail | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| near |  | 111 |  | 37 |  | 170 |  | 37 |
| above |  | 182 |  | 37 |  | 37 |  | 37 |
| below |  | 182 |  | 37 |  | 37 |  | 37 |
| on |  | 74 |  | 97 |  | 37 |  | 37 |
| inside |  | 74 |  | 157 |  | 255 |  | 37 |
| between |  | 130 |  | 129 |  | 1714 |  | 4647 |
| intrinsic.right |  | 137 |  | 39 |  | 39 |  | 39 |
| intrinsic.in_front_of |  | 163 |  | 41 |  | 41 |  | 41 |
| center_distance_m |  | 484 |  | 484 |  | 1173 |  | 1173 |
| user_heading.right |  | 44 |  | 44 |  | 62 |  | 62 |
| user_heading.in_front_of |  | 39 |  | 39 |  | 65 |  | 65 |
| user_to_anchor.right |  | 220 |  | 220 |  | 341 |  | 341 |
| user_to_anchor.in_front_of |  | 150 |  | 150 |  | 559 |  | 559 |
| **total** | **1,218** | **3,209** | **1,147** | **2,659** | **1,318** | **5,849** | **1,288** | **8,401** |

Marginal attribution: complete prefixes ending at each line are tokenized, and successive differences taken. They sum
to each total (checked). These are not standalone tokenizations of each block.

Work, augmented inputs only (each count is the number of tuples in that block's domain; coordinates_v1 makes no calls):

| Augmented input | near | above / below / on / inside (each) | between | intrinsic (each) | distances | user_heading (each) | user_to_anchor (each) | total = formula |
|---|---|---|---|---|---|---|---|---|
| `a17.annotated.coordinates_relations_v1` | 15 | 30 | 60 | 30 | 15 | 6 | 30 | 342 = 342 |
| `a17.restricted.coordinates_relations_v1` | 15 | 30 | 60 | 30 | 15 | 6 | 30 | 342 = 342 |
| `dirh10.annotated.coordinates_relations_v1` | 45 | 90 | 360 | 90 | 45 | 10 | 90 | 1190 = 1190 |
| `dirh10.restricted.coordinates_relations_v1` | 45 | 90 | 360 | 90 | 45 | 10 | 90 | 1190 = 1190 |

Conditional membership, diagnostic only (the delivered inputs keep these fields):

| Input | Block conditional maps | Object and pose `conditional_fields` |
|---|---|---|
| `a17.annotated.coordinates_v1` | 0 tokens, 0 B | 17 tokens, 42 B |
| `a17.annotated.coordinates_relations_v1` | 138 tokens, 624 B | 17 tokens, 42 B |
| `a17.restricted.coordinates_v1` | 0 tokens, 0 B | 32 tokens, 82 B |
| `a17.restricted.coordinates_relations_v1` | 311 tokens, 948 B | 32 tokens, 82 B |
| `dirh10.annotated.coordinates_v1` | 0 tokens, 0 B | 25 tokens, 54 B |
| `dirh10.annotated.coordinates_relations_v1` | 135 tokens, 624 B | 25 tokens, 54 B |
| `dirh10.restricted.coordinates_v1` | 0 tokens, 0 B | 55 tokens, 134 B |
| `dirh10.restricted.coordinates_relations_v1` | 1,574 tokens, 3,178 B | 55 tokens, 134 B |

The large restricted ten-object figures come from `between`:
- its 360 triples split 176 FALSE and 184 UNKNOWN;
- its conditional membership splits 280 true and 80 false: degenerate-anchor UNKNOWNs consult only centres.

So both its truth exceptions and its conditional exceptions are long.

## Self-checks (all passed)

The script checks:
- every compressed block expands back to the evaluated table;
- left and behind reconstructed by mirroring equal the evaluator's own results, using 132 and 380 verification calls that are not serialized;
- calls plus distances equal the §5 formula (342 at n = 6, 1,190 at n = 10);
- every line is one JSON value ending in LF, and re-encodes to itself;
- reversing the input object order leaves the bytes unchanged;
- the two template paths agree;
- the marginal sums equal the totals;
- the matched profiles hold the same IDs.

## Interpretation choices to confirm

These details aren't fixed by v0.2 and slightly change the measured input. Sizes were checked ad hoc on the
augmented inputs.
1. `pose` is a keyed object in §4's order, because the header declares no pose columns. A positional array would be
   15 tokens shorter.
2. `parameters` is `{"thresholds", "bands", "directions"}`, each sorted. One flat sorted object would be 9 tokens
   shorter.
3. `rotation_support` is never listed in `conditional_fields`; only the six evidence-bearing columns are. No audited
   object has a conditional rotation.
4. The header's `coverage` sentence is also in `coordinates_v1`, since §12.5 lists it for both.
5. The document's final LF comes before `<|im_end|>`, because the template inserts the user text unchanged.

## Not measured here

- The task instructions, output suffix or reserve, query grammar and ASK design of the later grounding designs.
- Actual KV-cache reuse in the app, Quest latency, memory and thermals.
- Dataset scenes.

Final fit therefore can't be certified from these counts (§12.5–12.6).

## Pending: run on your PC

From the repository root, in the project's `.venv` (transformers from `grounding/requirements.txt`), writing outside
the repository:

```
python analysis/A2.1d_token_sizing/token_sizing_audit.py --out $env:TEMP\a21d_audit --hf-reference grounding/models/qwen2.5-0.5b-instruct/hf --deployed-gguf grounding/models/qwen2.5-0.5b-instruct/gguf/qwen2.5-0.5b-instruct-q8_0.gguf --release-vocab third_party/llama.cpp/models/ggml-vocab-qwen2.gguf --a1-references runs/20260929_A1_r016/raw --compare-with analysis/A2.1d_token_sizing/output/results.json
```

It writes `report.md` and `results.json` there and prints the report. What to look for:
- *Against the delivered results*: same text and token IDs for all eight. This confirms your current tracked files and
  fixtures equal the snapshot measured here.
- *Deployed GGUF against the release vocabulary*: SHA-256 matching runs r027 and r032, and "equivalent for the audited
  texts: True". That holds if the BPE model, pre-tokenizer and merges are equal and the token lists differ only at
  151646–151664.
- *Calibration*: exact equality with A1's recorded reference token IDs, if the raw files are present.
- *Parity*: with A1's own Hugging Face download.

Any input whose files are missing is reported as skipped, not failed.

Optional, only if you already have a host build (`python tools/build_llama.py host`, which needs CMake, Ninja and a
compiler). This tokenizes with the deployed GGUF itself through `se_tokenize`; it loads the file but runs no inference:

```
python analysis/A2.1d_token_sizing/token_sizing_audit.py --out $env:TEMP\a21d_runtime --runtime-lib native/out/host/se_llama.dll --runtime-gguf grounding/models/qwen2.5-0.5b-instruct/gguf/qwen2.5-0.5b-instruct-q8_0.gguf --compare-with analysis/A2.1d_token_sizing/output/results.json
```

## How the sandbox run was made

The repository at the accepted state of `2fe98ee`: the earlier snapshot plus every accepted package.

```
pip install cmake ninja transformers==4.57.6
python tools/build_llama.py fetch && python tools/build_llama.py host
python native/tests/make_tiny_model.py third_party/llama.cpp /tmp/tiny-qwen2-tokenizer.gguf
python analysis/A2.1d_token_sizing/token_sizing_audit.py --out analysis/A2.1d_token_sizing/output --runtime-lib native/out/host/libse_llama.so --runtime-gguf /tmp/tiny-qwen2-tokenizer.gguf
```

`output/` holds:
- `inputs/*.jsonl`: the eight documents, exact bytes;
- `inputs/*.prompt.txt`: the eight wrapped prompts;
- `token_ids.json`: every wrapped prompt's and document's token IDs;
- `results.json`: everything measured, with hashes;
- `report.md`: the same tables.

| Input | Text SHA-256 (wrapped) | Token-ID SHA-256 (wrapped) |
|---|---|---|
| `a17.annotated.coordinates_v1` | `1a46ebcd7f2f990e…` | `d30c82e8cc9f71b9…` |
| `a17.annotated.coordinates_relations_v1` | `c6a7a320a3444ff8…` | `8944d64df10ed039…` |
| `a17.restricted.coordinates_v1` | `a7d48e2897c25e8d…` | `c0a895fd8720cfa1…` |
| `a17.restricted.coordinates_relations_v1` | `5aae08f5b070fcea…` | `972a735d286d4ccb…` |
| `dirh10.annotated.coordinates_v1` | `012f0f63d7d4b35b…` | `64622e47cb0adcc2…` |
| `dirh10.annotated.coordinates_relations_v1` | `881cb8c7010a8f08…` | `e28b25c19e8c5231…` |
| `dirh10.restricted.coordinates_v1` | `254c2e8f51c3758c…` | `c224d4fba6a75e3e…` |
| `dirh10.restricted.coordinates_relations_v1` | `ec690b413a77d8d4…` | `81c621f792adf06f…` |
