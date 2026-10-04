# A2.1d indexed relation encoding: bounded token audit

One candidate encoding, audited as `Second_Eyes_A2_1d_Indexed_Audit_Brief.md` specifies. It is not the serializer.
The v0.2 audit in `analysis/A2.1d_token_sizing/` stays the unchanged reference: this extension only reads its saved
inputs and `results.json`. The audit imports that audit's script, read-only, to reuse its tokenizer, special-token
options, chat wrapper and number spelling. No production module, fixture, configuration or tokenizer asset changes.

**Status.** Executed in the sandbox:
- all 64 verification checks pass;
- the eight new inputs are measured with the pinned tokenizer.

**Pending:** reproducing this on your PC (command below), and any direct check with the deployed GGUF.

## What was done

1. The eight saved v0.2 documents are read; each is confirmed to be the delivered file by its hash in the original
   `results.json`.
2. Every original derived block is validated and expanded over its complete domain: truth state and conditional
   membership for every legal tuple, using its explicit default; omitted exceptions are never taken as FALSE.
   Every distance is expanded the same way.
3. Each expansion is encoded in the brief's indexed rows: uniform shortcuts, `-` exclusions, upper-triangle rows and
   `[a,b,row]` rows. The new block is decoded by separately written layout logic and compared with the original
   expansion exactly, including UNKNOWN states, every conditional membership, and each distance's value and type,
   null included. The decoded block must also re-encode to identical bytes.
4. Only the derived blocks and the header change. The new header has `serializer_version` 2, the format names
   `coordinates_v2` and `coordinates_relations_v2`, and the brief's exact coverage legend, in both formats. The
   semantics, objects, pose and command lines are copied byte for byte, and the system message, wrapper, line order and
   final-newline placement are unchanged.
5. Tokens are counted only after every check passes. A failed check makes the run unsuccessful, and no tokens are
   reported.

No geometry is recomputed and no predicate is called. Codec conversion time was not measured.

## Verification (64 of 64 passed)

- **8.1 literal examples.** The brief's five Section 4 examples decode to the stated mappings and re-encode to the
  exact text. Their expectations sit in `expected_cases.py`, written before the codec.
- **8.2 unit cases.** 312 codec unit cases cover every domain at n = 0, 1, 2, 3, 6 and 10: empty domains, uniform
  T/F/U, uniform and mixed membership, and mixed truth rows. All round-trip in canonical form.
- **8.3 and 8.4 saved inputs.** In all four augmented inputs, all 13 derived blocks decode to exactly the original
  maps, conditional membership on UNKNOWNs included. For example, the restricted ten-object input's `between` holds
  360 tuples, 184 of them UNKNOWN and 280 conditional.
- **8.4 distances.** Every distance and null is preserved with its type. The literal 0.1006 and 0.1508 stay exact, and
  a zero distance stays distinct from a null one.
- **8.5 reverse retrieval.** near, distances and between's anchors read the same in either order. Ordered pairs aren't
  symmetrized: right(A,B)=T and right(B,A)=F.
- **8.6 exclusions.** Excluded cells, repeated IDs, empty domains and IDs outside objects get no value from a
  broadcast or from mirroring. Mirroring exchanges T and F, keeps U and keeps membership.
- **8.7 orientation.**
  - The literal directed example: row = target, column = anchor.
  - A hand-worked case from the a17 input (in `expected_cases.py`): right(obj_005, obj_001) is T and its transpose F.
  - D07 and D10 themselves aren't among the four audited inputs, which hold one pose each, so their viewer-reversal
    evidence isn't available here.
- **8.8 rejections.** 27 malformed blocks are rejected with a named error: wrong lengths and diagonals, illegal
  symbols, duplicate, missing or out-of-order between rows, out-of-range or non-integer indices, unknown fields, `[]`
  for a nonempty domain, `true` for an empty one, and bad distance rows.
- **8.9 shared content.** Semantics, objects, pose and command are byte-identical to the originals. The two new headers
  differ only in `format`, and both formats share the same evidence.
- **8.10 boundary.** The static/dynamic boundary stays at the pose line, with every viewer-relative block after it.
- **8.11 indices.** O = [obj_001, obj_1000, obj_999] reads T, F, U through O. Suffix-derived indices 999 and 998 don't
  exist, and numeric suffix order would misread obj_999.
- **8.12 tokenizer.** The same pinned tokenizer, special-token options and wrapper. Calibration reproduces A1's
  records: the four A1 prompts at 230, 232, 230 and 227 tokens, and their scene at 191.

| A1 prompt | Recorded | Pinned files | Runtime path | Scene recorded | Scene measured |
|---|---|---|---|---|---|
| a17-fixed | 230 | 230 | 230 | 191 | 191, 191 |
| a17-front | 232 | 232 | 232 | 191 | 191, 191 |
| a17-left | 230 | 230 | 230 | 191 | 191, 191 |
| a17-table | 227 | 227 | 227 | 191 | 191, 191 |

## Old against new

| Input | Old tokens | New tokens | Change | Old after reusable prefix | New after reusable prefix | Old bytes | New bytes |
|---|---|---|---|---|---|---|---|
| `a17.annotated.coordinates_relations_v2` | 3,209 | **2,266** | -943 | 526 | 220 | 8,789 | 7,081 |
| `a17.restricted.coordinates_relations_v2` | 2,659 | **2,106** | -553 | 526 | 220 | 7,876 | 6,777 |
| `dirh10.annotated.coordinates_relations_v2` | 5,849 | **2,998** | -2,851 | 1,085 | 290 | 13,332 | 8,304 |
| `dirh10.restricted.coordinates_relations_v2` | 8,401 | **3,625** | -4,776 | 1,085 | 290 | 17,868 | 8,913 |
| `a17.annotated.coordinates_v2` | 1,218 | **1,343** | +125 | 73 | 73 | 4,309 | 4,846 |
| `a17.restricted.coordinates_v2` | 1,147 | **1,272** | +125 | 73 | 73 | 4,249 | 4,786 |
| `dirh10.annotated.coordinates_v2` | 1,318 | **1,443** | +125 | 58 | 58 | 4,572 | 5,109 |
| `dirh10.restricted.coordinates_v2` | 1,288 | **1,413** | +125 | 58 | 58 | 4,532 | 5,069 |

- **The legend.** It costs +125 tokens in every input, the header going from 137–138 to 262–263 tokens. The matched
  coordinate controls carry it too, so the coordinate change, +125, is the legend alone.
- **After the reusable prefix.** The pose-dependent part of the augmented inputs is 220 tokens at six objects (was
  526) and 290 at ten (was 1,085). The coordinate inputs are unchanged at 73 and 58.

| Block (marginal tokens, old → new) | a17.annotated | a17.restricted | dirh10.annotated | dirh10.restricted |
|---|---|---|---|---|
| wrapper_head | 49 → 49 | 49 → 49 | 49 → 49 | 49 → 49 |
| header | 138 → 263 | 138 → 263 | 138 → 263 | 138 → 263 |
| semantics | 613 → 613 | 613 → 613 | 613 → 613 | 613 → 613 |
| objects | 346 → 346 | 275 → 275 | 461 → 461 | 431 → 431 |
| near | 111 → 34 | 37 → 22 | 170 → 45 | 37 → 22 |
| above | 182 → 48 | 37 → 22 | 37 → 22 | 37 → 22 |
| below | 182 → 48 | 37 → 22 | 37 → 22 | 37 → 22 |
| on | 74 → 48 | 97 → 62 | 37 → 22 | 37 → 22 |
| inside | 74 → 48 | 157 → 62 | 255 → 69 | 37 → 22 |
| between | 130 → 163 | 129 → 162 | 1714 → 509 | 4647 → 1236 |
| intrinsic.right | 137 → 50 | 39 → 24 | 39 → 24 | 39 → 24 |
| intrinsic.in_front_of | 163 → 52 | 41 → 26 | 41 → 26 | 41 → 26 |
| center_distance_m | 484 → 284 | 484 → 284 | 1173 → 583 | 1173 → 583 |
| pose | 57 → 57 | 57 → 57 | 38 → 38 | 38 → 38 |
| user_heading.right | 44 → 23 | 44 → 23 | 62 → 26 | 62 → 26 |
| user_heading.in_front_of | 39 → 24 | 39 → 24 | 65 → 28 | 65 → 28 |
| user_to_anchor.right | 220 → 48 | 220 → 48 | 341 → 88 | 341 → 88 |
| user_to_anchor.in_front_of | 150 → 52 | 150 → 52 | 559 → 90 | 559 → 90 |
| command | 11 → 11 | 11 → 11 | 15 → 15 | 15 → 15 |
| wrapper_tail | 5 → 5 | 5 → 5 | 5 → 5 | 5 → 5 |

Every input shares the same wrapper head (49), semantics (613), objects, pose, command and tail as before.

- **Where it got smaller.** All uniform blocks and every pair-indexed block shrink, since object IDs are no longer
  repeated per tuple.
- **Where it got larger.** `between` grows at six objects, 130 → 163 and 129 → 162: a complete mixed row layout costs
  more there than the old sparse exception lists did.
- **Distances.** They keep full precision; only the repeated ID pairs are gone.

## New inputs

| New input | Bytes | Tokens | Document alone | Static lines | Dynamic lines | After reusable prefix | Block conditional maps (overlapping, not additive) |
|---|---|---|---|---|---|---|---|
| `a17.annotated.coordinates_relations_v2` | 7,081 | 2,266 | 2,212 | 1,997 | 215 | 220 | 52 |
| `a17.restricted.coordinates_relations_v2` | 6,777 | 2,106 | 2,052 | 1,837 | 215 | 220 | 132 |
| `dirh10.annotated.coordinates_relations_v2` | 8,304 | 2,998 | 2,944 | 2,659 | 285 | 290 | 52 |
| `dirh10.restricted.coordinates_relations_v2` | 8,913 | 3,625 | 3,571 | 3,286 | 285 | 290 | 762 |
| `a17.annotated.coordinates_v2` | 4,846 | 1,343 | 1,289 | 1,221 | 68 | 73 | 0 |
| `a17.restricted.coordinates_v2` | 4,786 | 1,272 | 1,218 | 1,150 | 68 | 73 | 0 |
| `dirh10.annotated.coordinates_v2` | 5,109 | 1,443 | 1,389 | 1,336 | 53 | 58 | 0 |
| `dirh10.restricted.coordinates_v2` | 5,069 | 1,413 | 1,359 | 1,306 | 53 | 58 | 0 |

The conditional-map figures are diagnostic differences that overlap the blocks' own tokens; don't add them to the
totals. Marginal attribution works as before: successive complete prefixes are tokenized and differences taken, and
they sum to each total (checked).

`results.json` keeps the 165.5 tokens/s figures per input, cold and best-case warm, only as a labelled illustration.
It is a linear extrapolation from A1, not a gate and not Quest latency, and real costs, output scoring, scene updates
and cache behaviour remain unmeasured at these lengths.

## Tokenizer checks: executed versus metadata

| Check | Kind | Where | Result |
|---|---|---|---|
| Pinned tokenizer files (`quest-app/…/qwen2.5-0.5b-instruct`), transformers 4.57.6 | executed tokenization: all counts | sandbox | all counts in this README |
| llama.cpp b11277 `se_tokenize` (parse special, add none) with the release's `ggml-vocab-qwen2.gguf`, on A1's tiny test model | **executed** runtime-code tokenization; **not** the deployed GGUF | sandbox | 24 of 24 new texts identical (8 prompts, 8 documents, 8 static-prefix inputs) |
| A1's Hugging Face download against the pinned files | executed, but two Hugging Face copies, not the runtime | your PC, previous audit (old texts: 53 of 53) | for the new texts: pending your run with `--hf-reference` |
| Deployed Q8_0 GGUF against the release vocabulary | **metadata comparison only**, previous audit, your PC | your PC | SHA-256 matches runs r027/r032; model, pre-tokenizer, merges and tokens 0–151645 equal; 19 token strings (151646–151664) and 288 token types differ. Its `equivalent_for_audited_texts: True` does **not** evaluate token types, so it isn't executed parity |
| `se_tokenize` with the deployed GGUF itself | executed runtime parity with the deployed model | — | **pending**: no host build on your PC and no deployed GGUF in the sandbox |

The known non-NFC difference (Hugging Face normalizes, llama.cpp doesn't) is unchanged and still open. All inputs here
are NFC, and their text is unchanged.

## Observations, not decisions

- The header's unchanged `unknown` convention still says "[] means known-empty colours", while the new legend says
  "[] means an empty domain" in relation blocks. Both meanings now coexist, in different fields.
- Byte-level losslessness doesn't establish that the small model reads indexed rows as well as tuples. That is a
  separate development or zero-shot question.
- No claim follows from fewer tokens about saved geometry work, memory, model comprehension or Quest latency. The
  1,190 work units at ten objects describe the unchanged upstream construction.

## Reproduce on your PC

From the repository root, in the project's `.venv`, writing outside the repository. It needs
`analysis/A2.1d_token_sizing/` (the previous audit's ZIP) in place:

```
python analysis/A2.1d_indexed_token_sizing/indexed_token_audit.py --out $env:TEMP\a21d_indexed --hf-reference grounding/models/qwen2.5-0.5b-instruct/hf --compare-with analysis/A2.1d_indexed_token_sizing/output/results.json
```

Expect:
- all 64 checks PASS;
- "Against the delivered results": same for all eight;
- parity with the second Hugging Face copy on all 24 new texts.

If you later have a host build, adding `--runtime-lib native/out/host/se_llama.dll --runtime-gguf
grounding/models/qwen2.5-0.5b-instruct/gguf/qwen2.5-0.5b-instruct-q8_0.gguf` gives the direct deployed-GGUF check.

## Files

- `indexed_token_audit.py`: the codec, the checks, the conversion and the measurement.
- `expected_cases.py`: the expectations fixed before the codec.
- `output/inputs/*.jsonl` and `*.prompt.txt`: the eight new documents and wrapped prompts, exact bytes, protected by
  `output/.gitattributes`.
- `output/token_ids.json`, `output/results.json` (every check, measurement and hash) and `output/report.md`.

| Input | Text SHA-256 (wrapped) | Token-ID SHA-256 (wrapped) |
|---|---|---|
| `a17.annotated.coordinates_relations_v2` | `eb939c0f7463f446…` | `6b9a4fb4c5a6c701…` |
| `a17.restricted.coordinates_relations_v2` | `d3203c6df998f73e…` | `5fc1ae73229201b1…` |
| `dirh10.annotated.coordinates_relations_v2` | `bc9ed0393aae7022…` | `2c6cc28ebdb81fc9…` |
| `dirh10.restricted.coordinates_relations_v2` | `70167c443b5c988f…` | `326bb0a8e7e08876…` |
| `a17.annotated.coordinates_v2` | `c25f7d680ff42815…` | `2882be9a673cbeea…` |
| `a17.restricted.coordinates_v2` | `fd9dd8373f99678a…` | `623da5b55ab29f6c…` |
| `dirh10.annotated.coordinates_v2` | `c259fc988ab1276d…` | `671af6a68b89eead…` |
| `dirh10.restricted.coordinates_v2` | `182d96666e66afd9…` | `aafff6b3cc618f73…` |
