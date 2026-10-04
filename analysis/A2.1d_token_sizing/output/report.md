# A2.1d token-sizing audit: results

Tokenizers: Hugging Face Qwen2TokenizerFast from quest-app/Assets/SecondEyes/Models/qwen2.5-0.5b-instruct; llama.cpp b11277 (eae11d22) se_tokenize with tiny-qwen2-tokenizer.gguf. Configurations: relations.v1@d922fe902669, directions.v1@73590e939d4f.

## Calibration against A1's records

| A1 prompt | Recorded | Measured (each tokenizer) | Scene recorded | Scene measured |
|---|---|---|---|---|
| a17-fixed | 230 | 230, 230 | 191 | 191, 191 |
| a17-front | 232 | 232, 232 | 191 | 191, 191 |
| a17-left | 230 | 230, 230 | 191 | 191, 191 |
| a17-table | 227 | 227, 227 | 191 | 191, 191 |

## Parity

- llama.cpp b11277 (eae11d22) se_tokenize with tiny-qwen2-tokenizer.gguf: 51 of 53 texts give identical token IDs: differences in representative[22], representative[24]

## Whole inputs

| Input | Bytes | Tokens | Document alone | Static lines | Dynamic lines | After reusable prefix | Cold s* | Warm best-case s* | Work |
|---|---|---|---|---|---|---|---|---|---|
| a17.annotated.coordinates_v1 | 4309 | 1218 | 1164 | 1096 | 68 | 73 | 7.4 | 0.4 | 0 |
| a17.annotated.coordinates_relations_v1 | 8789 | 3209 | 3155 | 2634 | 521 | 526 | 19.4 | 3.2 | 342 |
| a17.restricted.coordinates_v1 | 4249 | 1147 | 1093 | 1025 | 68 | 73 | 6.9 | 0.4 | 0 |
| a17.restricted.coordinates_relations_v1 | 7876 | 2659 | 2605 | 2084 | 521 | 526 | 16.1 | 3.2 | 342 |
| dirh10.annotated.coordinates_v1 | 4572 | 1318 | 1264 | 1211 | 53 | 58 | 8.0 | 0.4 | 0 |
| dirh10.annotated.coordinates_relations_v1 | 13332 | 5849 | 5795 | 4715 | 1080 | 1085 | 35.3 | 6.6 | 1190 |
| dirh10.restricted.coordinates_v1 | 4532 | 1288 | 1234 | 1181 | 53 | 58 | 7.8 | 0.4 | 0 |
| dirh10.restricted.coordinates_relations_v1 | 17868 | 8401 | 8347 | 7267 | 1080 | 1085 | 50.8 | 6.6 | 1190 |

*linear extrapolation from A1, NOT measured Quest latency: 165.5 tokens/s (A1.8b run 20260930_A1_r027: median uncached whole-prompt time of the four A1 prompts (227-232 tokens) through se_llama_cli from adb shell, Q8_0, 2 threads, plugged in; llama-bench pp230 at 2 threads gave 170 tokens/s).

## Marginal tokens by block

| Block | a17.annotated.coordinates_v1 | a17.annotated.coordinates_relations_v1 | a17.restricted.coordinates_v1 | a17.restricted.coordinates_relations_v1 | dirh10.annotated.coordinates_v1 | dirh10.annotated.coordinates_relations_v1 | dirh10.restricted.coordinates_v1 | dirh10.restricted.coordinates_relations_v1 |
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

## Conditional membership (diagnostic differences; the inputs keep these fields)

| Input | Block conditional maps | Object and pose conditional_fields |
|---|---|---|
| a17.annotated.coordinates_v1 | 0 tokens, 0 B | 17 tokens, 42 B |
| a17.annotated.coordinates_relations_v1 | 138 tokens, 624 B | 17 tokens, 42 B |
| a17.restricted.coordinates_v1 | 0 tokens, 0 B | 32 tokens, 82 B |
| a17.restricted.coordinates_relations_v1 | 311 tokens, 948 B | 32 tokens, 82 B |
| dirh10.annotated.coordinates_v1 | 0 tokens, 0 B | 25 tokens, 54 B |
| dirh10.annotated.coordinates_relations_v1 | 135 tokens, 624 B | 25 tokens, 54 B |
| dirh10.restricted.coordinates_v1 | 0 tokens, 0 B | 55 tokens, 134 B |
| dirh10.restricted.coordinates_relations_v1 | 1574 tokens, 3178 B | 55 tokens, 134 B |

Self-checks: all passed. Skipped: nothing.
