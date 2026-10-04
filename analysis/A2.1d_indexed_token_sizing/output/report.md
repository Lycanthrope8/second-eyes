# A2.1d indexed encoding: token audit results

Verification: 64 of 64 checks passed. Tokenizers: Hugging Face Qwen2TokenizerFast from quest-app/Assets/SecondEyes/Models/qwen2.5-0.5b-instruct; llama.cpp b11277 (eae11d22) se_tokenize with tiny-qwen2-tokenizer.gguf.

## Calibration against A1's records

| A1 prompt | Recorded | Measured (each tokenizer) | Scene recorded | Scene measured |
|---|---|---|---|---|
| a17-fixed | 230 | 230, 230 | 191 | 191, 191 |
| a17-front | 232 | 232, 232 | 191 | 191, 191 |
| a17-left | 230 | 230, 230 | 191 | 191, 191 |
| a17-table | 227 | 227, 227 | 191 | 191, 191 |

## Old against new

| New input | Old input | Old tokens | New tokens | Change | Old after prefix | New after prefix | Old bytes | New bytes |
|---|---|---|---|---|---|---|---|---|
| a17.annotated.coordinates_v2 | a17.annotated.coordinates_v1 | 1218 | 1343 | +125 | 73 | 73 | 4309 | 4846 |
| a17.annotated.coordinates_relations_v2 | a17.annotated.coordinates_relations_v1 | 3209 | 2266 | -943 | 526 | 220 | 8789 | 7081 |
| a17.restricted.coordinates_v2 | a17.restricted.coordinates_v1 | 1147 | 1272 | +125 | 73 | 73 | 4249 | 4786 |
| a17.restricted.coordinates_relations_v2 | a17.restricted.coordinates_relations_v1 | 2659 | 2106 | -553 | 526 | 220 | 7876 | 6777 |
| dirh10.annotated.coordinates_v2 | dirh10.annotated.coordinates_v1 | 1318 | 1443 | +125 | 58 | 58 | 4572 | 5109 |
| dirh10.annotated.coordinates_relations_v2 | dirh10.annotated.coordinates_relations_v1 | 5849 | 2998 | -2851 | 1085 | 290 | 13332 | 8304 |
| dirh10.restricted.coordinates_v2 | dirh10.restricted.coordinates_v1 | 1288 | 1413 | +125 | 58 | 58 | 4532 | 5069 |
| dirh10.restricted.coordinates_relations_v2 | dirh10.restricted.coordinates_relations_v1 | 8401 | 3625 | -4776 | 1085 | 290 | 17868 | 8913 |

## Per-block marginal tokens, old -> new

| Block | a17.annotated.coordinates_v2 | a17.annotated.coordinates_relations_v2 | a17.restricted.coordinates_v2 | a17.restricted.coordinates_relations_v2 | dirh10.annotated.coordinates_v2 | dirh10.annotated.coordinates_relations_v2 | dirh10.restricted.coordinates_v2 | dirh10.restricted.coordinates_relations_v2 |
|---|---|---|---|---|---|---|---|---|
| wrapper_head | 49 -> 49 | 49 -> 49 | 49 -> 49 | 49 -> 49 | 49 -> 49 | 49 -> 49 | 49 -> 49 | 49 -> 49 |
| header | 137 -> 262 | 138 -> 263 | 137 -> 262 | 138 -> 263 | 137 -> 262 | 138 -> 263 | 137 -> 262 | 138 -> 263 |
| semantics | 613 -> 613 | 613 -> 613 | 613 -> 613 | 613 -> 613 | 613 -> 613 | 613 -> 613 | 613 -> 613 | 613 -> 613 |
| objects | 346 -> 346 | 346 -> 346 | 275 -> 275 | 275 -> 275 | 461 -> 461 | 461 -> 461 | 431 -> 431 | 431 -> 431 |
| pose | 57 -> 57 | 57 -> 57 | 57 -> 57 | 57 -> 57 | 38 -> 38 | 38 -> 38 | 38 -> 38 | 38 -> 38 |
| command | 11 -> 11 | 11 -> 11 | 11 -> 11 | 11 -> 11 | 15 -> 15 | 15 -> 15 | 15 -> 15 | 15 -> 15 |
| wrapper_tail | 5 -> 5 | 5 -> 5 | 5 -> 5 | 5 -> 5 | 5 -> 5 | 5 -> 5 | 5 -> 5 | 5 -> 5 |
| near |  | 111 -> 34 |  | 37 -> 22 |  | 170 -> 45 |  | 37 -> 22 |
| above |  | 182 -> 48 |  | 37 -> 22 |  | 37 -> 22 |  | 37 -> 22 |
| below |  | 182 -> 48 |  | 37 -> 22 |  | 37 -> 22 |  | 37 -> 22 |
| on |  | 74 -> 48 |  | 97 -> 62 |  | 37 -> 22 |  | 37 -> 22 |
| inside |  | 74 -> 48 |  | 157 -> 62 |  | 255 -> 69 |  | 37 -> 22 |
| between |  | 130 -> 163 |  | 129 -> 162 |  | 1714 -> 509 |  | 4647 -> 1236 |
| intrinsic.right |  | 137 -> 50 |  | 39 -> 24 |  | 39 -> 24 |  | 39 -> 24 |
| intrinsic.in_front_of |  | 163 -> 52 |  | 41 -> 26 |  | 41 -> 26 |  | 41 -> 26 |
| center_distance_m |  | 484 -> 284 |  | 484 -> 284 |  | 1173 -> 583 |  | 1173 -> 583 |
| user_heading.right |  | 44 -> 23 |  | 44 -> 23 |  | 62 -> 26 |  | 62 -> 26 |
| user_heading.in_front_of |  | 39 -> 24 |  | 39 -> 24 |  | 65 -> 28 |  | 65 -> 28 |
| user_to_anchor.right |  | 220 -> 48 |  | 220 -> 48 |  | 341 -> 88 |  | 341 -> 88 |
| user_to_anchor.in_front_of |  | 150 -> 52 |  | 150 -> 52 |  | 559 -> 90 |  | 559 -> 90 |

## New inputs

| Input | Bytes | Tokens | Document alone | Static lines | Dynamic lines | After reusable prefix | Block conditional maps (overlapping) |
|---|---|---|---|---|---|---|---|
| a17.annotated.coordinates_v2 | 4846 | 1343 | 1289 | 1221 | 68 | 73 | 0 |
| a17.annotated.coordinates_relations_v2 | 7081 | 2266 | 2212 | 1997 | 215 | 220 | 52 |
| a17.restricted.coordinates_v2 | 4786 | 1272 | 1218 | 1150 | 68 | 73 | 0 |
| a17.restricted.coordinates_relations_v2 | 6777 | 2106 | 2052 | 1837 | 215 | 220 | 132 |
| dirh10.annotated.coordinates_v2 | 5109 | 1443 | 1389 | 1336 | 53 | 58 | 0 |
| dirh10.annotated.coordinates_relations_v2 | 8304 | 2998 | 2944 | 2659 | 285 | 290 | 52 |
| dirh10.restricted.coordinates_v2 | 5069 | 1413 | 1359 | 1306 | 53 | 58 | 0 |
| dirh10.restricted.coordinates_relations_v2 | 8913 | 3625 | 3571 | 3286 | 285 | 290 | 762 |

Parity with llama.cpp b11277 (eae11d22) se_tokenize with tiny-qwen2-tokenizer.gguf: 24 of 24 texts identical.

Skipped: nothing.
