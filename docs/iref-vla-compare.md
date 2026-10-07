# Zero-shot three-system comparison (A2.3d, D95)

Rules, Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct on 256 more parent commands of the same inspected room, in both
inventory views and both formats: a broader development screen, not an unseen-room evaluation. The A2.3a pilot's 32
parents stay out; the reserved lab commands stay closed. Package `grounding/inference/iref_vla_compare/`; policy
`compare-policy.v1.json` (`iref.compare.zero_shot.v1`).

## Models

- `grounding/models/qwen2.5-0.5b-instruct.json`: A2.3a's accepted model, revision `7ae55760…`.
- `grounding/models/qwen2.5-7b-instruct.json`: the larger reference, revision `a09a35458c702b33eeacc393d103063234e8bc28`,
  Apache-2.0, four bfloat16 shards with published SHA-256 (loaded as float32, about 30.5 GB). It is an empirical
  reference, not an assumed upper bound.
- `python -m grounding.models.acquire --description <json> --out <folder> [--only tokenizer]` downloads at the pinned
  revision and proves every file (size, the shards' SHA-256, and the commit and ETag of the download metadata);
  `acquisition.json` records the hashes. The 7B's `vocab.json`, `merges.txt` and `tokenizer_config.json` are
  byte-identical to the 0.5B's pinned files; it also has `tokenizer.json` (`c0382117…`). Both tokenizers are still
  loaded and checked separately.

## Selection and the frozen answer-code policy

- Population: A2.3a's eligibility (both views materialized, all four documents rendered): 1,004 parents. A2.3a's 32
  pilot parents are recomputed from its own salt and count and checked against their pinned hash (`8c826ba3…`), then
  excluded: 972 remain.
- Selection: sort by SHA-256 of UTF-8(`second-eyes/a23d/compare/v1` + LF + parent ID), ties by ID; the first 256.
  1,024 requests per model: each parent in both views and both formats.
- Per parent and view, one letter assignment (stream `second-eyes/a23d/codes/v1`) and one list order (stream
  `second-eyes/a23d/list/v1`), each a sort by SHA-256 of UTF-8(salt + LF + parent + LF + view + LF + object ID). The
  inputs are only the parent, the view and the object IDs, so both formats and both models see the same mapping. K stays
  last; the scene document and its object order are unchanged. No grid, seed search, voting or averaging.

## Commands (laptop)

```text
python -m grounding.inference.iref_vla_compare prepare --bundle DIR --tokenizer-small DIR --tokenizer-large DIR --out NEW_DIR
python -m grounding.inference.iref_vla_compare rules --requests DIR --bundle DIR --relation-config FILE --direction-config FILE --out NEW_DIR
```

`prepare` writes `selection.json` (the population, the excluded pilot list, the literal selected list and its hash),
`request-index.jsonl` (identities, objects, letters, list order, the mapping with token IDs, document and prompt
hashes, and per model the token count, token hash and context status), `prompts.jsonl`, `tokens/<model>.jsonl`,
copies of the policy and A2.3a's protocol, and a manifest; then it reads everything back. Each model's tokenizer
checks every offered letter's one-token boundary; requests over 8,192 tokens are kept and marked per model, never
truncated. `rules` runs the unchanged A2.2b parser and A2.1e resolver once per parent and view on the same derived scene
and command the models are offered (A2.3b's join checks them), with no annotation read.

## Commands (RTX PC)

```text
python -m grounding.inference.iref_vla_compare smoke --requests DIR --model KEY --model-dir DIR --tokenizer-dir DIR --device cuda --out FILE
python -m grounding.inference.iref_vla_compare run --requests DIR --model KEY --model-dir DIR --tokenizer-dir DIR --device cuda --smoke FILE --out NEW_DIR [--resume]
```

`KEY` is `qwen2.5-0.5b-instruct` or `qwen2.5-7b-instruct`. Both commands first verify the request bundle, re-tokenize every
prompt with the model's own pinned tokenizer and compare it with the frozen token IDs, recheck every offered letter's
boundary, and tie the checkpoint's files to the pinned revision. Both use A2.3a's unchanged model path: float32, eager
attention, evaluation mode, batch size one, one final-position forward with no cache, no generation; offered-letter
scoring with exact ties to K.

`smoke` loads the model, runs two canaries (the first request and the longest; production forward against the
independent last-hidden-state path, A2.3a's tolerance) and the three longest requests, and writes the settings record:
precision, attention, device, library versions, load time, timings and peak GPU memory. `run` starts only from an
accepted smoke record that names the same model, requests, checkpoint and code, and stops if the loaded model's
settings differ from it. It runs the canaries again, then every request in canonical order, appending one row per
request to `<out>.partial/rows.jsonl`. After an interruption, `--resume` continues only if every frozen hash agrees (a torn
last line is dropped). Requests over the ceiling are kept as `context_budget_exceeded` and never sent; a failed forward is
kept as `execution_failed`; three consecutive failures stop the run, resumable. On completion the run writes
`results.jsonl`, `sessions.jsonl`, `canaries.json`, `summary.json`, `report.md` and a manifest, reads them back and removes
the partial folder. Exit 1 marks a published run with exclusions or failures. All timings are PC measurements.

## Scoring (laptop)

```text
python -m grounding.inference.iref_vla_compare score --requests DIR --rules DIR --small-run DIR --large-run DIR --bundle DIR --annotations FILE --out NEW_DIR
```

Every input is read back first; the annotation file must be the pinned IRef-VLA file (`06cfdfb4…`), read through
A2.3b's loader: one mapped target and one relation label per command, present in both views and offered in every
request. Annotations are read here and nowhere else. Per request and system the outcome is correct (the chosen object is
the published target), wrong object, ASK, over the ceiling or execution failure; an ASK on a uniquely annotated command
is not correct. The summary, per view (the full inventory first, as the primary comparison) and format, gives correct
over planned and over object choices; the rules' agreement and outcomes; the sample's actual always-B and
always-second-position rates; paired tables (0.5B against 7B, each model against the rules, and coordinates against
augmented per model, on common completed requests, with their coverage); breakdowns by source relation and by candidate
count; and a failure sample (six per model and view, by salted hash, for reading only). The policy is
`compare-scoring.v1.json`. Ratios keep numerator and denominator; no float sums enter the summary, so its readback does
not depend on the Python version.

## A2.3e checks (D96)

```text
python -m grounding.inference.iref_vla_compare audit --requests DIR --bundle DIR --tokenizer-small DIR --tokenizer-large DIR --out NEW_DIR
python -m grounding.inference.iref_vla_compare costs --requests DIR --bundle DIR --relation-config FILE --direction-config FILE --tokenizer-small DIR --tokenizer-large DIR --small-run DIR --large-run DIR --out NEW_DIR
python -m grounding.inference.iref_vla_compare cache --requests DIR --model-dir DIR --tokenizer-dir DIR --device cuda --out NEW_DIR
```

- **`audit`** (laptop) exports and checks the canary parents' eight requests: exact prompt bytes, both tokenizers' token
  IDs, role boundaries in characters and tokens, choices, document hash, command text, leak and scored-position
  checks. Exit 1 if any check fails.
- **`costs`** (laptop) reports token counts and paired format differences from the requests, and uncached forward
  timings, load times and peak memory from the accepted runs. It then times, for 16 parent/view cases (both formats)
  chosen by SHA-256 of UTF-8(`second-eyes/a23e/cases/v1` + LF + parent + LF + view), one warm-up and five repetitions
  of each stage separately: relation construction (the serializer's relation-table builders, wrapped from outside), the
  rest of serialization, prompt assembly, and tokenization per tokenizer. Disk reads and record validation are timed
  apart. Every rebuilt document, prompt and token list must equal the frozen one, or nothing is reported.
- **`cache`** (RTX PC) runs the 0.5B on the same cases. It compares the uncached forward with cached scoring under
  three kinds of reuse, each of an exactly matching token prefix only:
  - a repeated identical request (all tokens but the last);
  - another command: the other format of the same parent and view, and another parent;
  - an edited copy: scene evidence, pose or choices changed, which may reuse only the tokens before the edit.

  A different model or tokenizer identity gets no reuse. Every comparison applies D56 (the same best candidate, the
  same order among candidates with at least 1% restricted share, total variation distance at most 0.05). A pass
  preserves the model's output, wrong choices included. Exit 1 if any check fails.

All timings are PC measurements, not Quest figures.
