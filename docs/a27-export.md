# A2.7: export and desktop verification of the deployment candidate

A2.7's candidate is the fine-tuned parser with the accepted resolver, `a24.parser.s1` at epoch 2 (D115). Its plan
and frozen acceptance criteria are in `notes/phases/A2.7_plan.md`. Delivery 1 is the desktop part, four commands run
on two machines:

| Command | Machine | What it does |
|---|---|---|
| `python -m grounding.a27 export` | RTX PC (export environment) | The merged model to GGUF Q8_0, with llama.cpp's own converter at the pinned release |
| `python -m grounding.a27 bundle` | RTX PC (`.venv`) | The prompts, the float32 reference and the frozen headset subset |
| `python -m grounding.a27 desktop` | Laptop (host build) | The mask checks and the 2,715 decodes through `libse_llama` |
| `python -m grounding.a27 compare` | RTX PC (`.venv`) | The verdicts of criteria 4 and 7 (desktop halves) |

## export

`--merged` A2.4's merged folder of the candidate, `--llama` the llama.cpp checkout (`third_party\llama.cpp`, from
`python tools/build_llama.py fetch`), `--out` a new folder. It refuses anything but the candidate (model name, the
parser design, epoch 2), a merged file whose hash changed since the merge, and a checkout not at
`native/llama.cpp.pin`'s commit. The converter runs in the current Python, which needs torch, transformers, numpy and
sentencepiece: the pinned converter tries a SentencePiece tokenizer before Qwen's BPE one and fails on the import
without it. Writes `a24.parser.s1.q8_0.gguf`, `convert.log` and `export.json` (the identity, the command, the software
versions, the GGUF's hash, and its metadata read back by llama.cpp's gguf-py, with a hash of every tokenizer field).

## bundle

`--prep` and `--work` the A2.6 preparation and work folders, `--evaluation` A2.4's evaluation of the candidate
(`eval\parser-s1-parser`, the float32 reference), `--merged` the candidate's merged folder, `--out` a new folder. It
checks that the evaluation was of this model, on this preparation, with the current language and parser prompt, and
unchanged since. It rebuilds each prompt with A2.4's own functions, and every prompt's length must equal the one A2.4
recorded. It freezes the 300-command headset subset and its 20-command mask sample (the plan gives the rule). Writes
`requests.jsonl`, `grammar.gbnf` and `manifest.json`.

## desktop

`--bundle`, `--export`, `--base-gguf` (the accepted A2.5 base GGUF, `dd753cd6…`), `--lib` the host build
(`native\out\host\se_llama.dll`), `--llama` the checkout (for gguf-py), `--out` a new folder, `--workers` mask
processes, `--threads` decoding threads, `--resume` to continue an interrupted run under the same inputs. It refuses a
GGUF other than the exported one, and an export whose tokenizer differs in any field from the base GGUF's. Then:
- **masks:** at every distinct prefix of the reference decodes (4,332) and at the 21 frozen prefixes, the tokens
  llama.cpp's grammar allows must equal the Python decoder's (one full-vocabulary application each, so they run in
  parallel processes);
- **decodes:** each of the 2,715 prompts decoded once, from a fresh cache, written as it goes.

Writes `masks.json`, `results.jsonl` and `manifest.json`. It judges nothing.

## compare

`--bundle`, `--desktop`, `--prep`, `--work`, `--out`. Every Q8_0 output is resolved through A2.4's own path. Where an
output equals the reference, its two views must reproduce A2.4's recorded outcomes exactly, or the comparison stops.
Verdicts:
- **criterion 4 (desktop):** no prefix with differing allowed tokens;
- **criterion 7 (desktop):** at least 99% of the 2,715 decoded to exactly the reference's tokens and status, every
  difference listed with both outputs and both views' outcomes, technical failures apart.

Writes `compare.json` and `report.md`.

## The native functions

`native/se_llama.h` adds `se_grammar_new`, `se_grammar_free`, `se_grammar_reset`, `se_grammar_allowed` (the full
allowed set, for verification), `se_grammar_pick` (the greedy choice among the allowed tokens), `se_grammar_accept` and
`se_decode_constrained` (one whole request: a fresh cache, the prompt with keep 0, then decoding within the request).
`native/tests/test_se_grammar.py` checks them against the Python decoder on Qwen's real vocabulary;
`grounding/tests/test_a27.py --lib --gguf` runs the whole desktop flow on the library.

## Time

`bundle` and `compare` load A2.6's per-scene bundles through A2.4's loader, which first verifies every prepared file
against its bundle's manifest: about 45 minutes each on the RTX PC, before any output appears.
