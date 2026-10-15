# A2.7: the parser path on the headset (Delivery 2, D116)

The deployed grounding path (D115): the fine-tuned parser `a24.parser.s1` (epoch 2) decodes the typed command under
the v1 grammar; the v1 reader turns the string into an interpretation; A2.4's checker resolves it against the scene
with the accepted resolver. Delivery 2 puts all of it on the headset, in C#, checked against the Python reference at
every layer. Delivery 3 runs the headset acceptance and the measurements.

## The C# port (`quest-app/Assets/SecondEyes/Grounding/`)

Every file below is free of Unity, so the .NET projects in `tools/csharp-tests` compile and test it on a PC.

| Folder or file | Ports | Checked by (Resolution project, mode) |
|---|---|---|
| `Resolution/PyCompat.cs` | CPython 3.12's `min`, `max`, `sum` (Neumaier), `degrees`, code-point ordering | all modes |
| `Resolution/Geometry.cs`, `Relations.cs` | `grounding/relations`: geometry, the predicates and ranking, three-valued, with the conditional-evidence flag | `relations`: every value bit for bit, boundary scenes on both sides of every band |
| `Resolution/SchemaCheck.cs` | JSON Schema 2020-12 for exactly the keywords of the eight shipped schemas; any other keyword is refused at load | `schemas`: fuzzed records, every error's path and keyword |
| `Resolution/ContractCheck.cs` | `grounding/contract`: record type, schema, semantic and cross-record rules | `validation` |
| `Resolution/ResolverValidation.cs` | the resolver's own checks and the query's normalization | `validation`: issues, normalized form as text |
| `Resolution/ResolverCore.cs` | `grounding/resolution/resolve.py`: budgets, the evaluation, the result and its invariants | `resolution`: the whole run record, work counters included |
| `Resolution/CheckerCore.cs` | A2.4's checker: query construction, `invalid_label_query`, the resolver configuration, the outcome | `resolution`, `acceptance` |
| `Parsing/Language.cs` | the v1 reader (`parse_output`), `render`, `interpretation` | `reader`, `acceptance` |
| `Parsing/ParserCore.cs` | the parser pipeline, strict record parsing, outcome kinds | `pipeline` (a fake native side) |
| `LlamaParser.cs` | the native calls: tokens, `se_decode_constrained`, token bytes, allowed-set hashes | `native` (the host library, through P/Invoke) |
| `Parsing/ParserFolder.cs`, `ParserAcceptance.cs`, `ParserBridge.cs`, `PyJson.cs` | the pushed folder, the headset acceptance run, the session bridge, canonical JSON | `parser-check`, the Python suite |
| `Parsing/Glue/ParserCheckRunner.cs` | the panel's "Parser check (A2.7)" (Unity glue) | `UnityGlue`, `PanelPartial` (type checks) |

**Left out, by decision (D116):** directions (parser v1 has no frames: left, right, front, behind and
viewpoint-dependent commands remain unsupported, a future integration item; a constraint with a frame is a technical
failure); detailed provenance (evidence records and assumption lists stay offline; the known and unknown states, the
evidence rules, the unresolved-outcome rules, the conditional flag and the reason codes are kept); per-request
configuration hashes (the parser folder's manifest identifies the configuration).

## Outcome kinds

| Parser kind | When | A2.5 status | Shown on the panel |
|---|---|---|---|
| `selected` | resolved | `completed`, with the target | target `obj_...` |
| `ask` | ambiguous, no match, insufficient information | `ask` | ASK (the outcome) |
| `unsupported` | a label outside the vocabulary | `refused` | refused (unsupported): unsupported |
| `invalid` | invalid records, an invalid label query | `refused` | refused (invalid): the outcome |
| `budget` | a resolver limit | `refused` | refused (budget): budget_exceeded |
| `failed` | a technical failure: native, decoding that didn't complete, output outside the language, a resolver fault | `failed` | failed (failed): the cause |

The outcome record keeps A2.5's fields; on the parser path it adds `path: "parser"` and a `parser` block (the kind,
the resolver's outcome, the decoded text and tokens, the candidates, the reason code, the conditional flag, the parser
folder's manifest hash, decode and resolve times). Direct-path records are unchanged.

## The parser folder (files/parser on the headset)

Built on the RTX PC: `python -m grounding.a27 headset --bundle --desktop --export --prep --work --maps-from A22D
--maps-from WORK --out FOLDER [--port-out PORT]`. Pushed like A2.5's goldens, the manifest last. The app verifies every
file by SHA-256 at load and logs the manifest's hash with every outcome and run.

- `assets/`: the eight schemas, `relations.v1.json`, `directions.v1.json`, `resolver-protocol.json`, `language.v1.json`,
  `settings.v1.json`, `grammar.gbnf`, `parser.json` (end token, new tokens, context 1024, the GGUF's name and hash);
- `maps/`: every category map of A2.2d's bundle (the interactive snapshots) and A2.6's work (the acceptance scenes);
- `startup-check.json`: fixed parser prompts with their pinned token IDs;
- `acceptance/`: Delivery 3's material: the frozen 300-command subset's decodes, the 21 frozen prefixes and every step
  of the 20-command mask sample, the subset's 600 resolutions, and the readers: every distinct reference output of the
  whole bundle (1,622) and the 97-string corpus, 1,719 in all (D117). Every resolution and valid reader carries its
  expected grounding query (A2.4's `language.query`, canonical JSON); the readers' queries use one recorded context,
  `reader-context.json` (the first resolvable entry's scene and command).

`--port-out` also writes the full acceptance export (all 2,715 parents in both views, 5,430 resolutions) for the PC's
C# check, from the same load of A2.4's parent list (about 45 minutes on the RTX PC).

**Delivery 2's corrections (D117)** apply to existing folders without that load: `python -m grounding.a27 headset-amend
--parser OLD --port OLD_PORT --out-parser NEW --out-port NEW_PORT` adds the full readers, the queries and the reader
context, carries the decodes, masks, maps, assets, startup check and 600 resolutions over byte for byte, and records
what it amended in the manifest.

## On the headset

- **Session** (the panel's session row): with the parser folder present, the session first verifies the model file's
  SHA-256 against the exported candidate (D117; a mismatch refuses the session), then loads it with a
  1,024-token context, runs A2.5's startup guard and then the parser's startup check (each fixed prompt, built and
  tokenized natively, must equal the PC's IDs), and grounds every operational request through the parser. Without the
  folder, A2.5's direct path, for diagnostics. `session.json` records `grounding_path`, the folder's manifest hash, the candidate and
  the verified model hash.
- **Parser check**, on the panel's More page (`quest-panel.md`; labelled Parser check (A2.7) before A4 Delivery 1): verifies the folder and the model file's SHA-256,
  loads it, runs the startup check and, only if it passes (D117), the acceptance, and writes `files/parser-results/STAMP/` (decodes, masks, resolutions and readers as JSONL, `run.json`
  with the identity and memory, `done.json` last). The PC judges: `python -m grounding.a27 headset-judge --folder
  FOLDER --results RESULTS --out JUDGED`.

## The commands

| Command | Machine | What |
|---|---|---|
| `python -m grounding.a27 goldens --samples --official --out [--lib --gguf]` | laptop | every golden file the Resolution modes compare with, from A2.4's fixture |
| `python -m grounding.a27 headset ... [--port-out]` | RTX PC | the parser folder, and the full acceptance export |
| `python -m grounding.a27 port-goldens --bundle --prep --work --out` | RTX PC | the full acceptance export alone |
| `python -m grounding.a27 headset-amend --parser --port --out-parser --out-port` | laptop | Delivery 2's corrections on the existing folders (D117) |
| `python -m grounding.a27 headset-judge --folder --results --out` | any | Delivery 3's verdicts, criteria 1, 2, 4 and 7 (headset halves) |

**The judge verifies the evidence before judging (D117):** the folder, file by file against its manifest, with every
count; the results naming this folder and the exported candidate's model, after a passed startup check, with counts
consistent with the run's record; one result per expected case (a duplicate or an unexpected case refuses the evidence,
a missing one leaves its criterion not met). Any failed prerequisite leaves every criterion not met. A prompt whose
native tokens differ fails criterion 7; a resolution or reader whose constructed query differs fails criterion 1 or 2.

## Delivery 3: on the headset

| Command | What |
|---|---|
| `python -m grounding.a27 parser-push --folder FOLDER --run RUN --model GGUF` | the parser folder (the manifest last, the old one removed first) and the model, checked here by SHA-256 and on the headset by size; a receipt in the run |
| `python -m grounding.quest runtime-identity --out DIR --policy grounding/a27/runtime-identity-parser.v1.json` | criterion 6: the deployed library against its Android build record (with every function the header declares) and the parser model against the exported candidate, with read-only adb |
| `python -m grounding.a27 parser-pull --run RUN [--results STAMP]` | the newest finished Parser check (with `done.json`) into `runs/RUN/raw/parser/` |
| `python -m grounding.a27 headset-judge --folder FOLDER --results RESULTS --out JUDGED` | criteria 1, 2, 4 and 7, headset halves |
| `python -m grounding.quest outbox-pull --run RUN --sessions N` | A2.5's pull of the newest N sessions |
| `python -m grounding.a27 measure-sessions --run RUN --goldens DIR` | A2.5's `measure-report` for every session of that pull (each copied to `raw/outbox-sessions/`), then the summary across sessions |
| `python -m grounding.a27 measure-summary --run RUN` | the summary alone, over the run's parser reports |

**The measurements** (D115: measured, not judged) cover the time from submission to a validated target or a decline,
on the headset clock. A selection, an ask, or an unsupported, invalid or budget refusal counts as answered; a technical
failure is counted apart and never timed. Cold is each session's first such request, warm the rest; prompt and generated
tokens; decode and resolve times; memory as the runtime reports it; frames and heat from A1's `tools/profile.py` with
OVR Metrics' CSV. The commands come from the panel's Preset button (the scene's next golden dataset command), so the
headset stays on for the whole recording (open item O14). A2.5's `measure-report` adds a parser section to a parser
session's report: generated tokens, outcome kinds, cold and warm. Its stage names keep A2.5's columns: `eval` is the
constrained decode, `score` the resolution, `build` the strict parsing of the records.
