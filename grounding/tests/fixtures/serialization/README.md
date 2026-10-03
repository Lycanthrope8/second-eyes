# Serializer test fixtures

Test data for `grounding/tests/test_serialization.py` (A2.1d, D69). Not training data.

`golden/` holds the expected documents. They were fixed before the serializer was written, and the serializer never
generates them:
- **Eight documents** copied byte for byte from the accepted indexed audit (`analysis/A2.1d_indexed_token_sizing/output/inputs/`).
  Their SHA-256 values are checked against that audit's `results.json`.
- **Two literal one-object documents** from Appendix B of `Second_Eyes_A2_1d_Serializer_Implementation_Brief.md`:
  `obj_001` of the annotated a17 scene with command `fx.a17.c001`.

`expected.json` lists each golden with its hash, its source, and the inputs that render it:

| Golden | Scene | Command | Objects |
|---|---|---|---|
| a17, annotated / restricted | `contract/valid/scene.a17.{annotated,restricted}.json` | `contract/valid/command.a17.c001.json` | all six |
| dirh10, annotated / restricted | `directions/scene.h.{annotated,restricted}.json` | `directions/command.h.base.json` | the first ten IDs in lexicographic order: a test-only selection, not a cropping policy |
| one_object.a17 | `contract/valid/scene.a17.annotated.json` | `contract/valid/command.a17.c001.json` | `obj_001` only |

Selections are made on copies; the original records are never changed.

The other fixtures:
- `appendix_a.json`: the brief's literal headers and initial semantics line.
- `codec_cases.json`: the indexed audit brief's literal codec examples with their stated mappings.

`.gitattributes` keeps every file here byte-exact: the goldens end in LF, and Git must not convert them.
