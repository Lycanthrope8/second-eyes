# Association guard fixtures (A4)

Shared by `grounding/tests/test_resolution_v2.py` and `tools/csharp-tests/Resolution` (mode `association`). The scene is
`contract/valid/scene.bedroom.perceived_v3.json` with its map `contract/valid/category-map.bedroom.json`;
`command.bedroom.json` is a command for it, and the three queries are parser v1's readings `chair | closest 1 | table`,
`chair | farthest 1 | table` and `chair | closest 2 | table`, built by `language.query` (format 2, since the scene is
perceived). `expected.json` gives each answer before and after the association guard, worked out by hand from the
chairs' centres; the values are illustrative test values, not measurements.
