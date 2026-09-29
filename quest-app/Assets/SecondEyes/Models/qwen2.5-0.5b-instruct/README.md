# qwen2.5-0.5b-instruct

Files for Meta's on-device chat provider, written by `grounding/copy_to_unity.py` (A1.7c). Don't edit them by hand; run the tool again (`docs/setup/quest-model.md`).

- `vocab.json`, `merges.txt`, `tokenizer_config.json`: the tokenizer of Qwen/Qwen2.5-0.5B-Instruct at commit `7ae557604adf67be50417f59c2c2f167def9a775`, byte for byte (`.gitattributes`).
- `LICENSE.txt`: that model's licence, which covers these files.
- `model.json`: which `.sentis` file the provider loads, and the fingerprints that Second Eyes → Fill chat provider checks.
- The provider asset in this folder is filled by Second Eyes → Fill chat provider.

The model itself, `qwen2.5-0.5b-instruct-72303ef1-f16.sentis`, is made by Second Eyes → Convert model in `Assets/StreamingAssets/`, which Git ignores (D34, D35, D38).
