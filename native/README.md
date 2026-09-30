# native: llama.cpp for Second Eyes (A1.8b, D55)

`libse_llama` is a small C interface over llama.cpp, so the test program and later the Unity app only see plain ints,
floats and strings, never llama.cpp's structs, which change between releases. llama.cpp is pinned in
`llama.cpp.pin` (release and commit); `tools/build_llama.py` fetches exactly that into `third_party/llama.cpp` and
builds from it.

| File | What it is |
|---|---|
| `se_llama.h`, `se_llama.cpp` | the interface: load a GGUF model (memory-mapped), tokenize, evaluate after keeping part of the cache (a cached scene), the last position's log-probabilities, and scoring continuations, which restores the cache |
| `se_llama_cli.cpp` | A1.8b's test program: runs a job from `grounding/llama_headset.py` on the headset through `adb shell` and writes every token, answer, score, timing and the peak memory to a JSON file |
| `CMakeLists.txt` | builds llama.cpp as static libraries inside `libse_llama`, with the options of llama.cpp's `docs/android.md` |
| `tests/make_tiny_model.py` | a two-layer GGUF with random weights and Qwen's real tokenizer, for tests without the real model |
| `tests/test_se_llama.py` | tests of the interface through ctypes: tokenization against the PC references, the cache, scoring, errors |

Build: `python tools/build_llama.py android` for the headset (`native/out/android-arm64/`), or `host` for this PC.
Test: `python native/tests/make_tiny_model.py third_party/llama.cpp tiny.gguf`, then
`python native/tests/test_se_llama.py native/out/host/libse_llama.so tiny.gguf runs/<a reference run>/raw`.
