# llama.cpp on the headset, from the command line (A1.8b, D55)

The model runs in a test program started over `adb shell`, without Unity, to learn what llama.cpp can do on the
Quest 3 before anything goes into the app. Everything below runs on the PC, with the Python environment active and
`adb` on the PATH.

1. `pip install cmake ninja sentencepiece protobuf`
2. `python tools/build_llama.py fetch`: llama.cpp at the pinned release, into `third_party/llama.cpp`.
3. `python grounding/export_gguf.py grounding/models/qwen2.5-0.5b-instruct.json`: the model in 8-bit (Q8_0).
4. `python tools/build_llama.py android`: the test program, `libse_llama.so` and `llama-bench` for the headset, built
   with the NDK inside Unity's Android Build Support.
5. PC references for the candidates, in a run of their own:
   `python grounding/reference.py <run> grounding/models/qwen2.5-0.5b-instruct.json grounding/prompts/<id>.json --candidates`
6. A headset run: `python grounding/llama_headset.py prepare <run> grounding/models/qwen2.5-0.5b-instruct.json`, then
   `push`, `run` and `bench`, wearing the headset in Home meanwhile, so it runs as it does in use.
7. `python grounding/llama_headset.py check <run> --reference 20260929_A1_r016 <the candidates run>`.

The test program and model stay in `/data/local/tmp/se` on the headset; `adb shell rm -r /data/local/tmp/se` removes
them.

## Comparing configurations (A1.8c, O18)

`prepare` takes `--flash-attn auto|on|off`, `--no-repack`, `--no-mmap` and `--n-seq`; each configuration is a run of its
own. `check` prints the configuration, how far the cached scene lands from uncached, and how far scoring in one batch
lands from scoring one by one, with both times.

## In the app (A1.8c)

`python tools/build_llama.py android` also copies `libse_llama.so` into `quest-app/Assets/Plugins/Android/libs/arm64-v8a/`;
in Unity its import settings must say Android, ARM64. `python grounding/llama_headset.py push-model <description>` copies
the model into the app's data folder (the app must have run once). On the panel, Runtime chooses llama.cpp before Load.
