# llama.cpp: what we rely on

Facts about llama.cpp that the project depends on (A1.8b onwards), with where they come from, like `docs/meta-ai.md`
for Meta's runtime.

- **Release.** `b11277`, commit `eae11d22` (2026-09-30), pinned in `native/llama.cpp.pin`. Its C API (`include/llama.h`)
  differs from older write-ups: `llama_model_load_from_file`, `llama_init_from_model`, `llama_get_memory` with
  `llama_memory_seq_rm`, `llama_batch_get_one`, `llama_decode`, `llama_get_logits_ith`, `llama_vocab_*`.
- **Loading.** `llama_model_params.load_mode` replaces `use_mmap`; we use `LLAMA_LOAD_MODE_MMAP`, so the weights stay
  file-backed pages that Android can drop and re-read, rather than copies the app owns.
- **The cache.** `llama_memory_seq_rm(mem, 0, keep, -1)` drops everything after `keep` positions; a batch from
  `llama_batch_get_one` then continues at `keep`. Only the last evaluated position's scores exist, so after a drop there
  are none until the next evaluation.
- **Batch rounding.** A prompt evaluated in one batch or in two (a cached scene, then the rest) gives slightly different
  floats: up to 0.002 in log-probability on the tiny test model.
- **Tokenizer.** With special tokens parsed and none added, llama.cpp's Qwen2 tokenizer (`models/ggml-vocab-qwen2.gguf`)
  gives exactly the token IDs of Hugging Face's for our four prompts (230, 232, 230 and 227 tokens), the answer prefix
  and the candidates (`native/tests/test_se_llama.py`; references `20260929_A1_r016`). `<|im_end|>` ends generation.
- **Android build** (`docs/android.md`): the NDK's CMake toolchain, `arm64-v8a`, `GGML_NATIVE=OFF`, `GGML_OPENMP=OFF`,
  `GGML_LLAMAFILE=OFF`, `LLAMA_OPENSSL=OFF`, and no global `-march`. `GGML_CPU_ARM_ARCH` raises the instruction set
  for llama.cpp's CPU code alone; we use `armv8.2-a+dotprod+fp16`.
- **Conversion.** `convert_hf_to_gguf.py --outtype q8_0` writes 8-bit weights directly, no separate quantizing tool; it
  needs torch, transformers, numpy, sentencepiece and protobuf.

## On the headset (A1.8b, `20260930_A1_r027`)

Measured from `adb shell`, not inside the app, with 8-bit weights (Q8_0, 531 MB) and `armv8.2-a+dotprod+fp16`; the
runtime reports `NEON, ARM_FMA, FP16_VA, DOTPROD, REPACK`.

- **Speed.** The 230-token prompt takes 0.82 s at 4 threads and 1.39 s at 2; each further token 35 ms at either.
  llama-bench: 285, 170 and 77 prompt tokens per second at 4, 2 and 1 threads; 31.5, 33.5 and 25.8 generated tokens
  per second. So generation is fastest at 2 threads, the prompt pass at 4.
- **The cache.** The scene (191 tokens) takes 0.68 s once; each command's remaining 36 to 41 tokens then take 0.18 s.
- **Memory.** The test program holds 1.06 GB after loading and peaks at 1.08 GB. Weight repacking (`REPACK`) copies the
  weights, so the memory-mapped file's own pages add to that until Android reclaims them.
- **Determinism.** Repeats give bit-identical results, and 2 threads the same numbers as 4.
- **Not batch-invariant.** The cached path (scene in one batch, the rest in another) gives slightly other numbers than
  one batch: up to 0.30 in a candidate's log-probability, 0.053 in total variation distance (O18). Both stay within
  0.09 of PyTorch's 32-bit distribution, the cached one within 0.035, with the same decisions.
- **Build size.** Unstripped, `libse_llama.so` is 49 MB and `llama-bench` 123 MB.

## Options and batched scoring (A1.8c, D57)

- **Several sequences share one cache** when `kv_unified` is on; with it off, llama.cpp splits `n_ctx` between the
  `n_seq_max` sequences. `llama_memory_seq_cp(mem, 0, i, -1, -1)` then gives sequence i the cached prompt by tagging
  its cells, without copying. `se_score_many` scores each candidate as such a sequence, all in one batch; on the tiny
  test model it gives exactly `se_score`'s numbers.
- **Flash attention isn't batch-invariant.** On the tiny test model a cached scene plus the rest differs from the whole
  prompt by 0.0022 in log-probability with flash attention on (llama.cpp's automatic choice on the CPU) and by nothing
  with it off. Weight repacking changes nothing there.
- **Options** of `se_load_ex`: `use_extra_bufts = false` keeps the weights in the file's layout (no repacked copy),
  `flash_attn_type` forces flash attention off or on, `load_mode = LLAMA_LOAD_MODE_NONE` reads the file into memory.
- **Stripping.** `llvm-strip --strip-unneeded` keeps the exported functions that P/Invoke and `dlsym` need.
