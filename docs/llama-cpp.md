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
