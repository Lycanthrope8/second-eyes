// Second Eyes: a small C interface over llama.cpp (A1.8b, D55).
//
// It hides llama.cpp's structs, which change between versions, behind plain functions, so the test program and later
// the Unity app (C#, P/Invoke) only ever see ints, floats and strings. One se_llama holds a model and one context with a
// single sequence: the cache holds the tokens evaluated so far, and se_eval can keep a leading part of it (a cached
// scene) and evaluate new tokens after that. All scores are natural-log probabilities of the last evaluated position.
// Not thread-safe: use one se_llama from one thread at a time.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define SE_API __declspec(dllexport)
#else
#define SE_API __attribute__((visibility("default")))
#endif

typedef struct se_llama se_llama;

// llama.cpp's build and CPU features, e.g. "CPU : NEON = 1 | ARM_FMA = 1 | DOTPROD = 1 | ...".
SE_API const char * se_system_info(void);
// The llama.cpp release this library was built from, e.g. "b11277 (eae11d22)".
SE_API const char * se_llama_version(void);
// The last error, from this interface or from llama.cpp's log; empty if none.
SE_API const char * se_last_error(void);
// Echo llama.cpp's own log to stderr (0 or 1). Off by default.
SE_API void se_set_verbose(int32_t on);

// Options for se_load_ex, combined with |.
#define SE_NO_REPACK      1   // keep the weights in the file's layout: no repacked copy for faster arithmetic
#define SE_FLASH_ATTN_OFF 2   // never use flash attention (by default llama.cpp decides)
#define SE_FLASH_ATTN_ON  4   // always use flash attention
#define SE_NO_MMAP        8   // read the weights into memory instead of mapping the file

// Load a GGUF model with a context of n_ctx tokens shared by up to n_seq sequences (se_score_many scores each
// continuation as a sequence of its own, sharing the cached tokens), n_threads CPU threads and SE_* options. By default
// the weights are memory-mapped and repacked. NULL on failure.
SE_API se_llama * se_load_ex(const char * path, int32_t n_ctx, int32_t n_threads, int32_t n_seq, int32_t flags);
// se_load_ex(path, n_ctx, n_threads, 16, 0).
SE_API se_llama * se_load(const char * path, int32_t n_ctx, int32_t n_threads);
// The options and sequences se_load_ex was given.
SE_API int32_t se_flags(const se_llama * s);
SE_API int32_t se_n_seq(const se_llama * s);
SE_API void se_free(se_llama * s);
SE_API void se_set_threads(se_llama * s, int32_t n_threads);

SE_API int32_t se_n_vocab(const se_llama * s);
SE_API int32_t se_n_ctx(const se_llama * s);
// Tokenize UTF-8 text, parsing special tokens such as <|im_start|> and adding none (the prompt carries its template).
// Returns the number of tokens, or minus the number needed if max is too small.
SE_API int32_t se_tokenize(const se_llama * s, const char * text, int32_t * out, int32_t max);
// The text of one token, special tokens included. Returns its length, or minus the length needed.
SE_API int32_t se_piece(const se_llama * s, int32_t token, char * out, int32_t max);
// Whether a token ends generation (end of turn or of text).
SE_API int32_t se_is_eog(const se_llama * s, int32_t token);

// How many positions the cache holds.
SE_API int32_t se_n_cached(const se_llama * s);
// Keep the first `keep` cached positions, drop the rest, then evaluate n tokens after them. Afterwards the scores of
// the last token's position are available. keep = 0 starts afresh. With n = 0 it only drops; after a drop no scores
// are available (NaN) until the next evaluation, since llama.cpp only has those of the last evaluated position.
// Returns 0 or a negative error (see se_last_error).
SE_API int32_t se_eval(se_llama * s, const int32_t * tokens, int32_t n, int32_t keep);
// On the scores of the last evaluated position: the most likely token, a token's log-probability, and the raw row.
SE_API int32_t se_argmax(const se_llama * s);
SE_API double se_logprob(const se_llama * s, int32_t token);
SE_API int32_t se_logits(const se_llama * s, float * out, int32_t max);
// The log-probability of n tokens following the cache: log P(t0) + log P(t1 | t0) + ... The cache and the last scores
// are restored afterwards, so any number of continuations can be scored from the same point. NaN on error.
SE_API double se_score(se_llama * s, const int32_t * tokens, int32_t n);

// se_score for n continuations at once, in one evaluation: each gets a sequence of its own that shares the cache.
// tokens holds the continuations one after another, lengths[i] is the i-th one's length, and out[i] receives its
// log-probability. The cache and the last scores are restored. n must be below se_n_seq. Returns 0 or a negative error.
SE_API int32_t se_score_many(se_llama * s, const int32_t * tokens, const int32_t * lengths, int32_t n, double * out);

// This process's memory in KB, from /proc/self/status: peak (VmHWM) and current (VmRSS) resident size; -1 if unknown.
SE_API int64_t se_memory_kb(int32_t peak);

#ifdef __cplusplus
}
#endif
