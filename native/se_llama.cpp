// Second Eyes: a small C interface over llama.cpp (A1.8b, D55). See se_llama.h.
#include "se_llama.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#ifndef SE_LLAMA_VERSION
#define SE_LLAMA_VERSION "unknown"
#endif

namespace {

std::mutex g_mutex;
std::string g_error;
bool g_verbose = false;
std::once_flag g_init;

void set_error(const std::string & text) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_error = text;
}

std::string last_error() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_error;
}

void on_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR && text != nullptr) {
        std::string line(text);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        set_error(line);
    }
    if (g_verbose && text != nullptr) fputs(text, stderr);
}

}  // namespace

struct se_llama {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t n_vocab = 0;
    int32_t n_batch = 512;
    int32_t n_seq = 1;
    int32_t flags = 0;
    std::vector<float> last;     // scores of the last evaluated position (llama.cpp's own buffer goes stale on drops)
    double last_lse = 0.0;       // log-sum-exp of `last`
    bool have_last = false;
};

namespace {

void keep_last(se_llama * s, const float * row) {
    std::memcpy(s->last.data(), row, sizeof(float) * s->n_vocab);
    float top = *std::max_element(s->last.begin(), s->last.end());
    double sum = 0.0;
    for (float v : s->last) sum += std::exp(static_cast<double>(v) - top);
    s->last_lse = top + std::log(sum);
    s->have_last = true;
}

}  // namespace

extern "C" {

const char * se_system_info(void) { return llama_print_system_info(); }

const char * se_llama_version(void) { return SE_LLAMA_VERSION; }

const char * se_last_error(void) {
    static thread_local std::string copy;
    copy = last_error();
    return copy.c_str();
}

void se_set_verbose(int32_t on) { g_verbose = on != 0; }

se_llama * se_load(const char * path, int32_t n_ctx, int32_t n_threads) {
    return se_load_ex(path, n_ctx, n_threads, 16, 0);
}

se_llama * se_load_ex(const char * path, int32_t n_ctx, int32_t n_threads, int32_t n_seq, int32_t flags) {
    std::call_once(g_init, [] {
        llama_log_set(on_log, nullptr);
        llama_backend_init();
    });
    set_error("");
    if (path == nullptr) {
        set_error("no model path");
        return nullptr;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                    // CPU only
    mp.load_mode = (flags & SE_NO_MMAP) ? LLAMA_LOAD_MODE_NONE
                                        : LLAMA_LOAD_MODE_MMAP;   // file-backed pages the system can drop and re-read
    mp.use_extra_bufts = (flags & SE_NO_REPACK) == 0;              // repacked weights: faster, but a copy
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        std::string why = last_error();
        set_error(std::string("could not load ") + path + (why.empty() ? "" : ": " + why));
        return nullptr;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx > 0 ? static_cast<uint32_t>(n_ctx) : 1024;
    cp.n_batch = std::min<uint32_t>(512, cp.n_ctx);
    cp.n_ubatch = cp.n_batch;
    cp.n_seq_max = static_cast<uint32_t>(std::max(1, n_seq));
    cp.kv_unified = true;   // one cache for all sequences: se_score_many's share the cached prompt, and none splits n_ctx
    if (flags & SE_FLASH_ATTN_OFF) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (flags & SE_FLASH_ATTN_ON) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads = cp.n_threads_batch = n_threads > 0 ? n_threads : 4;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        std::string why = last_error();
        llama_model_free(model);
        set_error("could not create a context" + (why.empty() ? std::string() : ": " + why));
        return nullptr;
    }
    auto * s = new se_llama();
    s->model = model;
    s->ctx = ctx;
    s->vocab = llama_model_get_vocab(model);
    s->n_vocab = llama_vocab_n_tokens(s->vocab);
    s->n_batch = static_cast<int32_t>(cp.n_batch);
    s->n_seq = static_cast<int32_t>(cp.n_seq_max);
    s->flags = flags;
    s->last.assign(static_cast<size_t>(s->n_vocab), 0.0f);
    return s;
}

void se_free(se_llama * s) {
    if (s == nullptr) return;
    if (s->ctx != nullptr) llama_free(s->ctx);
    if (s->model != nullptr) llama_model_free(s->model);
    delete s;
}

void se_set_threads(se_llama * s, int32_t n_threads) {
    if (s != nullptr && n_threads > 0) llama_set_n_threads(s->ctx, n_threads, n_threads);
}

int32_t se_n_vocab(const se_llama * s) { return s != nullptr ? s->n_vocab : 0; }

int32_t se_flags(const se_llama * s) { return s != nullptr ? s->flags : 0; }

int32_t se_n_seq(const se_llama * s) { return s != nullptr ? s->n_seq : 0; }

int32_t se_n_ctx(const se_llama * s) { return s != nullptr ? static_cast<int32_t>(llama_n_ctx(s->ctx)) : 0; }

int32_t se_tokenize(const se_llama * s, const char * text, int32_t * out, int32_t max) {
    if (s == nullptr || text == nullptr) return 0;
    return llama_tokenize(s->vocab, text, static_cast<int32_t>(std::strlen(text)), out, max,
                          /*add_special=*/false, /*parse_special=*/true);
}

int32_t se_piece(const se_llama * s, int32_t token, char * out, int32_t max) {
    if (s == nullptr) return 0;
    return llama_token_to_piece(s->vocab, token, out, max, /*lstrip=*/0, /*special=*/true);
}

int32_t se_is_eog(const se_llama * s, int32_t token) {
    return s != nullptr && llama_vocab_is_eog(s->vocab, token) ? 1 : 0;
}

int32_t se_n_cached(const se_llama * s) {
    if (s == nullptr) return 0;
    llama_pos top = llama_memory_seq_pos_max(llama_get_memory(s->ctx), 0);
    return top < 0 ? 0 : static_cast<int32_t>(top) + 1;
}

int32_t se_eval(se_llama * s, const int32_t * tokens, int32_t n, int32_t keep) {
    if (s == nullptr) return -1;
    int32_t cached = se_n_cached(s);
    if (keep < 0 || keep > cached) {
        set_error("keep (" + std::to_string(keep) + ") must be between 0 and the cached " + std::to_string(cached));
        return -2;
    }
    if (keep < cached) {
        if (!llama_memory_seq_rm(llama_get_memory(s->ctx), 0, keep, -1)) {
            set_error("llama.cpp could not drop the cache after position " + std::to_string(keep));
            return -3;
        }
        s->have_last = false;
    }
    if (n <= 0) return 0;
    if (keep + n > se_n_ctx(s)) {
        set_error("the context holds " + std::to_string(se_n_ctx(s)) + " tokens; " + std::to_string(keep + n) +
                  " don't fit");
        return -4;
    }
    std::vector<llama_token> buf(tokens, tokens + n);
    for (int32_t i = 0; i < n; i += s->n_batch) {
        int32_t m = std::min(s->n_batch, n - i);
        int32_t r = llama_decode(s->ctx, llama_batch_get_one(buf.data() + i, m));
        if (r != 0) {
            s->have_last = false;
            std::string why = last_error();
            set_error("llama_decode returned " + std::to_string(r) + (why.empty() ? "" : ": " + why));
            return -10;
        }
    }
    const float * row = llama_get_logits_ith(s->ctx, -1);
    if (row == nullptr) {
        s->have_last = false;
        set_error("llama.cpp returned no scores for the last position");
        return -5;
    }
    keep_last(s, row);
    return 0;
}

int32_t se_argmax(const se_llama * s) {
    if (s == nullptr || !s->have_last) return -1;
    return static_cast<int32_t>(std::max_element(s->last.begin(), s->last.end()) - s->last.begin());
}

double se_logprob(const se_llama * s, int32_t token) {
    if (s == nullptr || !s->have_last || token < 0 || token >= s->n_vocab) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(s->last[static_cast<size_t>(token)]) - s->last_lse;
}

int32_t se_logits(const se_llama * s, float * out, int32_t max) {
    if (s == nullptr || !s->have_last) return 0;
    int32_t n = std::min(max, s->n_vocab);
    std::memcpy(out, s->last.data(), sizeof(float) * static_cast<size_t>(n));
    return n;
}

double se_score(se_llama * s, const int32_t * tokens, int32_t n) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (s == nullptr || !s->have_last || tokens == nullptr || n <= 0) {
        set_error("se_score needs evaluated tokens before it and at least one token to score");
        return nan;
    }
    const int32_t base = se_n_cached(s);
    const std::vector<float> saved = s->last;
    const double saved_lse = s->last_lse;
    double sum = se_logprob(s, tokens[0]);
    for (int32_t i = 0; i + 1 < n; i++) {
        if (se_eval(s, tokens + i, 1, base + i) != 0) {
            sum = nan;
            break;
        }
        sum += se_logprob(s, tokens[i + 1]);
    }
    if (n > 1 && se_n_cached(s) > base) {
        llama_memory_seq_rm(llama_get_memory(s->ctx), 0, base, -1);
    }
    s->last = saved;
    s->last_lse = saved_lse;
    s->have_last = true;
    return sum;
}

int32_t se_score_many(se_llama * s, const int32_t * tokens, const int32_t * lengths, int32_t n, double * out) {
    if (s == nullptr || tokens == nullptr || lengths == nullptr || out == nullptr || n <= 0 || !s->have_last) {
        set_error("se_score_many needs evaluated tokens before it and at least one continuation");
        return -1;
    }
    if (n >= s->n_seq) {
        set_error("se_score_many takes at most " + std::to_string(s->n_seq - 1) + " continuations at once (n_seq " +
                  std::to_string(s->n_seq) + ")");
        return -2;
    }
    const int32_t base = se_n_cached(s);
    int32_t fed = 0;
    const int32_t * t = tokens;
    for (int32_t i = 0; i < n; i++) {
        if (lengths[i] <= 0) {
            set_error("every continuation needs at least one token");
            return -3;
        }
        out[i] = se_logprob(s, t[0]);   // the first token's probability comes from the scores we already have
        fed += lengths[i] - 1;          // the others need the tokens before them evaluated
        t += lengths[i];
    }
    if (fed == 0) return 0;
    if (base + fed > se_n_ctx(s)) {
        set_error("the context holds " + std::to_string(se_n_ctx(s)) + " tokens; " + std::to_string(base + fed) +
                  " don't fit");
        return -4;
    }
    llama_memory_t mem = llama_get_memory(s->ctx);
    for (int32_t i = 1; i <= n; i++) llama_memory_seq_cp(mem, 0, i, -1, -1);   // shares the cells, copies nothing
    llama_batch batch = llama_batch_init(fed, 0, 1);
    t = tokens;
    int32_t k = 0;
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j + 1 < lengths[i]; j++, k++) {
            batch.token[k] = t[j];
            batch.pos[k] = base + j;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = i + 1;
            batch.logits[k] = 1;
        }
        t += lengths[i];
    }
    batch.n_tokens = fed;
    int32_t result = llama_decode(s->ctx, batch);
    if (result == 0) {
        t = tokens;
        k = 0;
        for (int32_t i = 0; i < n && result == 0; i++) {
            for (int32_t j = 0; j + 1 < lengths[i]; j++, k++) {
                const float * row = llama_get_logits_ith(s->ctx, k);
                if (row == nullptr) {
                    result = -5;
                    break;
                }
                float top = *std::max_element(row, row + s->n_vocab);
                double sum = 0.0;
                for (int32_t v = 0; v < s->n_vocab; v++) sum += std::exp(static_cast<double>(row[v]) - top);
                out[i] += static_cast<double>(row[t[j + 1]]) - (top + std::log(sum));
            }
            t += lengths[i];
        }
        if (result != 0) set_error("llama.cpp returned no scores for a continuation's token");
    } else {
        std::string why = last_error();
        set_error("llama_decode returned " + std::to_string(result) + (why.empty() ? "" : ": " + why));
        result = -10;
    }
    llama_batch_free(batch);
    for (int32_t i = 1; i <= n; i++) llama_memory_seq_rm(mem, i, -1, -1);   // sequence 0 and our scores stay as they were
    return result;
}

int64_t se_memory_kb(int32_t peak) {
    std::ifstream status("/proc/self/status");
    if (!status) return -1;
    const std::string key = peak ? "VmHWM:" : "VmRSS:";
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, key.size(), key) == 0) {
            return std::strtoll(line.c_str() + key.size(), nullptr, 10);
        }
    }
    return -1;
}

}  // extern "C"
