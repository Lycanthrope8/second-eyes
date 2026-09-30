// Second Eyes: A1.8b's test program for llama.cpp on the headset (D55). Runs from `adb shell`, without Unity.
//
//   se_llama_cli JOB.json RESULTS.json
//
// JOB.json (written by grounding/llama_headset.py):
//   {"model": path, "n_ctx": 1024, "threads": [4, 2], "repeats": 3, "max_new_tokens": 100,
//    "repack": true, "flash_attn": "auto" | "on" | "off", "mmap": true, "n_seq": 16,
//    "prompts": [{"id", "formatted", "prefix_chars", "answer_prefix", "candidates": [...], "candidate_suffix"}]}
// For every thread count and prompt it measures, from an empty cache: the prompt pass, the greedy answer token by
// token, and the log-probability of each candidate after the prompt plus answer_prefix, all candidates in one batch
// (se_score_many, as the app does) and, on the first repeat, also one by one (se_score) for comparison. Then the same from a cached
// scene: formatted[0:prefix_chars] is evaluated once and kept, and each command evaluates only the rest. RESULTS.json
// holds the token IDs, answers, scores, every timing and peak memory, for grounding/llama_headset.py to check.
#include "se_llama.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::vector<int32_t> tokenize(se_llama * s, const std::string & text) {
    std::vector<int32_t> out(text.size() + 16);
    int32_t n = se_tokenize(s, text.c_str(), out.data(), static_cast<int32_t>(out.size()));
    if (n < 0) {
        out.resize(static_cast<size_t>(-n));
        n = se_tokenize(s, text.c_str(), out.data(), static_cast<int32_t>(out.size()));
    }
    out.resize(n > 0 ? static_cast<size_t>(n) : 0);
    return out;
}

std::string piece(se_llama * s, int32_t token) {
    char buf[256];
    int32_t n = se_piece(s, token, buf, sizeof(buf));
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

void fail(const std::string & what) {
    std::cerr << "Error: " << what << (se_last_error()[0] ? std::string(": ") + se_last_error() : "") << std::endl;
    std::exit(1);
}

void eval(se_llama * s, const std::vector<int32_t> & t, int32_t keep) {
    if (se_eval(s, t.data(), static_cast<int32_t>(t.size()), keep) != 0) fail("evaluation failed");
}

// Greedy answer from the current cache end, as Meta's runner and grounding/reference.py do: stop at an end token.
json greedy(se_llama * s, int32_t max_new) {
    json out;
    std::vector<int32_t> ids;
    std::vector<double> step_ms;
    std::string text, stop = "max_new_tokens";
    for (int32_t i = 0; i < max_new; i++) {
        int32_t tok = se_argmax(s);
        if (se_is_eog(s, tok)) {
            stop = "eos";
            break;
        }
        ids.push_back(tok);
        text += piece(s, tok);
        if (i + 1 == max_new) break;
        double t0 = now_ms();
        eval(s, {tok}, se_n_cached(s));
        step_ms.push_back(now_ms() - t0);
    }
    out["token_ids"] = ids;
    out["text"] = text;
    out["stop"] = stop;
    out["step_ms"] = step_ms;
    return out;
}

// Each candidate's log-probability after the cache end plus answer_prefix, all in one batch (se_score_many) and, if
// asked, also one by one (se_score); the cache is left at `keep` plus the answer prefix.
json scores(se_llama * s, int32_t keep, const std::vector<int32_t> & answer_prefix,
            const std::vector<std::vector<int32_t>> & candidates, const json & names, bool also_sequential) {
    json out;
    double t0 = now_ms();
    eval(s, answer_prefix, keep);
    std::vector<int32_t> flat, lengths;
    for (const auto & c : candidates) {
        flat.insert(flat.end(), c.begin(), c.end());
        lengths.push_back(static_cast<int32_t>(c.size()));
    }
    std::vector<double> lp(candidates.size());
    if (se_score_many(s, flat.data(), lengths.data(), static_cast<int32_t>(candidates.size()), lp.data()) != 0) {
        fail("batched scoring failed");
    }
    json batched = json::object();
    for (size_t k = 0; k < candidates.size(); k++) batched[names[k].get<std::string>()] = lp[k];
    out["logprobs"] = batched;
    out["ms"] = now_ms() - t0;
    if (also_sequential) {
        json one = json::object();
        double t1 = now_ms();
        for (size_t k = 0; k < candidates.size(); k++) {
            one[names[k].get<std::string>()] =
                se_score(s, candidates[k].data(), static_cast<int32_t>(candidates[k].size()));
        }
        out["sequential"] = {{"logprobs", one}, {"ms", now_ms() - t1}};
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "Usage: se_llama_cli JOB.json RESULTS.json" << std::endl;
        return 2;
    }
    json job;
    try {
        std::ifstream in(argv[1]);
        job = json::parse(in);
    } catch (const std::exception & e) {
        std::cerr << "Error: can't read " << argv[1] << ": " << e.what() << std::endl;
        return 2;
    }
    const std::vector<int32_t> threads = job.value("threads", std::vector<int32_t>{4});
    const int32_t repeats = job.value("repeats", 3);
    const int32_t max_new = job.value("max_new_tokens", 100);
    const std::string flash = job.value("flash_attn", std::string("auto"));
    const int32_t flags = (job.value("repack", true) ? 0 : SE_NO_REPACK) | (job.value("mmap", true) ? 0 : SE_NO_MMAP) |
                          (flash == "off" ? SE_FLASH_ATTN_OFF : flash == "on" ? SE_FLASH_ATTN_ON : 0);

    json res;
    res["llama_cpp"] = se_llama_version();
    res["system_info"] = se_system_info();
    res["memory_kb_before_load"] = se_memory_kb(0);
    double t0 = now_ms();
    se_llama * s = se_load_ex(job.at("model").get<std::string>().c_str(), job.value("n_ctx", 1024), threads.front(),
                              job.value("n_seq", 16), flags);
    if (s == nullptr) fail("could not load the model");
    res["load_ms"] = now_ms() - t0;
    res["options"] = {{"repack", job.value("repack", true)}, {"flash_attn", flash}, {"mmap", job.value("mmap", true)},
                      {"n_seq", se_n_seq(s)}, {"flags", se_flags(s)}};
    res["memory_kb_after_load"] = se_memory_kb(0);
    res["n_vocab"] = se_n_vocab(s);
    std::cout << "Loaded in " << res["load_ms"].get<double>() << " ms; " << se_system_info() << std::endl;

    json runs = json::array();
    for (int32_t n_threads : threads) {
        se_set_threads(s, n_threads);
        // Phase 1, uncached: every prompt from an empty cache.
        std::vector<json> results;
        std::vector<std::vector<int32_t>> all_ids, all_prefix, all_answer_prefix;
        std::vector<std::vector<std::vector<int32_t>>> all_cand;
        for (const json & p : job.at("prompts")) {
            const std::string formatted = p.at("formatted");
            const auto ids = tokenize(s, formatted);
            const auto prefix_ids = tokenize(s, formatted.substr(0, p.at("prefix_chars").get<size_t>()));
            const bool boundary_ok = prefix_ids.size() <= ids.size() &&
                                     std::equal(prefix_ids.begin(), prefix_ids.end(), ids.begin());
            const auto answer_prefix = tokenize(s, p.at("answer_prefix"));
            std::vector<std::vector<int32_t>> cand;
            json cand_ids = json::object();
            for (const auto & c : p.at("candidates")) {
                cand.push_back(tokenize(s, c.get<std::string>() + p.value("candidate_suffix", std::string())));
                cand_ids[c.get<std::string>()] = cand.back();
            }
            json r;
            r["threads"] = n_threads;
            r["prompt_id"] = p.at("id");
            r["prompt_token_ids"] = ids;
            r["prefix_tokens"] = prefix_ids.size();
            r["prefix_is_token_prefix"] = boundary_ok;
            r["answer_prefix_ids"] = answer_prefix;
            r["candidate_ids"] = cand_ids;
            json uncached = json::array();
            for (int32_t k = 0; k < repeats; k++) {
                json u;
                double t = now_ms();
                eval(s, ids, 0);
                u["prompt_ms"] = now_ms() - t;
                u["answer"] = greedy(s, max_new);
                u["scores"] = scores(s, static_cast<int32_t>(ids.size()), answer_prefix, cand, p.at("candidates"), k == 0);
                uncached.push_back(u);
            }
            r["uncached"] = uncached;
            std::cout << "threads " << n_threads << ", " << p.at("id").get<std::string>() << ", uncached: "
                      << uncached.back()["answer"]["text"].get<std::string>().substr(0, 80) << " (prompt "
                      << uncached.back()["prompt_ms"].get<double>() << " ms)" << std::endl;
            results.push_back(r);
            all_ids.push_back(ids);
            all_prefix.push_back(prefix_ids);
            all_answer_prefix.push_back(answer_prefix);
            all_cand.push_back(cand);
        }
        // Phase 2, cached: a scene is evaluated once from an empty cache and kept; each command evaluates the rest.
        se_eval(s, nullptr, 0, 0);
        std::vector<int32_t> scene;
        size_t i = 0;
        for (const json & p : job.at("prompts")) {
            json & r = results[i];
            const auto & ids = all_ids[i];
            const auto & prefix_ids = all_prefix[i];
            json cached = json::array();
            if (r["prefix_is_token_prefix"].get<bool>()) {
                if (scene != prefix_ids) {
                    double t = now_ms();
                    eval(s, prefix_ids, 0);
                    r["scene_evaluated_ms"] = now_ms() - t;
                    scene = prefix_ids;
                } else {
                    r["scene_reused"] = true;
                }
                const int32_t keep = static_cast<int32_t>(prefix_ids.size());
                const std::vector<int32_t> rest(ids.begin() + keep, ids.end());
                for (int32_t k = 0; k < repeats; k++) {
                    json c;
                    double t = now_ms();
                    eval(s, rest, keep);
                    c["suffix_tokens"] = rest.size();
                    c["suffix_ms"] = now_ms() - t;
                    c["answer"] = greedy(s, max_new);
                    c["scores"] = scores(s, static_cast<int32_t>(ids.size()), all_answer_prefix[i], all_cand[i],
                                         p.at("candidates"), k == 0);
                    cached.push_back(c);
                }
                se_eval(s, nullptr, 0, keep);   // back to the scene alone, for the next command
                std::cout << "threads " << n_threads << ", " << p.at("id").get<std::string>() << ", cached: "
                          << cached.back()["answer"]["text"].get<std::string>().substr(0, 80) << " (" << rest.size() << " tokens in "
                          << cached.back()["suffix_ms"].get<double>() << " ms)" << std::endl;
            }
            r["cached"] = cached;
            runs.push_back(r);
            i++;
        }
        se_eval(s, nullptr, 0, 0);
    }
    res["runs"] = runs;
    res["memory_kb_peak"] = se_memory_kb(1);
    res["memory_kb_end"] = se_memory_kb(0);
    se_free(s);

    std::ofstream out(argv[2]);
    out << res.dump(1) << std::endl;
    if (!out) {
        std::cerr << "Error: can't write " << argv[2] << std::endl;
        return 1;
    }
    std::cout << "Wrote " << argv[2] << "; peak memory " << res["memory_kb_peak"].get<int64_t>() / 1024 << " MB" << std::endl;
    return 0;
}
