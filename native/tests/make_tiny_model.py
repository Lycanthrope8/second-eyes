"""A tiny Qwen2-architecture GGUF for testing libse_llama without the real model (A1.8b).

Two layers of width 64 with random weights, so its answers are nonsense, but it carries Qwen's real tokenizer, copied
from llama.cpp's models/ggml-vocab-qwen2.gguf. That tests everything that doesn't depend on trained weights:
tokenization against the PC reference's token IDs, the cache, scoring, and the test program's flow.

    python native/tests/make_tiny_model.py LLAMA_DIR OUT.gguf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EMBD, LAYERS, HEADS, KV_HEADS, FF = 64, 2, 4, 2, 128
TOKENIZER_KEYS = ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
                  "tokenizer.ggml.merges", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id",
                  "tokenizer.ggml.padding_token_id")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    llama_dir, out = Path(argv[0]), Path(argv[1])
    sys.path.insert(0, str(llama_dir / "gguf-py"))
    import gguf

    vocab = gguf.GGUFReader(str(llama_dir / "models" / "ggml-vocab-qwen2.gguf"))
    fields = {name: vocab.fields[name] for name in TOKENIZER_KEYS}
    tokens = fields["tokenizer.ggml.tokens"].contents()
    n_vocab = len(tokens)

    w = gguf.GGUFWriter(str(out), "qwen2")
    w.add_name("second-eyes tiny test model")
    w.add_context_length(2048)
    w.add_embedding_length(EMBD)
    w.add_block_count(LAYERS)
    w.add_feed_forward_length(FF)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_rope_freq_base(1_000_000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model(fields["tokenizer.ggml.model"].contents())
    w.add_tokenizer_pre(fields["tokenizer.ggml.pre"].contents())
    w.add_token_list(tokens)
    w.add_token_types(fields["tokenizer.ggml.token_type"].contents())
    w.add_token_merges(fields["tokenizer.ggml.merges"].contents())
    w.add_eos_token_id(fields["tokenizer.ggml.eos_token_id"].contents())
    w.add_bos_token_id(fields["tokenizer.ggml.bos_token_id"].contents())
    w.add_pad_token_id(fields["tokenizer.ggml.padding_token_id"].contents())

    rng = np.random.default_rng(1234)
    rand = lambda *shape: (rng.standard_normal(shape) * 0.05).astype(np.float32)
    kv = EMBD // HEADS * KV_HEADS
    w.add_tensor("token_embd.weight", rand(n_vocab, EMBD))    # also the output layer: tied, as in Qwen2.5-0.5B
    w.add_tensor("output_norm.weight", np.ones(EMBD, dtype=np.float32))
    for i in range(LAYERS):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(EMBD, dtype=np.float32))
        w.add_tensor(f"blk.{i}.attn_q.weight", rand(EMBD, EMBD))
        w.add_tensor(f"blk.{i}.attn_q.bias", rand(EMBD))
        w.add_tensor(f"blk.{i}.attn_k.weight", rand(kv, EMBD))
        w.add_tensor(f"blk.{i}.attn_k.bias", rand(kv))
        w.add_tensor(f"blk.{i}.attn_v.weight", rand(kv, EMBD))
        w.add_tensor(f"blk.{i}.attn_v.bias", rand(kv))
        w.add_tensor(f"blk.{i}.attn_output.weight", rand(EMBD, EMBD))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(EMBD, dtype=np.float32))
        w.add_tensor(f"blk.{i}.ffn_gate.weight", rand(FF, EMBD))
        w.add_tensor(f"blk.{i}.ffn_up.weight", rand(FF, EMBD))
        w.add_tensor(f"blk.{i}.ffn_down.weight", rand(EMBD, FF))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Wrote {out}: {n_vocab} tokens, {LAYERS} layers of width {EMBD}, {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
