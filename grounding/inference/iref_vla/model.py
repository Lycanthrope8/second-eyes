"""The pinned checkpoint in PyTorch for the A2.3a pilot (D81), and its identity.

float32, evaluation mode, no gradients, eager attention, TF32 off, deterministic algorithms requested (warn-only, so a
missing deterministic kernel is recorded rather than fatal), no cache within or across requests, and final-position
logits only (`logits_to_keep=1`). The canary's independent path reads the last hidden state and applies the model's
own `lm_head`, a different route to the same logits. Nothing here downloads, upgrades or substitutes anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import warnings
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, issue


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def checkpoint_evidence(model_dir, protocol, desc=None, expected_weights_sha256=None) -> dict:
    """Tie the local weights to the pinned revision, or refuse naming the missing evidence.

    Accepted ties: Hugging Face's local-folder download metadata (.cache/huggingface/download/<file>.metadata: the
    commit and the ETag, which is the SHA-256 of a large file and the Git blob SHA-1 of a small one), or a published
    SHA-256 of the weights the operator supplies. A folder name or a matching architecture alone is not evidence.
    """
    d, bad, files = Path(model_dir), [], {}
    if not d.is_dir():
        raise EvaluationInputError([issue(str(d), "E_PILOT_CHECKPOINT", "the model folder does not exist")])
    weights = sorted(p for p in d.iterdir() if p.is_file() and p.suffix == ".safetensors")
    if not weights:
        bad.append("no .safetensors weights in the model folder")
    if not (d / "config.json").is_file():
        bad.append("no config.json in the model folder")
    meta = d / ".cache" / "huggingface" / "download"
    revision = protocol["model"]["hf_revision"]
    for p in weights + [d / n for n in ("config.json", "generation_config.json", "vocab.json", "merges.txt",
                                         "tokenizer_config.json", "tokenizer.json") if (d / n).is_file()]:
        rec = {"sha256": _sha256(p), "bytes": p.stat().st_size, "metadata": None}
        m = meta / f"{p.name}.metadata"
        if m.is_file():
            lines = m.read_text(encoding="utf-8").splitlines()
            rec["metadata"] = {"commit": lines[0] if lines else None, "etag": lines[1] if len(lines) > 1 else None}
        files[p.name] = rec
    for name, want in protocol["tokenizer"]["files"].items():
        if name in files and files[name]["sha256"] != want:
            bad.append(f"{name} in the model folder differs from the pinned tokenizer file")
    tie = None
    if weights and all(files[p.name]["metadata"] for p in weights) and files.get("config.json", {}).get("metadata"):
        problems = []
        for name, rec in files.items():
            md = rec["metadata"]
            if md is None:
                continue
            if md["commit"] != revision:
                problems.append(f"{name}: download metadata names commit {md['commit']}, not {revision}")
            etag = (md["etag"] or "").strip('"')
            ok = etag == rec["sha256"] if len(etag) == 64 else etag == _git_blob_sha1(d / name)
            if not ok:
                problems.append(f"{name}: its contents do not match the download metadata's ETag")
        bad += problems
        if not problems:
            tie = "Hugging Face local-folder download metadata: commit and ETag of every weight file and config.json"
    elif expected_weights_sha256:
        if len(weights) == 1 and files[weights[0].name]["sha256"] == expected_weights_sha256.lower():
            tie = "operator-supplied published SHA-256 of the weights at the pinned revision"
        else:
            bad.append("the weights' SHA-256 does not equal the supplied published value")
    else:
        bad.append("no evidence ties these weights to the pinned revision: no download metadata in "
                   ".cache/huggingface/download/ and no --expected-weights-sha256")
    try:
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        cfg = {}
        bad.append(f"config.json is unreadable: {e}")
    arch = (desc or {}).get("architecture", {})
    if cfg and arch:
        head_dim = cfg.get("hidden_size", 0) // max(cfg.get("num_attention_heads", 1), 1)
        if (cfg.get("num_hidden_layers"), cfg.get("num_key_value_heads"), head_dim) != (
                arch.get("max_layers"), arch.get("num_key_value_heads"), arch.get("head_dim")):
            bad.append("config.json's layers, key-value heads or head size differ from the model description")
    if cfg and cfg.get("max_position_embeddings", 0) < protocol["context_limit_tokens"]:
        bad.append(f"the model supports {cfg.get('max_position_embeddings')} positions, below the pilot's "
                   f"{protocol['context_limit_tokens']}")
    if bad or tie is None:
        raise EvaluationInputError([issue(str(d), "E_PILOT_CHECKPOINT", m) for m in (bad or ["identity not established"])])
    return {"revision": revision, "tie": tie, "files": files, "max_position_embeddings": cfg.get("max_position_embeddings")}


class TorchModel:
    """Implements the worker's model interface: forward_last, independent_last, info, synchronize, peak_memory."""

    def __init__(self, model, torch, device, info):
        self.model, self.torch, self.device, self._info = model, torch, device, info

    @classmethod
    def load(cls, model_dir, device, protocol):
        if device not in ("cpu", "cuda"):
            raise EvaluationInputError([issue("device", "E_PILOT_DEVICE", "device must be cpu or cuda")])
        if device == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        if device == "cuda" and not torch.cuda.is_available():
            raise EvaluationInputError([issue("device", "E_PILOT_DEVICE", "cuda was requested but torch.cuda.is_available() "
                                                                         "is False; there is no CPU fallback")])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False,
                                                         torch_dtype=torch.float32,
                                                         attn_implementation=protocol["model"]["attention_implementation"])
            model.to(device)
            model.eval()
            if device == "cuda":
                torch.cuda.synchronize()
        load_s = time.perf_counter() - t0
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
        if dtypes != ["torch.float32"]:
            raise EvaluationInputError([issue("model", "E_PILOT_DTYPE", f"parameters are {dtypes}, not float32 only")])
        info = {"kind": "exact", "framework": "pytorch", "torch": torch.__version__, "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "device": device,
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or platform.machine(),
                "dtype": dtypes[0], "attention_implementation": getattr(model.config, "_attn_implementation", None),
                "max_position_embeddings": model.config.max_position_embeddings,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "deterministic_warn_only": True,
                "tf32": False, "logits_api": "forward(logits_to_keep=1, use_cache=False)", "load_s": load_s,
                "load_warnings": sorted({str(w.message)[:200] for w in caught}),
                "gpu_bytes_after_load": torch.cuda.max_memory_allocated() if device == "cuda" else None}
        return cls(model, torch, device, info)

    def info(self) -> dict:
        return dict(self._info)

    def synchronize(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def peak_memory(self):
        return self.torch.cuda.max_memory_allocated() if self.device == "cuda" else None

    def _ids(self, token_ids):
        t = self.torch
        x = t.tensor([list(token_ids)], dtype=t.long, device=self.device)
        return x, t.ones_like(x)

    def forward_last(self, token_ids) -> list:
        with self.torch.inference_mode():
            x, mask = self._ids(token_ids)
            out = self.model(input_ids=x, attention_mask=mask, use_cache=False, logits_to_keep=1)
            return out.logits[0, -1].float().cpu().tolist()

    def independent_last(self, token_ids) -> list:
        with self.torch.inference_mode():
            x, mask = self._ids(token_ids)
            hidden = self.model.model(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state[:, -1, :]
            return self.model.lm_head(hidden)[0].float().cpu().tolist()
