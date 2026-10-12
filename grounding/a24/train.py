"""A2.4 training (D109). Runs in the separate training environment (`.venv-train`: PEFT added to the evaluation
environment's pinned torch, transformers and tokenizers). Every number below comes from `settings.v1.json`, frozen in
the training specification.

**A fit** (`fit`) starts from the pinned base and the declared initialization (PEFT's default after
`torch.manual_seed(seed)`), trains LoRA in float32 with eager attention, one example per micro-step, and averages each
group of 16 examples into one AdamW step under the linear warmup-and-decay schedule. Only the response is supervised:
the answer letter (direct) or the semantic string and its end token (parser); every prompt position is masked, nothing
is truncated. The data order per epoch comes from the seed and the epoch, so both direct formats see the same rows in
the same order.

**Resumable, identity-locked.** `lock.json` holds everything the fit depends on (specification, settings, data files,
base evidence, configuration, seed, code, software). A resume refuses any difference. The resume state (written every
20 optimizer steps and at each epoch's end) restores the adapter, the optimizer, the scheduler, the random number
generators and the data position; a float32 fit has no gradient scaler.

**Each epoch's end** saves a checkpoint (hashed) and runs the inner validation on the retained validation rows: the
accepted decision rule for direct selection (largest offered logit, exact ties to K), constrained decoding then the
accepted resolver in each retained view for the parser. `select` applies the frozen rule (best metric, ties to the
earlier checkpoint). `merge` merges the selected adapter in float32, checks the adapter-loaded model against the merged
one on bounded rows, copies the pinned tokenizer files and writes the fine-tuned identity.

**The smoke** (`smoke`, Stage A only) is separate from the scientific fits: three optimizer steps per configuration
(the longest example included), an interrupted-and-resumed copy compared with an uninterrupted one, validation, merge
and identity, with measured times and peak GPU memory extrapolated to Stage B.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from . import decoder as DC
from . import identity as ID
from . import language as L
from . import prep as PR

BASE_KEY = "qwen2.5-0.5b-instruct"
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json")


class TrainError(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _replace(tmp, path)


def _replace(src, dst, attempts=3, pauses=(2.0, 6.0)) -> None:
    """os.replace, retried when a transient Windows lock (antivirus or indexer) refuses it (A2.6b's rule)."""
    for k in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == attempts:
                raise
            time.sleep(pauses[min(k, len(pauses)) - 1])


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def versions() -> dict:
    import accelerate
    import peft
    import tokenizers
    import torch
    import transformers
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__}


def configuration(settings, name) -> dict:
    for c in settings["configurations"]:
        if c["name"] == name:
            return c
    raise TrainError(f"unknown configuration {name!r}")


def model_name(settings, config, seed_index) -> str:
    return settings["model_name"].format(configuration=config, seed_index=seed_index)


def base_identity(base_dir):
    """The accepted checkpoint_evidence of the base folder, with the model protocol and description it needs."""
    from ..inference.iref_vla.model import checkpoint_evidence
    from ..inference.iref_vla.protocol import load_protocol
    from ..inference.iref_vla_compare import design as CD
    from ..inference.iref_vla_compare import run as CR
    mp, desc = CR.model_protocol(load_protocol(), CD.load_policy(), BASE_KEY)
    ev = checkpoint_evidence(base_dir, mp, desc)
    return ev, mp, desc


# --------------------------------------------------------------------------------------------- data
def data_file(cohort, cfg) -> str:
    if cfg["design"] == "direct":
        return f"training_data/{cohort}-direct-{cfg['format']}.jsonl"
    return f"training_data/{cohort}-parser.jsonl"


def load_rows(prep, cohort, cfg):
    name = data_file(cohort, cfg)
    data = (Path(prep) / name).read_bytes()
    man = _read(Path(prep) / "manifest.json")
    if man["files"].get(name) != _sha(data):
        raise TrainError(f"{name} differs from the preparation manifest")
    rows = [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]
    return sorted(rows, key=lambda r: r["key"]), _sha(data)


def sequence(row, design):
    """(input IDs, labels): the response only is supervised; -100 masks the prompt."""
    if design == "direct":
        return row["token_ids"] + [row["answer_token_id"]], [-100] * len(row["token_ids"]) + [row["answer_token_id"]]
    return row["prompt_ids"] + row["output_ids"], [-100] * len(row["prompt_ids"]) + row["output_ids"]


def example_loss(model, row, design, device):
    """The example's response-only cross-entropy, with logits computed at the supervised positions only (the same
    value as the masked full-sequence loss, without a full-vocabulary logit row per prompt position). Direct: the last
    prompt position predicting the answer letter, the position the accepted runner scores. Parser: the positions
    predicting each output token and the end token, averaged over them."""
    import torch
    import torch.nn.functional as F
    if design == "direct":
        x = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        logits = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1:]
        return F.cross_entropy(logits.float(), torch.tensor([row["answer_token_id"]], device=device))
    out = row["output_ids"]
    x = torch.tensor([row["prompt_ids"] + out], dtype=torch.long, device=device)
    logits = model(input_ids=x, use_cache=False, logits_to_keep=len(out) + 1).logits[0, :len(out)]
    return F.cross_entropy(logits.float(), torch.tensor(out, device=device))


def order(keys, seed, epoch) -> list:
    rng = random.Random(int(_sha(f"a24/order\n{seed}\n{epoch}".encode("utf-8"))[:16], 16))
    k = sorted(keys)
    rng.shuffle(k)
    return k


def lr_lambda(total, warmup):
    """Linear warmup over `warmup` optimizer steps from 0, then linear decay to 0 at `total` (the scheduler's step)."""
    def f(s):
        if s < warmup:
            return s / max(1, warmup)
        return max(0.0, (total - s) / max(1, total - warmup))
    return f


def torch_setup(device):
    import torch
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise TrainError("cuda was requested but is not available; there is no CPU fallback for a fit")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_base(base_dir, device):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, attn_implementation="eager",
                                             local_files_only=True, trust_remote_code=False)
    if sorted({str(p.dtype) for p in m.parameters()}) != ["torch.float32"]:
        raise TrainError("the base is not float32 throughout")
    return m


def build(base_dir, settings, seed, device):
    import torch
    from peft import LoraConfig, get_peft_model
    base = load_base(base_dir, device)
    torch.manual_seed(seed)
    lc = settings["lora"]
    pm = get_peft_model(base, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias=lc["bias"],
                                         target_modules=list(lc["target_modules"]), task_type="CAUSAL_LM"))
    return pm.to(device)


# --------------------------------------------------------------------------------------------- inner validation
def validate_direct(model, rows, device) -> list:
    import torch
    from ..inference.iref_vla.choices import score
    model.eval()
    out = []
    with torch.inference_mode():
        for r in rows:
            t0 = time.perf_counter()
            x = torch.tensor([r["token_ids"]], dtype=torch.long, device=device)
            logits = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu().tolist()
            s = score(list(zip(r["codes"], r["choice_object_ids"], r["code_token_ids"])), logits)
            out.append({"key": r["key"], "view": r["view"], "choice_code": s["choice_code"], "answer_code": r["answer_code"],
                        "correct": s["choice_code"] == r["answer_code"], "offered": [x["logit"] for x in s["scores"]],
                        "input_tokens": len(r["token_ids"]), "ms": round((time.perf_counter() - t0) * 1000, 3)})
    model.train()
    return out


class Resolver:
    """The accepted resolver over the preparation's derived scenes (labels.Checker, A2.3b's configuration)."""

    def __init__(self, work):
        from . import labels as LB
        self.work, self.checker, self._maps = Path(work), LB.Checker(L.load_spec()), {}

    def __call__(self, parsed, row, view):
        b = self.work / row["scene"] / "bundle"
        vp = row["view_paths"][view]
        if row["scene"] not in self._maps:
            self._maps[row["scene"]] = _read(b / "model_records" / "category-map.json")
        return self.checker.resolve(L.interpretation(parsed), _read(b / vp["scene_path"]), _read(b / vp["command_path"]),
                                    self._maps[row["scene"]])


def parse_rows(model, rows, constraint, spec, resolver, device, views_of=lambda r: r["direct_views"]) -> list:
    """Constrained decoding once per parent, then the resolver in each of the parent's retained views."""
    model.eval()
    out = []
    for r in rows:
        t0 = time.perf_counter()
        res = DC.validate(DC.decode(DC.torch_step(model, r["prompt_ids"], device), constraint, spec["max_new_tokens"]), spec)
        dec_ms = (time.perf_counter() - t0) * 1000
        for v in views_of(r):
            t1 = time.perf_counter()
            if res["valid"]:
                rr = resolver(res["parsed"], r, v)
                outcome, tid = rr["outcome"], rr.get("target_id")
            else:
                outcome, tid = (f"decode:{res['status']}" if res["status"] != "completed" else "invalid_output"), None
            out.append({"key": f"{r['parent_command_id']}|{v}", "parent_command_id": r["parent_command_id"], "view": v,
                        "status": res["status"], "text": res["text"], "valid": res["valid"], "steps": res["steps"],
                        "mask_ms": res["mask_ms"], "model_ms": res["model_ms"], "decode_ms": round(dec_ms, 3),
                        "resolve_ms": round((time.perf_counter() - t1) * 1000, 3), "outcome": outcome,
                        "target_id": tid, "correct": outcome == "resolved" and tid == r["target_object_id"],
                        "label": r["label"]})
    model.train()
    return out


def validate(model, cfg, val_rows, constraint, spec, resolver, device) -> dict:
    rows = validate_direct(model, val_rows, device) if cfg["design"] == "direct" else \
        parse_rows(model, val_rows, constraint, spec, resolver, device)
    n = len(rows)
    return {"rows": rows, "metric": sum(r["correct"] for r in rows) / n if n else None, "denominator": n,
            "metric_name": "target accuracy" if cfg["design"] == "direct" else "resolved-target accuracy",
            "technical_failures": 0}


# --------------------------------------------------------------------------------------------- the fit
def _lock(prep, cfg, seed, seed_index, settings, settings_path, base_ev, name, limit, device) -> dict:
    man_bytes = (Path(prep) / "manifest.json").read_bytes()
    man = json.loads(man_bytes)
    files = {data_file(c, cfg): man["files"][data_file(c, cfg)] for c in PR.COHORTS}
    return {"format_version": 1, "record_type": "a24_fit_lock", "model_name": name, "configuration": cfg, "seed": seed,
            "seed_index": seed_index, "settings_sha256": _sha(Path(settings_path).read_bytes()),
            "spec_sha256": man["spec_sha256"], "prep_manifest_sha256": _sha(man_bytes), "data_files": files,
            "base": {"hf_repo": settings["base"]["hf_repo"], "hf_revision": settings["base"]["hf_revision"],
                     "evidence_tie": base_ev["tie"], "evidence_sha256": _sha(json.dumps(base_ev, sort_keys=True).encode())},
            "language_sha256": L.spec_sha256(), "code": PR.code_hashes() | _shared_code(), "software": versions(),
            "device": device, "smoke": limit}


def _shared_code() -> dict:
    """The shared modules A2.4 relies on, beyond its own package (the modified importer included)."""
    names = ["grounding/adapters/iref_vla/release.py", "grounding/preparation/iref_vla_dev/dev.py",
             "grounding/inference/iref_vla/choices.py", "grounding/inference/iref_vla/model.py",
             "grounding/evaluation/iref_vla/parse.py", "grounding/resolution/resolve.py", "grounding/resolution/validate.py"]
    return {n: _sha((PR.REPO / n).read_bytes()) for n in names}


def _save_state(fit_dir: Path, model, opt, sched, position, device):
    import torch
    from peft import get_peft_model_state_dict
    st = {"adapter": get_peft_model_state_dict(model), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
          "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
          "position": position}
    d = fit_dir / "resume"
    d.mkdir(exist_ok=True)
    tmp = d / "state.pt.tmp"
    torch.save(st, tmp)
    _replace(tmp, d / "state.pt")
    _write(d / "state.json", {"position": position, "sha256": ID._sha_file(d / "state.pt"), "written_utc": _now()})


def _load_state(fit_dir: Path, model, opt, sched, device) -> dict:
    import torch
    from peft import set_peft_model_state_dict
    meta = _read(fit_dir / "resume" / "state.json")
    if ID._sha_file(fit_dir / "resume" / "state.pt") != meta["sha256"]:
        raise TrainError("the resume state's hash differs from its record: refusing to continue")
    st = torch.load(fit_dir / "resume" / "state.pt", map_location="cpu", weights_only=False)   # our own hashed file
    set_peft_model_state_dict(model, st["adapter"])
    opt.load_state_dict(st["optimizer"])
    sched.load_state_dict(st["scheduler"])
    torch.set_rng_state(st["rng_cpu"])
    if device == "cuda" and st["rng_cuda"] is not None:
        torch.cuda.set_rng_state_all(st["rng_cuda"])
    return st["position"]


def _checkpoint(fit_dir: Path, model, epoch, step) -> dict:
    d = fit_dir / "checkpoints" / f"epoch-{epoch}"
    if (d / "checkpoint.json").is_file():
        rec = _read(d / "checkpoint.json")
        if ID.folder_files(d) == {**rec["files"], "checkpoint.json": ID.folder_files(d).get("checkpoint.json")}:
            return rec
        raise TrainError(f"{d} exists but its files differ from its record")
    tmp = Path(tempfile.mkdtemp(prefix=f".epoch-{epoch}-", dir=str(fit_dir)))
    model.save_pretrained(str(tmp))
    rec = {"epoch": epoch, "optimizer_step": step, "files": ID.folder_files(tmp), "saved_utc": _now()}
    _write(tmp / "checkpoint.json", rec)
    d.parent.mkdir(exist_ok=True)
    _replace(tmp, d)
    return rec


def fit(*, prep, work, config, seed_index, base_dir, out, device="cuda", resume=False, settings_path=PR.SETTINGS_PATH,
        limit=None, stop_after=None, base_evidence=None, progress=print) -> dict:
    """One planned fit (or, with `limit`, a Stage A smoke fit: never a scientific one). stop_after: stop once that many
    optimizer steps are done (tests and the smoke's resume check)."""
    import torch
    settings = PR.load_settings(settings_path)
    cfg = configuration(settings, config)
    seed = settings["seeds"][seed_index - 1]
    name = model_name(settings, config, seed_index) if not limit else f"a24.smoke.{config}.s{seed_index}"
    torch_setup(device)
    base_ev, _, _ = (base_evidence or base_identity)(base_dir)
    lock = _lock(prep, cfg, seed, seed_index, settings, settings_path, base_ev, name, limit, device)
    out = Path(out)
    if out.exists():
        if not resume:
            raise TrainError(f"{out} exists; pass resume to continue it")
        old = _read(out / "lock.json")
        if {k: v for k, v in old.items() if k != "created_utc"} != lock:
            raise TrainError("the fit's identity differs from its lock (inputs, code, software or settings): refusing to "
                             "continue; the partial evidence is kept")
    elif resume:
        raise TrainError(f"{out} does not exist; nothing to resume")
    else:
        out.mkdir(parents=True)
        _write(out / "lock.json", dict(lock, created_utc=_now()))
    rows, _ = load_rows(prep, "fitting", cfg)
    val, _ = load_rows(prep, "validation", cfg)
    by_key = {r["key"]: r for r in rows}
    keys = sorted(by_key) if not limit else list(limit["keys"])
    epochs = settings["epochs"] if not limit else 1
    accum = settings["batch"]["accumulation"]
    per_epoch = math.ceil(len(keys) / accum)
    total = epochs * per_epoch if not limit else min(limit["steps"], per_epoch)
    warmup = math.ceil(settings["schedule"]["warmup_fraction"] * total)
    model = build(base_dir, settings, seed, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    o = settings["optimizer"]
    opt = torch.optim.AdamW(trainable, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"], weight_decay=o["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, warmup))
    position = {"epoch": 0, "group": 0, "step": 0}
    if resume and (out / "resume" / "state.json").is_file():
        position = _load_state(out, model, opt, sched, device)
        if (out / "log.jsonl").is_file():   # steps after the saved state are redone: their log lines are dropped
            kept = [x for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()
                    if x.strip() and json.loads(x)["step"] <= position["step"]]
            (out / "log.jsonl").write_text("".join(x + "\n" for x in kept), encoding="utf-8")
        progress(f"  resumed at epoch {position['epoch']}, group {position['group']}, step {position['step']}")
    spec, constraint, resolver = L.load_spec(), None, None
    if cfg["design"] == "parser":
        from ..preparation.iref_vla import tokens as T
        constraint = DC.Constraint(spec, DC.TokenTable(T.load_pinned_tokenizer(PR.DEV.TOKENIZER)._backend))
        resolver = Resolver(work)
    seqs = {k: sequence(by_key[k], cfg["design"]) for k in keys}
    if any(len(s[0]) > settings["context"]["max_tokens"] for s in seqs.values()):
        raise TrainError("an example exceeds the context limit; examples are never truncated")
    step, model_ = position["step"], model
    model.train()
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    try:
        for e in range(position["epoch"], epochs):
            ordered = order(keys, seed, e) if not limit else keys
            groups = [ordered[i:i + accum] for i in range(0, len(ordered), accum)][:total if limit else None]
            start = position["group"] if e == position["epoch"] else 0
            for gi in range(start, len(groups)):
                g, t0, losses, toks = groups[gi], time.perf_counter(), [], 0
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                for k in g:
                    ids = seqs[k][0]
                    loss = example_loss(model_, by_key[k], cfg["design"], device)
                    if not torch.isfinite(loss):
                        raise TrainError(f"a non-finite loss at epoch {e}, group {gi}, example {k}")
                    (loss / len(g)).backward()
                    losses.append(float(loss.detach()))
                    toks += len(ids)
                gn = torch.nn.utils.clip_grad_norm_(trainable, settings["optimizer"]["max_grad_norm"])
                lr = sched.get_last_lr()[0]
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if device == "cuda":
                    torch.cuda.synchronize()
                log.write(json.dumps({"epoch": e, "group": gi, "step": step, "examples": len(g), "tokens": toks,
                                      "loss": statistics.fmean(losses), "lr": lr, "grad_norm": float(gn),
                                      "seconds": round(time.perf_counter() - t0, 4),
                                      "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
                          + "\n")
                log.flush()
                last = gi == len(groups) - 1
                if step % settings["checkpoints"]["resume_state_every_optimizer_steps"] == 0 or last or limit:
                    _save_state(out, model, opt, sched, {"epoch": e, "group": gi + 1, "step": step}, device)
                if stop_after is not None and step >= stop_after and not last:
                    return {"stopped": True, "step": step, "folder": str(out)}
            ck = _checkpoint(out, model, e + 1, step)
            vfile = out / "validation" / f"epoch-{e + 1}.json"
            if not vfile.is_file():
                rows_v = val if not limit else val[:limit.get("validation_rows", 8)]
                t0 = time.perf_counter()
                v = validate(model, cfg, rows_v, constraint, spec, resolver, device)
                v.update(epoch=e + 1, optimizer_step=step, checkpoint=f"epoch-{e + 1}",
                         checkpoint_files=ck["files"], seconds=round(time.perf_counter() - t0, 3))
                _write(vfile, v)
            _save_state(out, model, opt, sched, {"epoch": e + 1, "group": 0, "step": step}, device)
    finally:
        log.close()
    metrics = [{k: v for k, v in _read(out / "validation" / f"epoch-{k}.json").items() if k != "rows"}
               for k in range(1, epochs + 1)]
    done = {"model_name": name, "optimizer_steps": step, "planned_optimizer_steps": total, "metrics": metrics,
            "finished_utc": _now()}
    _write(out / "done.json", done)
    return dict(done, folder=str(out))


# --------------------------------------------------------------------------------------------- selection and merge
def select(fit_dir) -> dict:
    """The frozen rule: the best inner-validation metric; a tie goes to the earlier checkpoint."""
    fit_dir = Path(fit_dir)
    done = _read(fit_dir / "done.json")
    if done["optimizer_steps"] != done["planned_optimizer_steps"]:
        raise TrainError("the fit did not complete its planned steps")
    m = done["metrics"]
    best = max(m, key=lambda x: (x["metric"], -x["epoch"]))
    sel = {"model_name": done["model_name"], "checkpoint": best["checkpoint"], "epoch": best["epoch"],
           "optimizer_step": best["optimizer_step"], "metric_name": best["metric_name"], "value": best["metric"],
           "denominator": best["denominator"], "all": [{"epoch": x["epoch"], "value": x["metric"]} for x in m],
           "rule": "the best inner-validation metric; ties to the earlier checkpoint (settings.selection)"}
    p = fit_dir / "selection.json"
    if p.is_file() and {k: v for k, v in _read(p).items() if k != "selected_utc"} != sel:
        raise TrainError("a different selection is already recorded")
    if not p.is_file():
        _write(p, dict(sel, selected_utc=_now()))
    return sel


def _bounded(cfg, val, settings):
    c = settings["merge"]["check"]
    return val[:c["direct_rows"]] if cfg["design"] == "direct" else val[:c["parser_parents"]]


def _probe(model, cfg, rows, constraint, spec, device):
    import torch
    if cfg["design"] == "direct":
        return validate_direct(model, rows, device)
    out = []
    model.eval()
    with torch.inference_mode():
        for r in rows:
            x = torch.tensor([r["prompt_ids"]], dtype=torch.long, device=device)
            last = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu().tolist()
            res = DC.decode(DC.torch_step(model, r["prompt_ids"], device), constraint, spec["max_new_tokens"])
            out.append({"key": r["key"], "token_ids": res["token_ids"], "status": res["status"], "last": last})
    return out


def merge(*, fit_dir, prep, base_dir, out, device="cuda", settings_path=PR.SETTINGS_PATH, base_evidence=None) -> dict:
    import peft
    import torch
    from peft import PeftModel
    fit_dir, out = Path(fit_dir), Path(out)
    if out.exists():
        raise TrainError(f"{out} exists; a merged folder is written once")
    settings = PR.load_settings(settings_path)
    lock, sel = _read(fit_dir / "lock.json"), select(fit_dir)
    cfg = lock["configuration"]
    ck = fit_dir / "checkpoints" / sel["checkpoint"]
    rec = _read(ck / "checkpoint.json")
    have = ID.folder_files(ck)
    have.pop("checkpoint.json", None)
    if have != rec["files"]:
        raise TrainError(f"{ck}: its files differ from their record")
    torch_setup(device)
    base_ev, mp, desc = (base_evidence or base_identity)(base_dir)
    if _sha(json.dumps(base_ev, sort_keys=True).encode()) != lock["base"]["evidence_sha256"]:
        raise TrainError("the base folder's evidence differs from the one the fit used")
    val, _ = load_rows(prep, "validation", cfg)
    rows = _bounded(cfg, val, settings)
    spec, constraint = L.load_spec(), None
    if cfg["design"] == "parser":
        from ..preparation.iref_vla import tokens as T
        constraint = DC.Constraint(spec, DC.TokenTable(T.load_pinned_tokenizer(PR.DEV.TOKENIZER)._backend))
    t0 = time.perf_counter()
    pm = PeftModel.from_pretrained(load_base(base_dir, device), str(ck)).to(device)
    before = _probe(pm, cfg, rows, constraint, spec, device)
    merged = pm.merge_and_unload()
    merged.eval()
    after = _probe(merged, cfg, rows, constraint, spec, device)
    c = settings["merge"]["check"]
    if cfg["design"] == "direct":
        diff = max((abs(a - b) for x, y in zip(before, after) for a, b in zip(x["offered"], y["offered"])), default=0.0)
        same = [x["choice_code"] for x in before] == [y["choice_code"] for y in after]
        ok = diff <= c["max_abs_offered_logit_difference"] and same
    else:
        diff = max((abs(a - b) for x, y in zip(before, after) for a, b in zip(x["last"], y["last"])), default=0.0)
        same = [x["token_ids"] for x in before] == [y["token_ids"] for y in after]
        ok = same
    check = {"rows": len(rows), "max_abs_difference": diff, "same_decisions_or_tokens": same, "passed": ok,
             "criterion": c, "what": c["what"], "device": device}
    _write(fit_dir / "merge-check.json", check)
    if not ok:
        raise TrainError(f"the merge check failed: {check}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=str(out.parent)))
    merged.save_pretrained(str(tmp), safe_serialization=True)
    for n in TOKENIZER_FILES:
        if (Path(base_dir) / n).is_file():
            shutil.copyfile(Path(base_dir) / n, tmp / n)
    record = {"model_name": sel["model_name"], "configuration": cfg, "seed": lock["seed"], "seed_index": lock["seed_index"],
              "base": dict(lock["base"]), "tokenizer": {"pinned_files": dict(mp["tokenizer"]["files"])},
              "training": {"lock_sha256": _sha((fit_dir / "lock.json").read_bytes()), "spec_sha256": lock["spec_sha256"],
                           "settings_sha256": lock["settings_sha256"], "prep_manifest_sha256": lock["prep_manifest_sha256"],
                           "data_files": lock["data_files"], "code": lock["code"], "software": lock["software"],
                           "fit_folder": fit_dir.name},
              "selection": {k: v for k, v in sel.items() if k != "model_name"},
              "adapter": {"checkpoint": sel["checkpoint"], "files": rec["files"]},
              "merge": {"procedure": settings["merge"]["procedure"], "software": versions(), "peft": peft.__version__,
                        "check": check, "seconds": round(time.perf_counter() - t0, 2), "merged_utc": _now()}}
    _replace(tmp, out)
    ident = ID.write_identity(out, record)
    ID.load_identity(out, mp)
    return dict(ident, folder=str(out))


# --------------------------------------------------------------------------------------------- continuation (Stage B)
def continuation(*, fits, merged, runner_smokes, settings_path=PR.SETTINGS_PATH) -> dict:
    """The mechanical conditions after seed 1 of all three configurations (settings.continuation); never accuracy.
    fits / merged / runner_smokes: configuration name -> folder or file."""
    from ..inference.iref_vla.protocol import load_protocol
    from ..inference.iref_vla_compare import design as CD
    from ..inference.iref_vla_compare import run as CR
    mp, _ = CR.model_protocol(load_protocol(), CD.load_policy(), BASE_KEY)
    settings = PR.load_settings(settings_path)
    rows = []
    for c in (x["name"] for x in settings["configurations"]):
        f = Path(fits[c])
        problems = []
        try:
            done = _read(f / "done.json")
            if done["optimizer_steps"] != done["planned_optimizer_steps"]:
                problems.append("planned optimizer steps not completed")
            log = [json.loads(x) for x in (f / "log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            if len({x["step"] for x in log}) != done["planned_optimizer_steps"] or not all(math.isfinite(x["loss"]) for x in log):
                problems.append("the log lacks a step or holds a non-finite loss")
            for k in range(1, settings["epochs"] + 1):
                ck = f / "checkpoints" / f"epoch-{k}"
                files = ID.folder_files(ck)
                files.pop("checkpoint.json", None)
                if files != _read(ck / "checkpoint.json")["files"]:
                    problems.append(f"checkpoint epoch-{k} does not read back")
                v = _read(f / "validation" / f"epoch-{k}.json")
                if v["technical_failures"] or v["denominator"] != len(v["rows"]):
                    problems.append(f"validation epoch-{k} incomplete")
                if "parser" in c:
                    if any(r["status"] not in ("completed", "length_exhausted", "empty_allowed_set", "premature_eos")
                           for r in v["rows"]):
                        problems.append(f"validation epoch-{k}: an undeclared decoder outcome")
                    if any((r["status"] == "completed") != r["valid"] for r in v["rows"]):
                        problems.append(f"validation epoch-{k}: the reader disagrees with the decoder")
            meta = _read(f / "resume" / "state.json")
            if ID._sha_file(f / "resume" / "state.pt") != meta["sha256"]:
                problems.append("the resume state does not read back")
            if not _read(f / "merge-check.json")["passed"]:
                problems.append("the merge check failed")
            ID.load_identity(merged[c], mp)
            sm = _read(runner_smokes[c])
            if not sm.get("accepted") or "A2.4 fine-tuned local artifact" not in (sm.get("checkpoint_tie") or ""):
                problems.append("the accepted runner's smoke did not pass through the fine-tuned identity")
        except Exception as e:   # noqa: BLE001 - every problem is a reported refusal
            problems.append(f"{type(e).__name__}: {e}"[:300])
        rows.append({"configuration": c, "problems": problems})
    return {"format_version": 1, "record_type": "a24_continuation_check", "checked_utc": _now(),
            "passed": all(not r["problems"] for r in rows), "configurations": rows,
            "rule": settings["continuation"]}


# --------------------------------------------------------------------------------------------- the Stage A smoke
def _linear(xs, ys):
    """Least-squares (a, b) of y = a + b x; b = 0 with one distinct x."""
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    b = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 and n > 1 else 0.0
    return my - b * mx, b


def smoke(*, prep, work, base_dir, out, device="cuda", a26_index=None, settings_path=PR.SETTINGS_PATH,
          base_evidence=None, progress=print) -> dict:
    """Stage A: per configuration a 3-step fit (the longest example first), an interrupted copy resumed and compared,
    validation, selection, merge and identity; then the measured costs extrapolated to Stage B."""
    import torch
    out = Path(out)
    if out.exists():
        raise TrainError(f"{out} exists")
    out.mkdir(parents=True)
    started = _now()
    settings = PR.load_settings(settings_path)
    accum, report = settings["batch"]["accumulation"], {"started_utc": started, "device": device, "software": versions(),
                                                        "configurations": {}}
    for cfg in settings["configurations"]:
        c = cfg["name"]
        rows, _ = load_rows(prep, "fitting", cfg)
        val, _ = load_rows(prep, "validation", cfg)
        length = (lambda r: len(r["token_ids"])) if cfg["design"] == "direct" else (lambda r: len(r["prompt_ids"]) + len(r["output_ids"]))
        longest = max(rows, key=lambda r: (length(r), r["key"]))
        rest = [k for k in order([r["key"] for r in rows], settings["seeds"][0], 0) if k != longest["key"]]
        keys = [longest["key"]] + rest[:3 * accum - 1]
        limit = {"label": "Stage A smoke, not a scientific fit", "keys": keys, "steps": 3, "validation_rows": 8}
        progress(f"smoke {c}: fit")
        fit(prep=prep, work=work, config=c, seed_index=1, base_dir=base_dir, out=out / c / "fit", device=device,
            limit=limit, settings_path=settings_path, base_evidence=base_evidence, progress=progress)
        progress(f"smoke {c}: interrupted and resumed copy")
        fit(prep=prep, work=work, config=c, seed_index=1, base_dir=base_dir, out=out / c / "fit-resumed", device=device,
            limit=limit, stop_after=2, settings_path=settings_path, base_evidence=base_evidence, progress=progress)
        fit(prep=prep, work=work, config=c, seed_index=1, base_dir=base_dir, out=out / c / "fit-resumed", device=device,
            limit=limit, resume=True, settings_path=settings_path, base_evidence=base_evidence, progress=progress)
        sa = torch.load(out / c / "fit" / "resume" / "state.pt", map_location="cpu", weights_only=False)["adapter"]
        sb = torch.load(out / c / "fit-resumed" / "resume" / "state.pt", map_location="cpu", weights_only=False)["adapter"]
        resume_diff = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
        progress(f"smoke {c}: merge")
        m = merge(fit_dir=out / c / "fit", prep=prep, base_dir=base_dir, out=out / c / "merged", device=device,
                  settings_path=settings_path, base_evidence=base_evidence)
        log = [json.loads(x) for x in (out / c / "fit" / "log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        v = _read(out / c / "fit" / "validation" / "epoch-1.json")
        rep = {"steps": log, "peak_gpu_bytes": max((x["peak_gpu_bytes"] or 0) for x in log) or None,
               "resume_max_abs_adapter_difference": resume_diff, "longest_example_tokens": length(longest),
               "validation": {k: v[k] for k in ("metric", "denominator", "seconds")},
               "validation_ms_per_row": statistics.fmean(r["ms"] if "ms" in r else r["decode_ms"] + r["resolve_ms"]
                                                         for r in v["rows"]),
               "merge_seconds": m["merge"]["seconds"], "merge_check": m["merge"]["check"], "model_name": m["model_name"]}
        if cfg["design"] == "direct":
            rep["forward_ms_model"] = list(_linear([r["input_tokens"] for r in v["rows"]], [r["ms"] for r in v["rows"]]))
        if cfg["design"] == "parser":
            rep["decode"] = {"mask_ms_mean": statistics.fmean(r["mask_ms"] for r in v["rows"]),
                             "model_ms_mean": statistics.fmean(r["model_ms"] for r in v["rows"]),
                             "steps_mean": statistics.fmean(r["steps"] for r in v["rows"]),
                             "resolve_ms_mean": statistics.fmean(r["resolve_ms"] for r in v["rows"]),
                             "statuses": sorted({r["status"] for r in v["rows"]})}
        per_example_s = [x["seconds"] / x["examples"] for x in log]
        tokens = [x["tokens"] / x["examples"] for x in log]
        a0, b0 = _linear(tokens, per_example_s)
        n_tokens = sum(length(r) for r in rows)
        rep["estimate"] = {"seconds_per_example_model": [a0, b0],
                           "fit_hours": (settings["epochs"] * (a0 * len(rows) + b0 * n_tokens)) / 3600,
                           "validation_hours": settings["epochs"] * len(val) * rep["validation_ms_per_row"] / 3.6e6,
                           "examples": len(rows), "validation_rows": len(val)}
        report["configurations"][c] = rep
    report["stage_b"] = stage_b_estimate(report, settings, a26_index)
    report["finished_utc"] = _now()
    _write(out / "smoke.json", report)
    (out / "smoke-report.md").write_text(smoke_markdown(report), encoding="utf-8")
    return report


def stage_b_estimate(report, settings, a26_index) -> dict:
    """Nine fits and nine development evaluations, from the smoke's measurements (planning estimates)."""
    seeds = len(settings["seeds"])
    cf = report["configurations"]
    fits = {c: seeds * (x["estimate"]["fit_hours"] + x["estimate"]["validation_hours"] + x["merge_seconds"] / 3600)
            for c, x in cf.items()}
    direct = [x for c, x in cf.items() if c.startswith("direct")]
    a_ms = statistics.fmean(x["forward_ms_model"][0] for x in direct)
    b_ms = statistics.fmean(x["forward_ms_model"][1] for x in direct)
    if a26_index:
        toks = [json.loads(x)["input_tokens"] for x in Path(a26_index).read_text(encoding="utf-8").splitlines() if x.strip()]
        dev_h = math.fsum(a_ms + b_ms * t for t in toks) / 3.6e6
        basis = (f"{len(toks)} frozen requests; forward time = a + b x tokens, fitted on the smoke's validation forwards "
                 f"(a = {a_ms:.2f} ms, b = {b_ms:.4f} ms per token)")
    else:
        dev_h = 10860 * statistics.fmean(x["validation_ms_per_row"] for x in direct) / 3.6e6
        basis = "10,860 requests at the smoke's mean validation forward time (no A2.6 index passed)"
    p = cf.get("parser", {})
    par_h = (2715 * (p.get("decode", {}).get("model_ms_mean", 0) + p.get("decode", {}).get("mask_ms_mean", 0))
             + 5430 * p.get("decode", {}).get("resolve_ms_mean", 0)) / 3.6e6
    total = sum(fits.values()) + 2 * seeds * dev_h + seeds * par_h
    return {"fits_hours": fits, "direct_evaluation_hours_each": dev_h, "direct_evaluations": 2 * seeds,
            "parser_evaluation_hours_each": par_h, "parser_evaluations": seeds, "total_hours": total, "basis": basis,
            "status": "planning estimate from Stage A measurements; Stage B records its own UTC start and finish"}


def smoke_markdown(r) -> str:
    L_ = ["# A2.4 Stage A smoke (separate from the scientific fits)", "",
          f"Started {r['started_utc']}, finished {r['finished_utc']}; device {r['device']}; software {json.dumps(r['software'])}.", ""]
    for c, x in r["configurations"].items():
        L_ += [f"## {c}", "",
               f"- Steps: {len(x['steps'])}; seconds per step {[s['seconds'] for s in x['steps']]}; tokens per step "
               f"{[s['tokens'] for s in x['steps']]}; losses {[round(s['loss'], 4) for s in x['steps']]}",
               f"- Peak GPU memory: {x['peak_gpu_bytes']} bytes (the longest example, {x['longest_example_tokens']} tokens, "
               "in the first step)",
               f"- Interrupted at step 2 and resumed: largest adapter difference from the uninterrupted fit "
               f"{x['resume_max_abs_adapter_difference']:.3g}",
               f"- Validation (8 rows): {x['validation']}; {x['validation_ms_per_row']:.1f} ms per row",
               f"- Merge: {x['merge_seconds']} s; check {json.dumps(x['merge_check'])}",
               f"- Estimate: {json.dumps(x['estimate'])}"]
        if "decode" in x:
            L_.append(f"- Decoding: {json.dumps(x['decode'])}")
        L_.append("")
    L_ += ["## Stage B (planning estimate)", "", "```", json.dumps(r["stage_b"], indent=1), "```", ""]
    return "\n".join(L_) + "\n"
