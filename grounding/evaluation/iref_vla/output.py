"""Publishing an A2.2b output folder (D75): never over an existing path; every file is written into a new sibling
folder, which is renamed into place only once all are written. Any failure removes it, so nothing that looks complete
is left behind."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .protocol import EvaluationInputError, EvaluationOutputError, issue


def _write_file(path: Path, data: bytes) -> None:
    """The one place an output file is written."""
    with open(path, "xb") as f:
        f.write(data)


def refuse_existing(out) -> Path:
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise EvaluationInputError([issue(str(out), "E_EVAL_OUTPUT_EXISTS",
                                          "the output folder already exists; choose a new one (nothing is overwritten)")])
    return out


def publish(out, files: dict) -> None:
    out = refuse_existing(out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    except OSError as e:
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    try:
        for rel in files:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_file(target, files[rel])
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO",
                                           f"{type(e).__name__}: {e}; nothing was published")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
