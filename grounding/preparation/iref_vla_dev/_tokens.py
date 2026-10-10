"""The pinned tokenizer, loaded as A2.2d loads it (its pin check included)."""
from __future__ import annotations

from ..iref_vla import tokens as T


def load(tokenizer_dir):
    return T.load_pinned_tokenizer(tokenizer_dir)
