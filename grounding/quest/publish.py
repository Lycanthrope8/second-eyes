"""Publishing a finished folder: rename its staging folder into place.

On Windows a virus scanner or the search indexer can hold a just-written file open for a moment, and renaming the
folder that holds it then fails with PermissionError (WinError 5 or 32). The rename is retried briefly before giving up,
which waits about nine seconds at most. Nothing is copied or overwritten: either the whole folder appears under its
final name, or the error is raised and the staging folder is handled by the caller as before.
"""
from __future__ import annotations

import os
import time


def publish(staging, out, attempts: int = 10, wait: float = 0.2, sleep=time.sleep) -> int:
    """Rename staging to out; returns how many attempts it took."""
    for k in range(1, attempts + 1):
        try:
            os.rename(staging, out)
            return k
        except PermissionError:
            if k == attempts:
                raise
            sleep(wait * k)
    return attempts
