"""Publishing a finished folder: rename its staging folder into place.

On Windows, renaming a folder that holds a just-written file can fail for a moment with PermissionError (WinError 5 or
32), as it did twice in this project's tests. A transient lock is the likely reason; a virus scanner or the search
indexer is suspected, not established. The rename is retried briefly before giving up, which waits about nine seconds
at most. Nothing is copied or overwritten: either the whole folder appears under its
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
