"""Optional numba acceleration.

On desktop, numba compiles the per-sample dynamics loops. Where numba can't be installed
(Termux / Android, some ARM boards) the same kernels run as plain Python at a reduced
control rate (see dynamics.py), which is still fast enough for full songs.
Set STUDIOMIX_PURE_PYTHON=1 to force the fallback.
"""

from __future__ import annotations

import os

HAS_NUMBA = False
if os.environ.get("STUDIOMIX_PURE_PYTHON", "") not in ("1", "true", "yes"):
    try:
        from numba import njit

        HAS_NUMBA = True
    except Exception:  # ImportError, or a broken llvmlite build
        HAS_NUMBA = False


def kernel(fn):
    """Compile with numba when available, otherwise return the plain Python function."""
    if HAS_NUMBA:
        return njit(cache=True)(fn)
    return fn
