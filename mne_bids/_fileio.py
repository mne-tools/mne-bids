"""File I/O helpers with automatic cleanup of lock files."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from mne.utils.config import _open_lock as _mne_open_lock


@contextmanager
def _open_lock(path, *args, **kwargs):
    """Wrap :func:`mne.utils.config._open_lock` and remove stale ``.lock`` files."""
    lock_path = Path(f"{os.fspath(path)}.lock")
    try:
        with _mne_open_lock(path, *args, **kwargs) as fid:
            yield fid
    finally:
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
