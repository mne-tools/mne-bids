"""File I/O helpers with file locking support."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import inspect
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

from mne.utils import _soft_import, logger, warn

_LOCK_TIMEOUT_FALLBACK = 60.0
_READ_MODES = (None, "r", "rt", "rb")
_REPLACE_MODES = ("w", "wt", "wb")
_HELD = threading.local()  # lock files held by this thread
_env_lock_timeout = os.getenv("MNE_BIDS_FILELOCK_TIMEOUT", "")
try:
    DEFAULT_LOCK_TIMEOUT = (
        float(_env_lock_timeout) if _env_lock_timeout else _LOCK_TIMEOUT_FALLBACK
    )
    if DEFAULT_LOCK_TIMEOUT <= 0:
        raise ValueError
except ValueError:
    DEFAULT_LOCK_TIMEOUT = _LOCK_TIMEOUT_FALLBACK


@contextmanager
def _hold(lock, key):
    """Acquire ``lock``, so that nested calls in this thread can reuse it."""
    held = _HELD.__dict__.setdefault("keys", set())
    with lock:
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)


def _canonical_lock_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute, normalised path without following symlinks.

    Symlinks are preserved so that the ``.lock`` companion file is written
    next to the symlink itself rather than its target. This matters for
    datalad/git-annex datasets, where the symlink target lives in a
    read-only ``.git/annex/objects/`` directory (see issue #1569).
    """
    path = Path(path)
    return Path(os.path.abspath(path.expanduser()))


@contextmanager
def _get_lock_context(path, *, timeout=None, lock=True):
    """Get a file lock context for the given path.

    Internal helper function that creates a FileLock if available,
    or returns a nullcontext() as fallback.

    Parameters
    ----------
    path : str
        The path to acquire a lock for.
    timeout : float
        Timeout in seconds for acquiring the lock. If ``None``, the value of
        ``DEFAULT_LOCK_TIMEOUT`` (default 60 seconds) is used. The timeout can
        be overridden via the ``MNE_BIDS_FILELOCK_TIMEOUT`` environment
        variable.

    Yields
    ------
    context : context manager
        Either a FileLock or nullcontext.
    """
    if timeout is None:
        timeout = DEFAULT_LOCK_TIMEOUT

    canonical_path = _canonical_lock_path(path)

    filelock = _soft_import(
        "filelock",
        purpose="parallel file I/O locking",
        strict=False,
        min_version="3.20.4",
    )

    lock_path = canonical_path.with_name(f"{canonical_path.name}.lock")
    lock_context = contextlib.nullcontext()

    where = None
    stack = "unknown"
    try:  # this should always work but let's be safe
        # [0] = here
        # [1] = contextlib __enter__
        # [2] = _open_lock
        # [3] = contextlib __enter__
        # [4] = caller of _open_lock
        # Using inspect.stack is expensive, so traverse directly instead
        where = inspect.currentframe().f_back.f_back.f_back.f_back
        stack = f"{where.f_code.co_filename}:{where.f_lineno} {where.f_code.co_name}"
    except Exception:
        pass
    finally:
        del where
    logger.debug(f"Lock: acquiring {canonical_path} from {stack}")

    key = (os.getpid(), lock_path)
    if lock and filelock and key not in getattr(_HELD, "keys", ()):
        try:
            # Ensure parent directory exists
            Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
            lock_context = _hold(
                filelock.SoftFileLock(str(lock_path), timeout=timeout), key
            )
        except (OSError, TypeError) as exp:
            # OSError: permission issues creating lock file
            # TypeError: invalid timeout parameter
            warn(
                f"Could not create lock at {lock_path} ({exp}); "
                "proceeding without a lock."
            )
    try:
        yield lock_context
    except Exception:
        logger.debug(f"Lock: exception {canonical_path} from {stack}")
        raise
    finally:
        logger.debug(f"Lock: released  {canonical_path} from {stack}")


def _has_text(path, text, encoding="utf-8"):
    """Whether ``path`` already holds exactly ``text``."""
    try:
        with open(path, encoding=encoding, newline="") as fid:
            return fid.read() == text.replace("\n", os.linesep)
    except (OSError, UnicodeError):
        return False


@contextmanager
def _replace_on_exit(path, target, fid, tmp, *, timeout=None, lock=True):
    """Move ``tmp`` over ``target`` (``path`` or its symlink target) once written.

    The new file keeps the permissions of the one it replaces. A target that
    cannot be replaced (open elsewhere, on Windows) is overwritten under the lock.
    """
    try:
        with fid:
            yield fid
        with contextlib.suppress(OSError):  # nothing to replace yet
            st = os.stat(target)
            if hasattr(os, "chown"):
                uid = st.st_uid if os.geteuid() == 0 else -1
                gid = st.st_gid if st.st_gid != os.stat(tmp).st_gid else -1
                if uid != -1 or gid != -1:
                    with contextlib.suppress(OSError):
                        os.chown(tmp, uid, gid)
            shutil.copymode(target, tmp)
        try:
            os.replace(tmp, target)
        except PermissionError:
            with (
                _get_lock_context(path, timeout=timeout, lock=lock) as lock_context,
                lock_context,
            ):
                shutil.copyfile(tmp, target)
            os.unlink(tmp)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextmanager
def _open_lock(path, *args, lock_timeout=None, lock=True, **kwargs):
    """Like :func:`open_lock`, but reads and ``"w"`` writes take no lock.

    Writes replace the file in one step, so readers never see a partial file.
    """
    canonical_path = _canonical_lock_path(path)
    mode = args[0] if args else kwargs.get("mode")
    # On Windows an open file cannot be replaced, so reads keep the lock there.
    if (args or kwargs) and mode in _READ_MODES and os.name != "nt":
        with open(canonical_path, *args, **kwargs) as fid:
            yield fid
        return
    # A read-only directory (git-annex, #1569) falls back to writing under the lock.
    if mode in _REPLACE_MODES:
        target = canonical_path
        if target.is_symlink():
            target = target.resolve()
        tmp = target.with_name(f".{target.name}.{os.urandom(4).hex()}.tmp")
        try:
            fid = open(tmp, *args, **kwargs)
        except PermissionError:
            pass
        else:
            with _replace_on_exit(
                canonical_path, target, fid, tmp, timeout=lock_timeout, lock=lock
            ):
                yield fid
            return

    with _get_lock_context(
        canonical_path,
        timeout=lock_timeout,
        lock=lock,
    ) as lock_context:
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(lock_context)
            except OSError as exp:
                warn(
                    f"Could not acquire lock for {canonical_path} "
                    f"({exp}); proceeding without a lock."
                )
            if args or kwargs:
                fid = stack.enter_context(open(canonical_path, *args, **kwargs))
                yield fid
            else:
                yield None


@contextmanager
def open_lock(path, *args, lock_timeout=None, lock=True, **kwargs):
    """Context manager that acquires a file lock with optional file opening.

    If the ``filelock`` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock') and held for the whole
    ``with`` block. The lock file is removed when the lock is released.

    If file opening arguments (``*args``, ``**kwargs``) are provided, the file is
    opened in the specified mode. Otherwise, just the lock is acquired. A file
    opened with ``"w"`` or ``"wb"`` is written to a temporary file and moved into
    place at the end of the block, so readers never see it partially written.

    The lock is re-entrant per thread: a nested call for the same ``path`` reuses
    the lock instead of waiting for it. To read a file and then write it, do
    both inside one ``with open_lock(path):`` block.

    Parameters
    ----------
    path : str
        The path to acquire a lock for (and optionally open).
    *args : tuple
        Additional positional arguments forwarded to ``open``.
    lock_timeout : float | None
        Timeout in seconds for acquiring the lock. If ``None``, the default
        timeout applies.
    lock : bool
        If True, use a FileLock (if available) to lock the file.
    **kwargs : dict
        Additional keyword arguments forwarded to ``open``.

    Yields
    ------
    fid : io.IOBase | None
        File object if file opening args were provided, None otherwise.
    """
    with _open_lock(path, lock_timeout=lock_timeout, lock=lock):
        if args or kwargs:
            with _open_lock(path, *args, lock=False, **kwargs) as fid:
                yield fid
        else:
            yield None


def _chmod_rw_R(path):
    assert os.path.isdir(path), f"Expected a directory, got {path}"
    os.chmod(path, os.stat(path).st_mode | 0o770)
    for root, dirs, files in os.walk(path):
        for name in files:
            os.chmod(os.path.join(root, name), 0o660)
        for name in dirs:
            os.chmod(os.path.join(root, name), 0o770)
