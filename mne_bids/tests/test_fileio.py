"""Test file I/O helpers with file locking support."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

import mne_bids._fileio
from mne_bids._fileio import (
    _canonical_lock_path,
    _get_lock_context,
    _open_lock,
)


@contextmanager
def _readonly(directory):
    """Temporarily strip write bit from ``directory``."""
    original = directory.stat().st_mode
    os.chmod(directory, 0o555)
    try:
        yield directory
    finally:
        os.chmod(directory, original)


def _worker_increment_file(file_path, iterations):
    """Worker function for multiprocess test - must be at module level for pickling."""
    file_path = Path(file_path)
    for _ in range(iterations):
        with _open_lock(file_path, "r+") as fid:
            value = int(fid.read().strip())
            fid.seek(0)
            fid.write(str(value + 1))
            fid.truncate()
    return True


def _access_file_with_lock(thread_id, test_file, access_log, access_lock, errors):
    """Access file with lock to read and log - extracted for flatness."""
    try:
        for i in range(3):
            with _open_lock(test_file, "r+") as fid:
                # Read content
                content = fid.read()

                # Write thread ID and iteration
                fid.seek(0)
                new_content = f"thread-{thread_id}-iter-{i}"
                fid.write(new_content)
                fid.truncate()

                # Log this access
                with access_lock:
                    access_log.append((thread_id, i, content, new_content))

                # Small delay to encourage contention
                time.sleep(0.001)
    except Exception as e:
        with access_lock:
            errors.append((thread_id, str(e)))


def _mock_filelock_oserror(*args, **kwargs):
    """Mock FileLock that raises OSError."""
    raise OSError("Permission denied")


def _mock_filelock_typeerror(*args, **kwargs):
    """Mock FileLock that raises TypeError."""
    raise TypeError("Invalid timeout")


def test_canonical_lock_path(tmp_path):
    """Test path canonicalization."""
    # Test with string path
    path_str = str(tmp_path / "test.txt")
    result = _canonical_lock_path(path_str)
    assert isinstance(result, Path)
    assert result.is_absolute()

    # Test with Path object
    path_obj = tmp_path / "test.txt"
    result = _canonical_lock_path(path_obj)
    assert isinstance(result, Path)
    assert result.is_absolute()

    # Test with expanduser
    home_path = "~/test.txt"
    result = _canonical_lock_path(home_path)
    assert "~" not in str(result)

    # Test that it doesn't require file to exist
    nonexistent = tmp_path / "nonexistent" / "test.txt"
    result = _canonical_lock_path(nonexistent)
    assert isinstance(result, Path)


def test_get_lock_context_without_filelock(tmp_path, monkeypatch):
    """Test lock context when filelock is not available."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    # Mock _soft_import to return None (filelock not available)
    with patch("mne_bids._fileio._soft_import", return_value=None):
        with _get_lock_context(test_file) as lock_ctx:
            # Should get a nullcontext
            assert lock_ctx is not None


def test_get_lock_context_with_filelock(tmp_path):
    """Test lock context when filelock is available."""
    pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with _get_lock_context(test_file, timeout=5) as lock_ctx:
        # Should get a FileLock instance
        assert lock_ctx is not None
        # Lock file is created by filelock when acquired, not before
        # So we need to manually acquire it to test
        if hasattr(lock_ctx, "acquire"):
            lock_ctx.acquire()
            lock_file = test_file.parent / f"{test_file.name}.lock"
            assert lock_file.exists()
            lock_ctx.release()


def test_get_lock_context_oserror(tmp_path):
    """Test lock context when OSError occurs."""
    filelock = pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with patch.object(filelock, "FileLock", side_effect=_mock_filelock_oserror):
        with pytest.warns(RuntimeWarning, match="Could not create lock"):
            with _get_lock_context(test_file) as lock_ctx:
                assert lock_ctx is not None


def test_get_lock_context_typeerror(tmp_path):
    """Test lock context when TypeError occurs (invalid timeout)."""
    filelock = pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with patch.object(filelock, "FileLock", side_effect=_mock_filelock_typeerror):
        with pytest.warns(RuntimeWarning, match="Could not create lock"):
            with _get_lock_context(test_file) as lock_ctx:
                assert lock_ctx is not None


def test_open_lock_basic(tmp_path):
    """Test basic lock acquisition and file opening."""
    pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"
    test_file.write_text("initial content")

    # Test with file opening
    with _open_lock(test_file, "r") as fid:
        content = fid.read()
        assert content == "initial content"

    # Test without file opening (just lock)
    with _open_lock(test_file) as fid:
        assert fid is None


def test_open_lock_write(tmp_path):
    """Test lock with file writing."""
    pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"

    with _open_lock(test_file, "w") as fid:
        fid.write("new content")

    assert test_file.read_text() == "new content"


def test_open_lock_reentrant(tmp_path):
    """Test re-entrant lock behavior."""
    pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    # Nested lock calls should work
    with _open_lock(test_file, "r") as fid1:
        assert fid1 is not None
        with _open_lock(test_file, "r", lock=False) as fid2:
            assert fid2 is not None
            # Both should be able to read
            content1 = fid1.read()
            fid2.seek(0)
            content2 = fid2.read()
            assert content1 == content2 == "test"


def test_open_lock_multithread(tmp_path):
    """Test lock with multiple threads - verify locks prevent corruption."""
    pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"
    test_file.write_text("initial content")

    # Track thread access
    access_log = []
    access_lock = threading.Lock()
    errors = []

    # Run with 4 threads, each accessing 3 times
    num_threads = 4
    threads = [
        threading.Thread(
            target=_access_file_with_lock,
            args=(i, test_file, access_log, access_lock, errors),
        )
        for i in range(num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Should have 12 accesses total (4 threads × 3 iterations)
    assert len(access_log) == 12, f"Expected 12 accesses, got {len(access_log)}"

    # Verify the file contains valid content from one of the threads
    final_content = test_file.read_text()
    assert final_content.startswith("thread-"), (
        f"Final content should be from a thread: {final_content}"
    )

    # The key test: verify that no partial writes occurred
    # All content read should be complete strings, not corrupted
    for thread_id, iteration, content_read, content_written in access_log:
        # Content read should either be initial or from a complete thread write
        assert content_read == "initial content" or content_read.startswith(
            "thread-"
        ), (
            f"Thread {thread_id} iter {iteration} read corrupted "
            f"content: {content_read!r}"
        )


def test_open_lock_without_filelock(tmp_path):
    """Test lock behavior when filelock is not available."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with patch("mne_bids._fileio._soft_import", return_value=None):
        with _open_lock(test_file, "r") as fid:
            content = fid.read()
            assert content == "test"


def test_open_lock_symlink_to_readonly_target(tmp_path):
    """Regression test for issue #1569 (datalad/git-annex symlinks).

    When the symlink target lives in a read-only directory, the lock must
    be placed next to the symlink itself rather than resolved through it.
    Previously this raised ``PermissionError``.
    """
    pytest.importorskip("filelock")

    annex_dir = tmp_path / "annex"
    annex_dir.mkdir()
    target = annex_dir / "blob.json"
    target.write_text('{"k": 2}')

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    link = work_dir / "sidecar.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    with _readonly(annex_dir):
        with _open_lock(link, encoding="utf-8") as fin:
            assert fin.read() == '{"k": 2}'
        assert not (annex_dir / "blob.json.lock").exists()
        assert not (work_dir / "sidecar.json.lock").exists()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX chmod 0o555 does not restrict directory writes on Windows",
)
def test_open_lock_readonly_parent_falls_back(tmp_path):
    """If the lock file cannot be created, warn and fall back gracefully."""
    pytest.importorskip("filelock")

    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    test_file = ro_dir / "file.json"
    test_file.write_text("payload")

    with _readonly(ro_dir):
        with pytest.warns(RuntimeWarning, match="without a lock"):
            with _open_lock(test_file, "r", encoding="utf-8") as fid:
                assert fid.read() == "payload"


def test_open_lock_custom_timeout(tmp_path):
    """Test custom timeout parameter."""
    pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with _open_lock(test_file, lock_timeout=10) as fid:
        assert fid is None


def test_lock_refcount_multiprocess(tmp_path):
    """Test refcount with multiple processes."""
    pytest.importorskip("filelock")

    test_file = tmp_path / "test.txt"
    test_file.write_text("0")

    # Run with multiple processes
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_worker_increment_file, str(test_file), 5) for _ in range(4)
        ]
        results = [f.result() for f in futures]

    assert all(results)

    # Should have incremented 20 times (4 processes × 5 iterations)
    final_value = int(test_file.read_text())
    assert final_value == 20

    # Lock files should be cleaned up
    lock_file = tmp_path / f"{test_file.name}.lock"
    refcount_file = tmp_path / f"{test_file.name}.lock.refcount"

    # Wait for lock files to be cleaned up (may take longer on slow CI)
    for _ in range(20):  # up to 10 seconds
        if not lock_file.exists() or not refcount_file.exists():
            break
        time.sleep(0.5)

    assert not lock_file.exists() or not refcount_file.exists()


def test_environment_variable_timeout(tmp_path, monkeypatch):
    """Test DEFAULT_LOCK_TIMEOUT respects environment variable."""
    # This test checks the module-level initialization
    # We need to reload the module with the env var set
    monkeypatch.setenv("MNE_BIDS_FILELOCK_TIMEOUT", "120")

    importlib.reload(mne_bids._fileio)

    assert mne_bids._fileio.DEFAULT_LOCK_TIMEOUT == 120.0


def test_environment_variable_invalid_timeout(tmp_path, monkeypatch):
    """Test DEFAULT_LOCK_TIMEOUT with invalid environment variable."""
    monkeypatch.setenv("MNE_BIDS_FILELOCK_TIMEOUT", "invalid")

    importlib.reload(mne_bids._fileio)

    # Should fall back to default
    assert mne_bids._fileio.DEFAULT_LOCK_TIMEOUT == 60.0
