"""Test for the tsv handling functions."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from collections import OrderedDict as odict

import pytest

import mne_bids._fileio as _fileio
from mne_bids.tsv_handler import (
    _combine_rows,
    _contains_row,
    _drop,
    _from_compressed_tsv,
    _from_tsv,
    _to_tsv,
    _tsv_to_str,
)


def test_tsv_handler(tmp_path):
    """Test the TSV handling."""
    # create some dummy data
    d = odict(a=[1, 2, 3, 4], b=["five", "six", "seven", "eight"])
    assert _contains_row(d, {"a": 1, "b": "five"})
    d2 = odict(a=[5], b=["nine"])
    d = _combine_rows(d, d2)
    assert 5 in d["a"]
    d2 = odict(a=[5])
    d = _combine_rows(d, d2)
    assert "n/a" in d["b"]
    d2 = odict(a=[5], b=["ten"])
    d = _combine_rows(d, d2, drop_column="a")
    # make sure that the repeated data was dropped
    assert "nine" not in d["b"]
    print(_tsv_to_str(d))

    d_path = tmp_path / "output.tsv"

    # write the data to an output tsv file
    _to_tsv(d, d_path)
    # now read it back
    d = _from_tsv(d_path)
    # test reading the file in with the incorrect number of datatypes raises
    # an Error
    with pytest.raises(ValueError):
        d = _from_tsv(d_path, dtypes=[str])
    # we can also pass just a single data type and it will be applied to all
    # columns
    d = _from_tsv(d_path, str)

    # remove any rows with 2 or 5 in them
    d = _drop(d, [2, 5], "a")
    assert 2 not in d["a"]

    # test combining data with differing numbers of columns
    d = odict(a=[1, 2], b=["three", "four"])
    d2 = odict(a=[4], b=["five"], c=[3.1415])
    # raise error if a new column is tried to be added
    with pytest.raises(KeyError):
        d = _combine_rows(d, d2)
    d2 = odict(a=[5])
    d = _combine_rows(d, d2)
    assert d["b"] == ["three", "four", "n/a"]
    assert _contains_row(d, {"a": 5})

    # test reading a single column
    _to_tsv(odict(a=[1, 2, 3, 4]), d_path)
    d = _from_tsv(d_path)
    assert d["a"] == ["1", "2", "3", "4"]

    # test an empty tsv (just headers)
    _to_tsv(odict(onset=[], duration=[], trial_type=[]), d_path)
    with pytest.warns(RuntimeWarning, match="TSV file is empty"):
        d = _from_tsv(d_path)
    d = _drop(d, "n/a", "trial_type")


def test_to_tsv_without_filelock(monkeypatch, tmp_path):
    """Ensure TSV writes succeed when filelock is unavailable."""
    data = odict(a=[1, 2], b=["five", "six"])
    tsv_path = tmp_path / "file.tsv"
    lock_path = tsv_path.parent / f"{tsv_path.name}.lock"
    refcount_path = tsv_path.parent / f"{tsv_path.name}.lock.refcount"

    monkeypatch.setattr(_fileio, "_soft_import", lambda *args, **kwargs: False)

    _to_tsv(data, tsv_path)

    assert tsv_path.exists()
    assert tsv_path.read_text().strip()
    assert not lock_path.exists()
    assert not refcount_path.exists()


def test_write_compressed_tsv(tmp_path):
    """Ensure compressed TSV files are headerless."""
    data = dict(onset=[0.1, 0.2], duration=[1, 2], trial_type=["a", "b"])
    tsv_path = tmp_path / "physioevents.tsv.gz"

    _to_tsv(data, tsv_path, compress=True)

    assert tsv_path.exists()
    parsed = _from_tsv(tsv_path)
    assert list(parsed.keys()) == ["column_0", "column_1", "column_2"]
    assert parsed["column_0"] == ["0.1", "0.2"]
    assert parsed["column_1"] == ["1", "2"]
    assert parsed["column_2"] == ["a", "b"]


def test_read_compressed_tsv(tmp_path):
    """Compressed TSV reader should get column names from sidecar JSON."""
    data = dict(onset=[0.1, 0.2], duration=[1, 2], trial_type=["a", "b"])
    tsv_path = tmp_path / "physioevents.tsv.gz"
    json_path = tmp_path / "physioevents.json"

    _to_tsv(data, tsv_path, compress=True)
    # Dummy file
    json_path.write_text(
        '{"Columns": ["onset", "duration", "trial_type"]}', encoding="utf-8-sig"
    )

    parsed = _from_compressed_tsv(tsv_path)
    assert list(parsed.keys()) == ["onset", "duration", "trial_type"]
    assert parsed["onset"] == ["0.1", "0.2"]
    assert parsed["duration"] == ["1", "2"]
    assert parsed["trial_type"] == ["a", "b"]


def test_contains_row_different_types():
    """Test that _contains_row() can handle different dtypes without warning.

    This is to check if we're successfully avoiding a FutureWarning emitted by
    NumPy, see https://github.com/mne-tools/mne-bids/pull/372
    (pytest must be configured to fail on warnings for this to work!)
    """
    data = odict(age=[20, 30, 40, "n/a"])  # string
    row = dict(age=60)  # int
    _contains_row(data, row)


def test_drop_different_types():
    """Test that _drop() can handle different dtypes without warning.

    This is to check if we're successfully avoiding a FutureWarning emitted by
    NumPy, see https://github.com/mne-tools/mne-bids/pull/372
    (pytest must be configured to fail on warnings for this to work!)
    """
    column = "age"
    data = odict([(column, [20, 30, 40, "n/a"])])  # string
    values_to_drop = (20,)  # int

    result = _drop(data, values=values_to_drop, column=column)
    for value in values_to_drop:
        assert value not in result
