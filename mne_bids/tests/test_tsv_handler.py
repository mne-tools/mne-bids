"""Test for the tsv handling functions."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import codecs
from collections import OrderedDict as odict

import pytest

import mne_bids._fileio as _fileio
from mne_bids.tsv_handler import (
    _combine_rows,
    _contains_row,
    _detect_tsv_encoding,
    _drop,
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


def test_contains_row_different_types():
    """Test that _contains_row() can handle different dtypes without warning.

    This is to check if we're successfully avoiding a FutureWarning emitted by
    NumPy, see https://github.com/mne-tools/mne-bids/pull/372
    (pytest must be configured to fail on warnings for this to work!)
    """
    data = odict(age=[20, 30, 40, "n/a"])  # string
    row = dict(age=60)  # int
    _contains_row(data, row)


@pytest.mark.parametrize(
    "payload,expected",
    [
        (b"name\tunit\nEEG\tuV\n", "utf-8"),
        ("name\tunit\nEEG\tµV\n".encode(), "utf-8"),
        (codecs.BOM_UTF8 + b"name\tunit\nEEG\tuV\n", "utf-8-sig"),
        ("name\tunit\nEEG\tµV\n".encode("utf-16"), "utf-16"),
        (b"name\tunit\nEEG\t\xb5V\n", "latin-1"),
    ],
    ids=["ascii", "utf8", "utf8-bom", "utf16", "latin1"],
)
def test_detect_tsv_encoding(tmp_path, payload, expected):
    """Encoding is detected deterministically from BOM and UTF-8 validity."""
    tsv = tmp_path / "test.tsv"
    tsv.write_bytes(payload)
    assert _detect_tsv_encoding(tsv) == expected


def test_from_tsv_latin1_warns(tmp_path):
    """``_from_tsv`` reads non-UTF-8 TSV files as latin-1 with a warning."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_bytes(b"name\tunit\nEEG\t\xb5V\n")  # 'µV' in latin-1
    with pytest.warns(RuntimeWarning, match="not UTF-8"):
        d = _from_tsv(tsv)
    assert d["unit"] == ["µV"]


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
