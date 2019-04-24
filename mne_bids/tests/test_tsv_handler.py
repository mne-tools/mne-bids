"""Test for the tsv handling functions."""
# Authors: Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)


import os.path as op
from collections import OrderedDict as odict
from mne_bids.tsv_handler import (_from_tsv, _to_tsv, _combine, _drop,
                                  _contains_row, _tsv_to_str)
import pytest

from mne.utils import _TempDir


def test_tsv_handler():
    """Test the TSV handling."""
    # create some dummy data
    d = odict(a=[1, 2, 3, 4], b=['five', 'six', 'seven', 'eight'])
    assert _contains_row(d, {'a': 1, 'b': 'five'})
    d2 = odict(a=[5], b=['nine'])
    d = _combine(d, d2)
    assert 5 in d['a']
    d2 = odict(a=[5])
    d = _combine(d, d2)
    assert 'n/a' in d['b']
    d2 = odict(a=[5], b=['ten'])
    d = _combine(d, d2, drop_column='a')
    # make sure that the repeated data was dropped
    assert 'nine' not in d['b']
    print(_tsv_to_str(d))

    tempdir = _TempDir()
    d_path = op.join(tempdir, 'output.tsv')

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
    d = _drop(d, [2, 5], 'a')
    assert 2 not in d['a']

    # test combining data with differing numbers of columns
    d = odict(a=[1, 2], b=['three', 'four'])
    d2 = odict(a=[4], b=['five'], c=[3.1415])
    # raise error if a new column is tried to be added
    with pytest.raises(KeyError):
        d = _combine(d, d2)
    d2 = odict(a=[5])
    d = _combine(d, d2)
    assert d['b'] == ['three', 'four', 'n/a']
    assert _contains_row(d, {'a': 5})
