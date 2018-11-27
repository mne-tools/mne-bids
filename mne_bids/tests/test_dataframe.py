"""Test for the MockDataFrame object"""
# Authors: Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)

import os.path as op
from collections import OrderedDict as odict
from mne_bids.dataframe_func import (odict_to_ndarray, from_tsv, to_tsv,
                                     combine_data, drop, is_contained,
                                     column_data)
import pytest

from mne.utils import _TempDir


def test_dataframe():
    # create df
    d = odict([('a', [1, 2, 3, 4]), ('b', [5, 6, 7, 8])])
    df = odict_to_ndarray(d)
    # create another
    df2 = odict_to_ndarray(odict([('a', [1]), ('b', [5])]))
    assert is_contained(df, df2)
    df2 = odict_to_ndarray(odict([('a', [5]), ('b', [9])]))
    df = combine_data(df, df2)
    assert '5' in column_data(df, 'a')
    df2 = odict_to_ndarray(odict([('a', [5]), ('b', [10])]))
    df = combine_data(df, df2, drop_column='a')
    assert '9' not in column_data(df, 'b')

    tempdir = _TempDir()
    df_path = op.join(tempdir, 'output.tsv')

    # write the MockDataFrame to an output tsv file
    to_tsv(df, df_path)
    # now read it back
    df = from_tsv(df_path)

    # remove any rows with 2 or 5 in them
    df = drop(df, [2, 5], 'a')
    assert '2' not in column_data(df, 'a')
    #assert not is_contained(df, ['5', '10'])
    #assert np.array(['3', '7']) in df
    df = drop(df, [], 'a')
    with pytest.raises(ValueError):
        df = drop(df, ['5'], 'a')
