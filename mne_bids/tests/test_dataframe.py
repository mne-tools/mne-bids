"""Test for the DataFrame object"""
# Authors: Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)

import os.path as op
from collections import OrderedDict as odict
from mne_bids.dataframe import DataFrame
import numpy as np
import pytest

from mne.utils import _TempDir


def test_dataframe():
    # create df
    d = odict([('a', [1, 2, 3, 4]), ('b', [5, 6, 7, 8])])
    df = DataFrame(d)
    # create another
    df2 = DataFrame(odict([('a', [1]), ('b', [5])]))
    assert df2 in df
    df2 = DataFrame(odict([('a', [5]), ('b', [9])]))
    df.append(df2)
    assert '5' in df['a']
    df2 = DataFrame(odict([('a', [5]), ('b', [10])]))
    df.append(df2, drop_column='a')
    assert '9' not in df['b']

    tempdir = _TempDir()
    df_path = op.join(tempdir, 'output.tsv')

    # write the DataFrame to an output tsv file
    df.to_tsv(df_path)
    # now read it back
    df = DataFrame.from_tsv(df_path)

    # remove any rows with 2 or 5 in them
    df.drop([2, 5], 'a')
    assert '2' not in df['a']
    assert ['5', '10'] not in df
    assert np.array(['3', '7']) in df
    df.drop([], 'a')
    with pytest.raises(ValueError):
        df.drop(['5'], 'a')
