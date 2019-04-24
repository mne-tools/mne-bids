"""Test for the coil type picking function."""
# Authors: Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)

import os.path as op

from mne.datasets import testing
from mne.io import read_raw_fif
from mne_bids.pick import coil_type


def test_coil_type():
    """Test the correct coil type is retrieved."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = read_raw_fif(raw_fname)
    assert coil_type(raw.info, 0) == 'meggradplanar'
    assert coil_type(raw.info, 2) == 'megmag'
    assert coil_type(raw.info, 306) == 'misc'
    assert coil_type(raw.info, 315) == 'eeg'
    raw.info['chs'][0]['coil_type'] = 1234
    assert coil_type(raw.info, 0) == 'n/a'
