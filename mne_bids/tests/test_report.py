"""Testing automatic BIDS report."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import os.path as op

import mne
import pytest
from mne.datasets import testing
from mne.utils import _TempDir

from mne_bids import (make_bids_basename,
                      create_methods_paragraph)
from mne_bids.write import write_raw_bids

subject_id = '01'
session_id = '01'
run = '01'
acq = '01'
task = 'testing'

bids_basename = make_bids_basename(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)

# Get the MNE testing sample data
data_path = testing.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_raw.fif')

warning_str = dict(
    channel_unit_changed='ignore:The unit for chann*.:RuntimeWarning:mne',
)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_raw_kind():
    """Test that read_raw_bids() can infer the kind if need be."""
    bids_root = _TempDir()
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)

    methods_summ = create_methods_paragraph(bids_root)

    assert methods_summ == 'The dataset consists of 1 patients ' \
                           'with 1 sessions (01) ' \
                           'consisting of 1 kinds of data (meg). ' \
                           'The dataset consists of 1 subjects ' \
                           '(ages  with 1 unknown age; ' \
                           '1 unknown hand ; 1 unknown sex ). ' \
                           'There are 1 datasets (20.00 +/- 0.00 seconds) ' \
                           'with sampling rates 300.0 (n=1).'
