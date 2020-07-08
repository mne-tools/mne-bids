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
                      make_report)
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

    report = make_report(bids_root)
    print(report)

    expected_report = \
    """This dataset was created with BIDS version 1.4.0 by Please cite MNE-BIDS in your
publication before removing this (citations in README). This report was
generated with MNE-BIDS (https://doi.org/10.21105/joss.01896). There is 1
subject. The dataset consists of 1 recording sessions 01.The sex of the subjects
are all unknown. The handedness of the subjects are all unknown. The ages of the
subjects are all unknown. Data was acquired using a MEG system (Elekta
manufacturer) with line noise at 60 Hz . There is 1 scan in total, 376.0 +/- 0.0
total recording channels per scan (374.0 +/- 0.0 are used and 2.0 +/- 0.0 are
removed from analysis). The dataset lengths range from 20.0 to 20.0 seconds, for
a total of 20.0 seconds of data recorded over all scans (20.0 +/- 0.0)."""  # noqa
    assert report == expected_report
