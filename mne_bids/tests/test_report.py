"""Testing automatic BIDS report."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD-3-Clause
import os.path as op

import mne
import pytest
from mne.datasets import testing

from mne_bids import (BIDSPath,
                      make_report)
from mne_bids.write import write_raw_bids
from mne_bids.config import BIDS_VERSION


subject_id = '01'
session_id = '01'
run = '01'
acq = '01'
task = 'testing'

bids_path = BIDSPath(
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
def test_report(tmp_path):
    """Test that report generated works as intended."""
    bids_root = str(tmp_path)
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw.info['line_freq'] = 60
    bids_path.update(root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    report = make_report(bids_root)

    expected_report = \
    f"""This dataset was created by [Unspecified] and conforms to BIDS version {BIDS_VERSION}.
This report was generated with MNE-BIDS (https://doi.org/10.21105/joss.01896).
The dataset consists of 1 participants (sex were all unknown; handedness were
all unknown; ages all unknown) and 1 recording sessions: 01. Data was recorded
using a MEG system (Elekta manufacturer) sampled at 300.31 Hz with line noise at
60.0 Hz. The following software filters were applied during recording:
SpatialCompensation. There was 1 scan in total. Recording durations ranged from
20.0 to 20.0 seconds (mean = 20.0, std = 0.0), for a total of 20.0 seconds of
data recorded over all scans. For each dataset, there were on average 376.0 (std
= 0.0) recording channels per scan, out of which 374.0 (std = 0.0) were used in
analysis (2.0 +/- 0.0 were removed from analysis)."""  # noqa

    assert report == expected_report
