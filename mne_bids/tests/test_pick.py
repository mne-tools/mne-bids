"""Test for the coil type picking function."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from mne.datasets import testing
from mne.io import read_raw_fif

from mne_bids.pick import coil_type

data_path = testing.data_path(download=False)


@testing.requires_testing_data
def test_coil_type():
    """Test the correct coil type is retrieved."""
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
    raw = read_raw_fif(raw_fname)
    assert coil_type(raw.info, 0) == "meggradplanar"
    assert coil_type(raw.info, 2) == "megmag"
    assert coil_type(raw.info, 306) == "misc"
    assert coil_type(raw.info, 315) == "eeg"
    raw.info["chs"][0]["coil_type"] = 1234
    assert coil_type(raw.info, 0) == "n/a"
