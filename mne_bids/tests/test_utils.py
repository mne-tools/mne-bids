""" Testing utilities for the MNE BIDS converter
"""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

from mne.utils import _TempDir
from mne_bids.utils import make_bids_folders, make_bids_filename
import pytest
import os


def test_make_filenames():
    """Test that we create filenames according to the BIDS spec."""
    # All keys work
    prefix_data = dict(subject='one', session='two', task='three',
                       acquisition='four', run='five', processing='six',
                       recording='seven', suffix='suffix.csv')
    assert make_bids_filename(**prefix_data) == 'sub-one_ses-two_task-three_acq-four_run-five_proc-six_recording-seven_suffix.csv' # noqa

    # subsets of keys works
    assert make_bids_filename(subject='one', task='three') == 'sub-one_task-three' # noqa
    assert make_bids_filename(subject='one', task='three', suffix='hi.csv') == 'sub-one_task-three_hi.csv' # noqa

    with pytest.raises(ValueError):
        make_bids_filename(subject='one-two', suffix='there.csv')


def test_make_folders():
    """Test that folders are created and named properly."""
    # Make sure folders are created properly
    output_path = _TempDir()
    make_bids_folders(subject='hi', session='foo', kind='ba', root=output_path)
    assert os.path.isdir(os.path.join(output_path, 'sub-hi', 'ses-foo', 'ba'))
    # If we remove a kwarg the folder shouldn't be created
    output_path = _TempDir()
    make_bids_folders(subject='hi', kind='ba', root=output_path)
    assert os.path.isdir(os.path.join(output_path, 'sub-hi', 'ba'))
