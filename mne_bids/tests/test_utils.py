"""Testing utilities for the MNE BIDS converter."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os

import pytest

from mne.io import read_raw_brainvision, BaseRaw
from mne.utils import _TempDir
from mne_bids.utils import (make_bids_folders, make_bids_filename,
                            _check_types, make_test_brainvision_data,
                            copyfile_brainvision)


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


def test__check_types():
    """Test the check whether vars are str or None."""
    assert _check_types(['foo', 'bar', None]) is None
    with pytest.raises(ValueError):
            _check_types([None, 1, 3.14, 'eeg', [1, 2]])


def test_brainvision_utils():
    """Test generation of brainvision data and moving it around."""
    # Make some test brainvision data
    bv_ext = ['.eeg', '.vhdr', '.vmrk']
    basename = 'testnow'
    data_dir = _TempDir()
    _vhdr = make_test_brainvision_data(output_dir=data_dir, basename=basename)

    # Assert that we can read it
    raw = read_raw_brainvision(_vhdr)
    assert isinstance(raw, BaseRaw)

    bv_file_paths = []
    for ext in bv_ext:
        bv_file_paths.append(os.path.join(data_dir, basename + ext))

    # quick check of make_test_brainvision_data's return value
    assert _vhdr in bv_file_paths

    # Now try to move it to a new place and name
    new_data_dir = _TempDir()
    new_basename = 'testedalready'
    new_bv_file_paths = []
    for ext in bv_ext:
        new_bv_file_paths.append(os.path.join(new_data_dir,
                                              new_basename + ext))
    for src, dest in zip(bv_file_paths, new_bv_file_paths):
        copyfile_brainvision(src, dest)

    # Assert we can read the new file with its new pointers
    new_vhdr = os.path.join(new_data_dir, new_basename + '.vhdr')
    raw = read_raw_brainvision(new_vhdr)
    assert isinstance(raw, BaseRaw)

    # Assert that errors are raised
    with pytest.raises(IOError):
        # source file does not exist
        copyfile_brainvision('I_dont_exist.vhdr', 'dest.vhdr')

    with pytest.raises(ValueError):
        # Unequal extensions
        copyfile_brainvision(new_vhdr, 'dest.vmrk')

    with pytest.raises(ValueError):
        # Wrong extension
        wrong_ext_f = os.path.join(new_data_dir, 'wrong_ext.x')
        open(wrong_ext_f, 'w').close()
        copyfile_brainvision(wrong_ext_f, 'dest.x')
