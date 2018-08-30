"""Testing utilities for the MNE BIDS converter."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os

import pytest

from datetime import datetime

from mne.utils import _TempDir
from mne_bids.utils import (make_bids_folders, make_bids_filename,
                            _check_types, print_dir_tree, age_on_date)


def test_print_dir_tree():
    """Test printing a dir tree."""
    with pytest.raises(ValueError):
        print_dir_tree('i_dont_exist')

    tmp_dir = _TempDir()
    assert print_dir_tree(tmp_dir) is None


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
            _check_types([None, 1, 3.14, 'meg', [1, 2]])


def test_age_on_date():
    """Test whether the age is determined correctly."""
    bday = datetime(1994, 1, 26)
    exp1 = datetime(2018, 1, 25)
    exp2 = datetime(2018, 1, 26)
    exp3 = datetime(2018, 1, 27)
    exp4 = datetime(1990, 1, 1)
    assert age_on_date(bday, exp1) == 23
    assert age_on_date(bday, exp2) == 24
    assert age_on_date(bday, exp3) == 24
    with pytest.raises(ValueError):
        age_on_date(bday, exp4)
