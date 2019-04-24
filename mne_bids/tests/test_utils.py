"""Testing utilities for the MNE BIDS converter."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import pytest
from datetime import datetime

from scipy.io import savemat
from numpy.random import random
import mne
from mne.datasets import testing
from mne.utils import _TempDir

from mne_bids.utils import (make_bids_folders, make_bids_basename,
                            _check_types, print_dir_tree, _age_on_date,
                            _get_brainvision_encoding, _get_brainvision_paths,
                            copyfile_brainvision, copyfile_eeglab,
                            _infer_eeg_placement_scheme, _handle_kind)


def test_print_dir_tree():
    """Test printing a dir tree."""
    with pytest.raises(ValueError):
        print_dir_tree('i_dont_exist')

    tmp_dir = _TempDir()
    assert print_dir_tree(tmp_dir) is None
