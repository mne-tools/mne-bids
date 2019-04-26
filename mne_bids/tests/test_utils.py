"""Testing utilities for the MNE BIDS converter."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import pytest
from datetime import datetime

from numpy.random import random
import mne
from mne.utils import _TempDir

from mne_bids import make_bids_folders, make_bids_basename
from mne_bids.utils import (_check_types, print_dir_tree, _age_on_date,
                            _infer_eeg_placement_scheme, _handle_kind)


base_path = op.join(op.dirname(mne.__file__), 'io')


def test_handle_kind():
    """Test the automatic extraction of kind from the data."""
    # Create a dummy raw
    n_channels = 1
    sampling_rate = 100
    data = random((n_channels, sampling_rate))
    channel_types = ['grad', 'eeg', 'ecog']
    expected_kinds = ['meg', 'eeg', 'ieeg']
    # do it once for each type ... and once for "no type"
    for chtype, kind in zip(channel_types, expected_kinds):
        info = mne.create_info(n_channels, sampling_rate, ch_types=[chtype])
        raw = mne.io.RawArray(data, info)
        assert _handle_kind(raw) == kind

    # if the situation is ambiguous (EEG and iEEG channels both), raise error
    with pytest.raises(ValueError, match='Both EEG and iEEG channels found'):
        info = mne.create_info(2, sampling_rate,
                               ch_types=['eeg', 'ecog'])
        raw = mne.io.RawArray(random((2, sampling_rate)), info)
        _handle_kind(raw)

    # if we cannot find a proper channel type, we raise an error
    with pytest.raises(ValueError, match='Neither MEG/EEG/iEEG channels'):
        info = mne.create_info(n_channels, sampling_rate, ch_types=['misc'])
        raw = mne.io.RawArray(data, info)
        _handle_kind(raw)


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
    assert make_bids_basename(**prefix_data) == 'sub-one_ses-two_task-three_acq-four_run-five_proc-six_recording-seven_suffix.csv'  # noqa

    # subsets of keys works
    assert make_bids_basename(subject='one', task='three') == 'sub-one_task-three'  # noqa
    assert make_bids_basename(subject='one', task='three', suffix='hi.csv') == 'sub-one_task-three_hi.csv'  # noqa

    with pytest.raises(ValueError):
        make_bids_basename(subject='one-two', suffix='there.csv')


def test_make_folders():
    """Test that folders are created and named properly."""
    # Make sure folders are created properly
    output_path = _TempDir()
    make_bids_folders(subject='hi', session='foo', kind='ba',
                      output_path=output_path)
    assert op.isdir(op.join(output_path, 'sub-hi', 'ses-foo', 'ba'))
    # If we remove a kwarg the folder shouldn't be created
    output_path = _TempDir()
    make_bids_folders(subject='hi', kind='ba', output_path=output_path)
    assert op.isdir(op.join(output_path, 'sub-hi', 'ba'))
    # check overwriting of folders
    make_bids_folders(subject='hi', kind='ba', output_path=output_path,
                      overwrite=True, verbose=True)


def test_check_types():
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
    assert _age_on_date(bday, exp1) == 23
    assert _age_on_date(bday, exp2) == 24
    assert _age_on_date(bday, exp3) == 24
    with pytest.raises(ValueError):
        _age_on_date(bday, exp4)


def test_infer_eeg_placement_scheme():
    """Test inferring a correct EEG placement scheme."""
    # no eeg channels case (e.g., MEG data)
    data_path = op.join(base_path, 'bti', 'tests', 'data')
    raw_fname = op.join(data_path, 'test_pdf_linux')
    config_fname = op.join(data_path, 'test_config_linux')
    headshape_fname = op.join(data_path, 'test_hs_linux')
    raw = mne.io.read_raw_bti(raw_fname, config_fname, headshape_fname)
    placement_scheme = _infer_eeg_placement_scheme(raw)
    assert placement_scheme == 'n/a'

    # 1020 case
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')
    raw = mne.io.read_raw_brainvision(raw_fname)
    placement_scheme = _infer_eeg_placement_scheme(raw)
    assert placement_scheme == 'based on the extended 10/20 system'

    # Unknown case, use raw from 1020 case but rename a channel
    raw.rename_channels({'P3': 'foo'})
    placement_scheme = _infer_eeg_placement_scheme(raw)
    assert placement_scheme == 'n/a'
