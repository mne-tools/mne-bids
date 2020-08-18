"""Testing utilities for the MNE BIDS converter."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
# This is here to handle mne-python <0.20
import warnings
from datetime import datetime
from pathlib import Path

import pytest
from numpy.random import random

with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne

from mne_bids import make_bids_basename
from mne_bids.utils import (_check_types, _age_on_date,
                            _infer_eeg_placement_scheme, _handle_kind,
                            _get_ch_type_mapping)
from mne_bids.path import _path_to_str

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
session_id = '01'
run = '01'
acq = None
task = 'testing'

bids_basename = make_bids_basename(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)


def test_get_ch_type_mapping():
    """Test getting a correct channel mapping."""
    with pytest.raises(ValueError, match='specified from "bogus" to "mne"'):
        _get_ch_type_mapping(fro='bogus', to='mne')


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


def test_make_filenames():
    """Test that we create filenames according to the BIDS spec."""
    # All keys work
    prefix_data = dict(subject='one', session='two', task='three',
                       acquisition='four', run='five', processing='six',
                       recording='seven', kind='ieeg', extension='.json')
    expected_str = 'sub-one_ses-two_task-three_acq-four_run-five_proc-six_rec-seven_ieeg.json'  # noqa
    assert str(make_bids_basename(**prefix_data)) == expected_str

    # subsets of keys works
    assert (make_bids_basename(subject='one', task='three', run=4) ==
            'sub-one_task-three_run-04')
    assert (make_bids_basename(subject='one', task='three',
                               kind='meg', extension='.json') ==
            'sub-one_task-three_meg.json')

    with pytest.raises(ValueError):
        make_bids_basename(subject='one-two', kind='ieeg', extension='.edf')

    with pytest.raises(ValueError, match='At least one'):
        make_bids_basename()

    # emptyroom checks
    with pytest.raises(ValueError, match='empty-room session should be a '
                                         'string of format YYYYMMDD'):
        make_bids_basename(subject='emptyroom', session='12345', task='noise')
    with pytest.raises(ValueError, match='task must be'):
        make_bids_basename(subject='emptyroom', session='20131201',
                           task='blah')


def test_check_types():
    """Test the check whether vars are str or None."""
    assert _check_types(['foo', 'bar', None]) is None
    with pytest.raises(ValueError):
        _check_types([None, 1, 3.14, 'meg', [1, 2]])


def test_path_to_str():
    """Test that _path_to_str returns a string."""
    path_str = 'foo'
    assert _path_to_str(path_str) == path_str
    assert _path_to_str(Path(path_str)) == path_str

    with pytest.raises(ValueError):
        _path_to_str(1)


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
