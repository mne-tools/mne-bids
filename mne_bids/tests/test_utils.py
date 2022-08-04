"""Testing utilities for the MNE BIDS converter."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
import os.path as op
from datetime import datetime
from pathlib import Path

import pytest
from numpy.random import random, RandomState

import mne

from mne_bids import BIDSPath
from mne_bids.utils import (_check_types, _age_on_date, _handle_datatype,
                            _infer_eeg_placement_scheme, _get_ch_type_mapping,
                            _check_datatype)
from mne_bids.path import _path_to_str

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
session_id = '01'
run = '01'
acq = None
task = 'testing'

bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)


def test_get_ch_type_mapping():
    """Test getting a correct channel mapping."""
    with pytest.raises(ValueError, match='specified from "bogus" to "mne"'):
        _get_ch_type_mapping(fro='bogus', to='mne')


def test_handle_datatype():
    """Test the automatic extraction of datatype from the data."""
    # Create a dummy raw
    n_channels = 2
    sampling_rate = 100
    data = random((n_channels, sampling_rate))
    # datatype is given, check once for each datatype
    channel_types = ['grad', 'eeg', 'ecog', 'seeg', 'dbs']
    datatypes = ['meg', 'eeg', 'ieeg', 'ieeg', 'ieeg']
    for ch_type, datatype in zip(channel_types, datatypes):
        info = mne.create_info(n_channels, sampling_rate,
                               ch_types=[ch_type] * 2)
        raw = mne.io.RawArray(data, info)
        assert _handle_datatype(raw, datatype) == datatype
    # datatype is not given, will be inferred if possible
    datatype = None
    # check if datatype is correctly inferred (combined EEG and iEEG/MEG data)
    channel_types = [['grad', 'eeg'], ['eeg', 'mag'], ['eeg', 'seeg'],
                     ['ecog', 'eeg']]
    expected_modalities = ['meg', 'meg', 'ieeg', 'ieeg']
    for ch_type, expected_mod in zip(channel_types, expected_modalities):
        info = mne.create_info(n_channels, sampling_rate, ch_types=ch_type)
        raw = mne.io.RawArray(random((2, sampling_rate)), info)
        assert _handle_datatype(raw, datatype) == expected_mod
    # set type to MEG if type is EEG/iEEG but there are MEG channels as well
    channel_types = [['grad', 'eeg'], ['grad', 'seeg']]
    datatypes = ['eeg', 'ieeg']
    for ch_type, datatype in zip(channel_types, datatypes):
        info = mne.create_info(n_channels, sampling_rate, ch_types=ch_type)
        raw = mne.io.RawArray(random((2, sampling_rate)), info)
        assert _handle_datatype(raw, datatype) == 'meg'
    # if the situation is ambiguous (iEEG and MEG), raise ValueError
    datatype = None
    channel_types = [['grad', 'ecog'], ['grad', 'seeg']]
    for ch_type in channel_types:
        with pytest.raises(ValueError, match='Multiple data types'):
            info = mne.create_info(n_channels, sampling_rate, ch_types=ch_type)
            raw = mne.io.RawArray(random((2, sampling_rate)), info)
            _handle_datatype(raw, datatype)
    # if proper channel type (iEEG, EEG or MEG) is not found, raise ValueError
    ch_type = ['misc']
    with pytest.raises(ValueError, match='No MEG, EEG or iEEG channels found'):
        info = mne.create_info(n_channels, sampling_rate,
                               ch_types=ch_type * 2)
        raw = mne.io.RawArray(data, info)
        _handle_datatype(raw, datatype)


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


def test_check_datatype():
    """Test checking if datatype exists in raw data."""
    sfreq, n_points = 1024., int(1e6)
    rng = RandomState(99)
    info_eeg = mne.create_info(['ch1', 'ch2', 'ch3'], sfreq, ['eeg'] * 3)
    raw_eeg = mne.io.RawArray(rng.random((3, n_points)) * 1e-6, info_eeg)
    info_meg = mne.create_info(['ch1', 'ch2', 'ch3'], sfreq, ['mag'] * 3)
    raw_meg = mne.io.RawArray(rng.random((3, n_points)) * 1e-6, info_meg)
    info_ieeg = mne.create_info(['ch1', 'ch2', 'ch3'], sfreq, ['seeg'] * 3)
    raw_ieeg = mne.io.RawArray(rng.random((3, n_points)) * 1e-6, info_ieeg)
    # check behavior for unsupported data types
    for datatype in (None, 'anat'):
        with pytest.raises(ValueError, match=f'The specified datatype '
                                             f'{datatype} is currently not'):
            _check_datatype(raw_eeg, datatype)
    # check behavior for matching data type
    for raw, datatype in [(raw_eeg, 'eeg'), (raw_meg, 'meg'),
                          (raw_ieeg, 'ieeg')]:
        _check_datatype(raw, datatype)
    # check for missing data type
    for raw, datatype in [(raw_ieeg, 'eeg'), (raw_meg, 'eeg'),
                          (raw_ieeg, 'meg'), (raw_eeg, 'meg'),
                          (raw_meg, 'ieeg'), (raw_eeg, 'ieeg')]:
        with pytest.raises(ValueError, match=f'The specified datatype '
                                             f'{datatype} was not found'):
            _check_datatype(raw, datatype)
