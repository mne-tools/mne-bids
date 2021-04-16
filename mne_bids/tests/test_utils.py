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
from numpy.random import random, RandomState

with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
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
    channel_types = ['grad', 'eeg', 'ecog', 'seeg']
    expected_modalities = ['meg', 'eeg', 'ieeg', 'ieeg']
    # do it once for each data type
    for chtype, datatype in zip(channel_types, expected_modalities):
        info = mne.create_info(n_channels, sampling_rate,
                               ch_types=[chtype] * 2)
        raw = mne.io.RawArray(data, info)
        assert _handle_datatype(raw) == datatype

    # if the situation is ambiguous (multiple data types), raise ValueError
    channel_types = [['grad', 'eeg'], ['grad', 'ecog'], ['eeg', 'seeg']]
    for chtype in channel_types:
        with pytest.raises(ValueError, match='Multiple data types (MEG, EEG '):
            info = mne.create_info(n_channels, sampling_rate, ch_types=chtype)
            raw = mne.io.RawArray(random((2, sampling_rate)), info)
            _handle_datatype(raw)

    # if we cannot find a proper channel type, raise ValueError
    with pytest.raises(ValueError, match='No MEG, EEG or iEEG channels found'):
        info = mne.create_info(n_channels, sampling_rate,
                               ch_types=['misc'] * 2)
        raw = mne.io.RawArray(data, info)
        _handle_datatype(raw)


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
    datatype = None
    with pytest.raises(ValueError, match=f'The specified datatype {datatype} '
                                         'is currently not supported.'):
        _check_datatype(raw_eeg, datatype)
    datatype = 'anat'
    with pytest.raises(ValueError, match=f'The specified datatype {datatype} '
                                         'is currently not supported.'):
        _check_datatype(raw_eeg, datatype)
    datatype = 'eeg'
    _check_datatype(raw_eeg, datatype)
    with pytest.raises(ValueError, match=f'The specified datatype {datatype} '
                                         'was not found in the raw object.'):
        _check_datatype(raw_meg, datatype)
    with pytest.raises(ValueError, match=f'The specified datatype {datatype} '
                                         'was not found in the raw object.'):
        _check_datatype(raw_ieeg, datatype)
    datatype = 'ieeg'
    _check_datatype(raw_ieeg, datatype)
    with pytest.raises(ValueError, match=f'The specified datatype {datatype} '
                                         'was not found in the raw object.'):
        _check_datatype(raw_meg, datatype)
    with pytest.raises(ValueError, match=f'The specified datatype {datatype} '
                                         'was not found in the raw object.'):
        _check_datatype(raw_eeg, datatype)
    datatype = 'meg'
    _check_datatype(raw_meg, datatype)
    with pytest.raises(ValueError, match=f'The specified datatype {datatype} '
                                         'was not found in the raw object.'):
        _check_datatype(raw_eeg, datatype)
    with pytest.raises(ValueError, match=f'The specified datatype {datatype} '
                                         'was not found in the raw object.'):
        _check_datatype(raw_ieeg, datatype)
