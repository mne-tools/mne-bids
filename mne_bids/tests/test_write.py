# -*- coding: utf-8 -*-
"""Test the MNE BIDS converter.

For each supported file format, implement a test.
"""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Teon L Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD-3-Clause
import sys
import os
import os.path as op
import pytest
from glob import glob
from datetime import datetime, timezone, timedelta
import shutil as sh
import json
from pathlib import Path
import codecs

from pkg_resources import parse_version

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_array_almost_equal)

import mne
from mne.datasets import testing
from mne.utils import check_version, requires_nibabel, requires_version
from mne.io import anonymize_info
from mne.io.constants import FIFF
from mne.io.kit.kit import get_kit_info

from mne_bids import (write_raw_bids, read_raw_bids, BIDSPath,
                      write_anat, make_dataset_description,
                      mark_channels, write_meg_calibration,
                      write_meg_crosstalk, get_entities_from_fname,
                      get_anat_landmarks, write, anonymize_dataset)
from mne_bids.write import _get_fid_coords
from mne_bids.utils import (_stamp_to_dt, _get_anonymization_daysback,
                            get_anonymization_daysback, _write_json)
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from mne_bids.sidecar_updates import _update_sidecar, update_sidecar_json
from mne_bids.path import _find_matching_sidecar, _parse_ext
from mne_bids.pick import coil_type
from mne_bids.config import REFERENCES, BIDS_COORD_FRAME_DESCRIPTIONS

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
subject_id2 = '02'
session_id = '01'
run = '01'
acq = '01'
run2 = '02'
task = 'testing'

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)
_bids_path_minimal = BIDSPath(subject=subject_id, task=task)

warning_str = dict(
    channel_unit_changed='ignore:The unit for chann*.:RuntimeWarning:mne',
    meas_date_set_to_none="ignore:.*'meas_date' set to None:RuntimeWarning:"
                          "mne",
    nasion_not_found='ignore:.*nasion not found:RuntimeWarning:mne',
    unraisable_exception='ignore:.*Exception ignored.*:'
                         'pytest.PytestUnraisableExceptionWarning',
    encountered_data_in='ignore:Encountered data in*.:RuntimeWarning:mne',
    edf_warning=r'ignore:^EDF\/EDF\+\/BDF files contain two fields .*'
                r':RuntimeWarning:mne',
    maxshield='ignore:.*Internal Active Shielding:RuntimeWarning:mne',
    edfblocks='ignore:.*EDF format requires equal-length data '
              'blocks:RuntimeWarning:mne',
    brainvision_unit='ignore:Encountered unsupported '
                     'non-voltage units*.:UserWarning'
)


def _wrap_read_raw(read_raw):
    def fn(fname, *args, **kwargs):
        raw = read_raw(fname, *args, **kwargs)
        raw.info['line_freq'] = 60
        return raw
    return fn


_read_raw_fif = _wrap_read_raw(mne.io.read_raw_fif)
_read_raw_ctf = _wrap_read_raw(mne.io.read_raw_ctf)
_read_raw_kit = _wrap_read_raw(mne.io.read_raw_kit)
_read_raw_bti = _wrap_read_raw(mne.io.read_raw_bti)
_read_raw_edf = _wrap_read_raw(mne.io.read_raw_edf)
_read_raw_bdf = _wrap_read_raw(mne.io.read_raw_bdf)
_read_raw_eeglab = _wrap_read_raw(mne.io.read_raw_eeglab)
_read_raw_brainvision = _wrap_read_raw(mne.io.read_raw_brainvision)
_read_raw_persyst = _wrap_read_raw(mne.io.read_raw_persyst)
_read_raw_nihon = _wrap_read_raw(mne.io.read_raw_nihon)

# parametrized directory, filename and reader for EEG/iEEG data formats
test_eegieeg_data = [
    ('EDF', 'test_reduced.edf', _read_raw_edf),
    ('Persyst', 'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay', _read_raw_persyst),  # noqa
    ('NihonKohden', 'MB0400FU.EEG', _read_raw_nihon)
]
test_convert_data = test_eegieeg_data.copy()
test_convert_data.append(('CTF', 'testdata_ctf.ds', _read_raw_ctf))

# parametrization for testing conversion of file formats for MEG
test_convertmeg_data = [
    ('CTF', 'FIF', 'testdata_ctf.ds', _read_raw_ctf),
    ('CTF', 'auto', 'testdata_ctf.ds', _read_raw_ctf),
]

# parametrization for testing converting file formats for EEG/iEEG
test_converteeg_data = [
    ('Persyst', 'BrainVision', 'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay', _read_raw_persyst),  # noqa
    ('NihonKohden', 'BrainVision', 'MB0400FU.EEG', _read_raw_nihon),
    ('Persyst', 'EDF', 'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay', _read_raw_persyst),  # noqa
    ('NihonKohden', 'EDF', 'MB0400FU.EEG', _read_raw_nihon),
]


def _test_anonymize(root, raw, bids_path, events_fname=None, event_id=None):
    """Write data to `root` for testing anonymization."""
    bids_path = _bids_path.copy().update(root=root)
    if raw.info['meas_date'] is not None:
        daysback, _ = get_anonymization_daysback(raw)
    else:
        # just pass back any arbitrary number if no measurement date
        daysback = 3300
    write_raw_bids(raw, bids_path, events_data=events_fname,
                   event_id=event_id, anonymize=dict(daysback=daysback),
                   overwrite=False)
    scans_tsv = BIDSPath(
        subject=subject_id, session=session_id,
        suffix='scans', extension='.tsv', root=root)
    data = _from_tsv(scans_tsv)
    if data['acq_time'] is not None and data['acq_time'][0] != 'n/a':
        assert datetime.strptime(data['acq_time'][0],
                                 '%Y-%m-%dT%H:%M:%S.%fZ').year < 1925

    return root


def test_write_participants(_bids_validate, tmp_path):
    """Test participants.tsv/.json file writing.

    Test that user modifications of the participants
    files are kept, and mne-bids correctly writes all
    the subject info it can using ``raw.info['subject_info']``.
    """
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)

    # add fake participants data
    raw.set_meas_date(datetime(year=1994, month=1, day=26,
                               tzinfo=timezone.utc))
    raw.info['subject_info'] = {'his_id': subject_id2,
                                'birthday': (1993, 1, 26),
                                'sex': 1, 'hand': 2}

    bids_path = _bids_path.copy().update(root=tmp_path)
    write_raw_bids(raw, bids_path)

    # assert age of participant is correct
    participants_tsv = tmp_path / 'participants.tsv'
    data = _from_tsv(participants_tsv)
    assert data['age'][data['participant_id'].index('sub-01')] == '1'

    # if we remove some fields, they should be filled back in upon
    # re-writing with 'n/a'
    data = _from_tsv(participants_tsv)
    data.pop('hand')
    _to_tsv(data, participants_tsv)

    # write in now another subject
    bids_path.update(subject='02')
    write_raw_bids(raw, bids_path, verbose=False)
    data = _from_tsv(participants_tsv)

    # hand should have been written properly with now 'n/a' for sub-01
    # but 'L' for sub-02
    assert data['hand'][data['participant_id'].index('sub-01')] == 'n/a'
    assert data['hand'][data['participant_id'].index('sub-02')] == 'L'

    # check to make sure participant data is overwritten, but keeps the fields
    # if there are extra fields that were user defined
    data = _from_tsv(participants_tsv)
    participant_idx = data['participant_id'].index(f'sub-{subject_id}')
    # create a new test column in participants file tsv
    data['subject_test_col1'] = ['n/a'] * len(data['participant_id'])
    data['subject_test_col1'][participant_idx] = 'S'
    data['test_col2'] = ['n/a'] * len(data['participant_id'])
    orig_key_order = list(data.keys())
    _to_tsv(data, participants_tsv)
    # create corresponding json entry
    participants_json_fpath = tmp_path / 'participants.json'
    json_field = {
        'Description': 'trial-outcome',
        'Levels': {
            'S': 'success',
            'F': 'failure'
        }
    }
    _update_sidecar(participants_json_fpath, 'subject_test_col1', json_field)
    # bids root should still be valid because json reflects changes in tsv
    _bids_validate(tmp_path)
    write_raw_bids(raw, bids_path, overwrite=True)
    data = _from_tsv(participants_tsv)
    with open(participants_json_fpath, 'r', encoding='utf-8') as fin:
        participants_json = json.load(fin)
    assert 'subject_test_col1' in participants_json
    assert data['subject_test_col1'][participant_idx] == 'S'
    # in addition assert the original ordering of the new overwritten file
    assert list(data.keys()) == orig_key_order

    # if overwrite is False, then nothing should change from the above
    with pytest.raises(FileExistsError, match='already exists'):
        raw.info['subject_info'] = None
        write_raw_bids(raw, bids_path, overwrite=False)
    data = _from_tsv(participants_tsv)
    with open(participants_json_fpath, 'r', encoding='utf-8') as fin:
        participants_json = json.load(fin)
    assert 'subject_test_col1' in participants_json
    assert data['age'][data['participant_id'].index('sub-01')] == '1'
    assert data['subject_test_col1'][participant_idx] == 'S'
    # in addition assert the original ordering of the new overwritten file
    assert list(data.keys()) == orig_key_order


def test_write_correct_inputs():
    """Test that inputs of write_raw_bids is correct."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)

    bids_path_str = 'sub-01_ses-01_meg.fif'
    with pytest.raises(RuntimeError, match='"bids_path" must be a '
                                           'BIDSPath object'):
        write_raw_bids(raw, bids_path_str)

    bids_path = _bids_path.copy()
    assert bids_path.root is None
    with pytest.raises(
            ValueError,
            match='The root of the "bids_path" must be set'):
        write_raw_bids(raw=raw, bids_path=bids_path)

    bids_path = _bids_path.copy().update(root='/foo', subject=None)
    with pytest.raises(
            ValueError,
            match='The subject of the "bids_path" must be set'):
        write_raw_bids(raw=raw, bids_path=bids_path)

    bids_path = _bids_path.copy().update(root='/foo', task=None)
    with pytest.raises(
            ValueError,
            match='The task of the "bids_path" must be set'):
        write_raw_bids(raw=raw, bids_path=bids_path)


def test_make_dataset_description(tmp_path, monkeypatch):
    """Test making a dataset_description.json."""
    with pytest.raises(ValueError, match='`dataset_type` must be either "raw" '
                                         'or "derivative."'):
        make_dataset_description(path=tmp_path, name='tst', dataset_type='src')

    make_dataset_description(path=tmp_path, name='tst')

    with open(op.join(tmp_path, 'dataset_description.json'), 'r',
              encoding='utf-8') as fid:
        dataset_description_json = json.load(fid)
        assert dataset_description_json["Authors"] == ["[Unspecified]"]

    make_dataset_description(
        path=tmp_path, name='tst', authors='MNE B., MNE P.',
        funding='GSOC2019, GSOC2021',
        references_and_links='https://doi.org/10.21105/joss.01896',
        dataset_type='derivative', overwrite=False, verbose=True
    )

    with open(op.join(tmp_path, 'dataset_description.json'), 'r',
              encoding='utf-8') as fid:
        dataset_description_json = json.load(fid)
        assert dataset_description_json["Authors"] == ["[Unspecified]"]

    make_dataset_description(
        path=tmp_path, name='tst2', authors='MNE B., MNE P.',
        funding='GSOC2019, GSOC2021',
        references_and_links='https://doi.org/10.21105/joss.01896',
        dataset_type='derivative', overwrite=True, verbose=True
    )

    with open(op.join(tmp_path, 'dataset_description.json'), 'r',
              encoding='utf-8') as fid:
        dataset_description_json = json.load(fid)
        assert dataset_description_json["Authors"] == ['MNE B.', 'MNE P.']

    monkeypatch.setattr(write, 'BIDS_VERSION', 'old')
    with pytest.raises(ValueError, match='Previous BIDS version used'):
        make_dataset_description(path=tmp_path, name='tst')


def test_stamp_to_dt():
    """Test conversions of meas_date to datetime objects."""
    meas_date = (1346981585, 835782)
    meas_datetime = _stamp_to_dt(meas_date)
    assert(meas_datetime == datetime(2012, 9, 7, 1, 33, 5, 835782,
                                     tzinfo=timezone.utc))
    meas_date = (1346981585,)
    meas_datetime = _stamp_to_dt(meas_date)
    assert(meas_datetime == datetime(2012, 9, 7, 1, 33, 5, 0,
                                     tzinfo=timezone.utc))


def test_get_anonymization_daysback():
    """Test daysback querying for anonymization."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)
    daysback_min, daysback_max = _get_anonymization_daysback(raw)
    # max_val off by 1 on Windows for some reason
    assert abs(daysback_min - 28461) < 2 and abs(daysback_max - 36880) < 2
    raw2 = raw.copy()
    with raw2.info._unlock():
        raw2.info['meas_date'] = (np.int32(1158942080), np.int32(720100))
    raw3 = raw.copy()
    with raw3.info._unlock():
        raw3.info['meas_date'] = (np.int32(914992080), np.int32(720100))
    daysback_min, daysback_max = get_anonymization_daysback([raw, raw2, raw3])
    assert abs(daysback_min - 29850) < 2 and abs(daysback_max - 35446) < 2
    raw4 = raw.copy()
    with raw4.info._unlock():
        raw4.info['meas_date'] = (np.int32(4992080), np.int32(720100))
    raw5 = raw.copy()
    with raw5.info._unlock():
        raw5.info['meas_date'] = None
    daysback_min2, daysback_max2 = get_anonymization_daysback([raw, raw2,
                                                               raw3, raw5])
    assert daysback_min2 == daysback_min and daysback_max2 == daysback_max
    with pytest.raises(ValueError, match='The dataset spans more time'):
        daysback_min, daysback_max = \
            get_anonymization_daysback([raw, raw2, raw4])


def test_create_fif(_bids_validate, tmp_path):
    """Test functionality for very short raw file created from data."""
    out_dir = tmp_path / 'out'
    bids_root = tmp_path / 'bids'
    out_dir.mkdir()

    bids_path = _bids_path.copy().update(root=bids_root)
    sfreq, n_points = 1024., int(1e6)
    info = mne.create_info(['ch1', 'ch2', 'ch3', 'ch4', 'ch5'], sfreq,
                           ['seeg'] * 5)
    rng = np.random.RandomState(99)
    raw = mne.io.RawArray(rng.random((5, n_points)) * 1e-6, info)
    raw.info['line_freq'] = 60
    raw.save(op.join(out_dir, 'test-raw.fif'))
    raw = _read_raw_fif(op.join(out_dir, 'test-raw.fif'))
    write_raw_bids(raw, bids_path, verbose=False, overwrite=True)
    _bids_validate(bids_root)


@pytest.mark.parametrize('line_freq', [60, None])
def test_line_freq(line_freq, _bids_validate, tmp_path):
    """Test the power line frequency is written correctly."""
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    bids_root = tmp_path / 'bids'
    bids_path = _bids_path.copy().update(root=bids_root)
    sfreq, n_points = 1024., int(1e6)
    info = mne.create_info(['ch1', 'ch2', 'ch3', 'ch4', 'ch5'], sfreq,
                           ['eeg'] * 5)
    rng = np.random.RandomState(99)
    raw = mne.io.RawArray(rng.random((5, n_points)) * 1e-6, info)

    raw.save(op.join(out_dir, 'test-raw.fif'))
    raw = _read_raw_fif(op.join(out_dir, 'test-raw.fif'))
    raw.info['line_freq'] = line_freq
    write_raw_bids(raw, bids_path, verbose=False, overwrite=True)
    _bids_validate(bids_root)

    eeg_json_fpath = (bids_path.copy()
                      .update(suffix='eeg', extension='.json')
                      .fpath)
    with open(eeg_json_fpath, 'r', encoding='utf-8') as fin:
        eeg_json = json.load(fin)

    if line_freq == 60:
        assert eeg_json['PowerLineFrequency'] == line_freq
    elif line_freq is None:
        assert eeg_json['PowerLineFrequency'] == 'n/a'


@requires_version('pybv', '0.6')
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
@pytest.mark.filterwarnings(warning_str['maxshield'])
def test_fif(_bids_validate, tmp_path):
    """Test functionality of the write_raw_bids conversion for fif."""
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    raw = _read_raw_fif(raw_fname)
    write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                   overwrite=False)

    # Read the file back in to check that the data has come through cleanly.
    # Events and bad channel information was read through JSON sidecar files.
    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_path=bids_path, extra_params=dict(foo='bar'))

    raw2 = read_raw_bids(bids_path=bids_path)
    assert set(raw.info['bads']) == set(raw2.info['bads'])
    events, _ = mne.events_from_annotations(raw2)
    events2 = mne.read_events(events_fname)
    events2 = events2[events2[:, 2] != 0]
    assert_array_equal(events2[:, 0], events[:, 0])

    # check if write_raw_bids works when there is no stim channel
    raw.set_channel_types({raw.ch_names[i]: 'misc'
                           for i in
                           mne.pick_types(raw.info, stim=True, meg=False)})
    bids_root = tmp_path / 'bids2'
    bids_path.update(root=bids_root)
    with pytest.warns(RuntimeWarning, match='No events found or provided.'):
        write_raw_bids(raw, bids_path, overwrite=False)

    _bids_validate(bids_root)

    # try with eeg data only (conversion to bv)
    bids_root = tmp_path / 'bids3'
    bids_root.mkdir()
    bids_path.update(root=bids_root)
    raw = _read_raw_fif(raw_fname)
    raw.load_data()
    raw2 = raw.pick_types(meg=False, eeg=True, stim=True, eog=True, ecg=True)
    raw2.save(bids_root / 'test-raw.fif', overwrite=True)
    raw2 = mne.io.Raw(op.join(bids_root, 'test-raw.fif'), preload=False)
    events = mne.find_events(raw2)
    event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
                'visual/right': 4, 'smiley': 5, 'button': 32}
    # XXX: Need to remove "Status" channel until pybv supports
    # channels that are non-Volt
    idxs = mne.pick_types(raw.info, meg=False, stim=True)
    stim_ch_names = np.array(raw.ch_names)[idxs]
    raw2.drop_channels(stim_ch_names)
    raw.drop_channels(stim_ch_names)

    epochs = mne.Epochs(raw2, events, event_id=event_id, tmin=-0.2, tmax=0.5,
                        preload=True)
    bids_path = bids_path.update(datatype='eeg')
    with pytest.warns(RuntimeWarning,
                      match='Converting data files to BrainVision format'):
        write_raw_bids(raw2, bids_path,
                       events_data=events, event_id=event_id,
                       verbose=True, overwrite=False)
    bids_dir = op.join(bids_root, 'sub-%s' % subject_id,
                       'ses-%s' % session_id, 'eeg')
    sidecar_basename = bids_path.copy()
    for sidecar in ['channels.tsv', 'eeg.eeg', 'eeg.json', 'eeg.vhdr',
                    'eeg.vmrk', 'events.tsv']:
        suffix, extension = sidecar.split('.')
        sidecar_basename.update(suffix=suffix, extension=extension)
        assert op.isfile(op.join(bids_dir, sidecar_basename.basename))

    bids_path.update(root=bids_root, datatype='eeg')
    if check_version('mne', '0.24'):
        with pytest.warns(RuntimeWarning, match='Not setting position'):
            raw2 = read_raw_bids(bids_path=bids_path)
    else:
        raw2 = read_raw_bids(bids_path=bids_path)
    os.remove(op.join(bids_root, 'test-raw.fif'))

    events2, _ = mne.events_from_annotations(raw2, event_id)
    epochs2 = mne.Epochs(raw2, events2, event_id=event_id, tmin=-0.2, tmax=0.5,
                         preload=True)
    assert_array_almost_equal(raw.get_data(), raw2.get_data())
    assert_array_almost_equal(epochs.get_data(), epochs2.get_data(), decimal=4)
    _bids_validate(bids_root)

    # write the same data but pretend it is empty room data:
    raw = _read_raw_fif(raw_fname)
    meas_date = raw.info['meas_date']
    if not isinstance(meas_date, datetime):
        meas_date = datetime.fromtimestamp(meas_date[0], tz=timezone.utc)
    er_date = meas_date.strftime('%Y%m%d')
    er_bids_path = BIDSPath(subject='emptyroom', session=er_date,
                            task='noise', root=bids_root)
    write_raw_bids(raw, er_bids_path, overwrite=False)
    assert op.exists(op.join(
        bids_root, 'sub-emptyroom', 'ses-{0}'.format(er_date), 'meg',
        'sub-emptyroom_ses-{0}_task-noise_meg.json'.format(er_date)))

    _bids_validate(bids_root)

    # test that an incorrect date raises an error.
    er_bids_basename_bad = BIDSPath(subject='emptyroom', session='19000101',
                                    task='noise', root=bids_root)
    with pytest.raises(ValueError, match='The date provided'):
        write_raw_bids(raw, er_bids_basename_bad, overwrite=False)

    # test that the acquisition time was written properly
    scans_tsv = BIDSPath(
        subject=subject_id, session=session_id,
        suffix='scans', extension='.tsv', root=bids_root)
    data = _from_tsv(scans_tsv)
    assert data['acq_time'][0] == meas_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # give the raw object some fake participant data (potentially overwriting)
    raw = _read_raw_fif(raw_fname)
    bids_path_meg = bids_path.copy().update(datatype='meg')
    write_raw_bids(raw, bids_path_meg, events_data=events,
                   event_id=event_id, overwrite=True)

    # try and write preloaded data
    raw = _read_raw_fif(raw_fname, preload=True)
    with pytest.raises(ValueError, match='allow_preload'):
        write_raw_bids(raw, bids_path_meg, events_data=events,
                       event_id=event_id, allow_preload=False, overwrite=False)

    # test anonymize
    raw = _read_raw_fif(raw_fname)
    raw.anonymize()

    raw_fname2 = tmp_path / 'tmp_anon' / 'sample_audvis_raw.fif'
    raw_fname2.parent.mkdir()
    raw.save(raw_fname2)

    # add some readme text
    readme = op.join(bids_root, 'README')
    with open(readme, 'w', encoding='utf-8-sig') as fid:
        fid.write('Welcome to my dataset\n')

    bids_path2 = bids_path_meg.copy().update(subject=subject_id2)
    raw = _read_raw_fif(raw_fname2)
    bids_output_path = write_raw_bids(raw, bids_path2,
                                      events_data=events,
                                      event_id=event_id, overwrite=False)

    # check that the overwrite parameters work correctly for the participant
    # data
    # change the gender but don't force overwrite.
    raw = _read_raw_fif(raw_fname)
    raw.info['subject_info'] = {'his_id': subject_id2,
                                'birthday': (1994, 1, 26), 'sex': 2, 'hand': 1}
    with pytest.raises(FileExistsError, match="already exists"):  # noqa: F821
        write_raw_bids(raw, bids_path2,
                       events_data=events, event_id=event_id, overwrite=False)

    # assert README has references in it
    with open(readme, 'r', encoding='utf-8-sig') as fid:
        text = fid.read()
        assert 'Welcome to my dataset\n' in text
        assert REFERENCES['mne-bids'] in text
        assert REFERENCES['meg'] in text
        assert REFERENCES['eeg'] not in text
        assert REFERENCES['ieeg'] not in text

    # now force the overwrite
    write_raw_bids(raw, bids_path2, events_data=events, event_id=event_id,
                   overwrite=True)

    with open(readme, 'r', encoding='utf-8-sig') as fid:
        text = fid.read()
        assert 'Welcome to my dataset\n' in text
        assert REFERENCES['mne-bids'] in text
        assert REFERENCES['meg'] in text

    with pytest.raises(ValueError, match='raw_file must be'):
        write_raw_bids('blah', bids_path)

    _bids_validate(bids_root)

    assert op.exists(op.join(bids_root, 'participants.tsv'))

    # asserting that single fif files do not include the split key
    files = glob(op.join(bids_output_path, 'sub-' + subject_id2,
                         'ses-' + subject_id2, 'meg', '*.fif'))
    ii = 0
    for ii, FILE in enumerate(files):
        assert 'split' not in FILE
    assert ii < 1

    # check that split files have split key
    raw = _read_raw_fif(raw_fname)
    raw_fname3 = tmp_path / 'test-split-key' / 'sample_audvis_raw.fif'
    raw_fname3.parent.mkdir()
    raw.save(raw_fname3, buffer_size_sec=1.0, split_size='10MB',
             split_naming='neuromag', overwrite=True)
    raw = _read_raw_fif(raw_fname3)
    subject_id3 = '03'
    bids_path3 = bids_path.copy().update(subject=subject_id3)
    bids_output_path = write_raw_bids(raw, bids_path3,
                                      overwrite=False)
    files = glob(op.join(bids_output_path, 'sub-' + subject_id3,
                         'ses-' + subject_id3, 'meg', '*.fif'))
    for FILE in files:
        assert 'split' in FILE

    # test unknown extension
    raw = _read_raw_fif(raw_fname)
    raw._filenames = (raw.filenames[0].replace('.fif', '.foo'),)
    with pytest.raises(ValueError, match='Unrecognized file format'):
        write_raw_bids(raw, bids_path)

    # test whether extra points in raw.info['dig'] are correctly used
    # to set DigitizedHeadShape in the JSON sidecar
    # unchanged sample data includes extra points
    meg_json_path = Path(
        _find_matching_sidecar(
            bids_path=bids_path.copy().update(
                root=bids_root, datatype='meg'
            ),
            suffix='meg',
            extension='.json'
        )
    )

    meg_json = json.loads(meg_json_path.read_text(encoding='utf-8'))
    assert meg_json['DigitizedHeadPoints'] is True

    # drop extra points from raw.info['dig'] and write again
    raw_no_extra_points = _read_raw_fif(raw_fname)
    new_dig = []
    for dig_point in raw_no_extra_points.info['dig']:
        if dig_point['kind'] != FIFF.FIFFV_POINT_EXTRA:
            new_dig.append(dig_point)

    with raw_no_extra_points.info._unlock():
        raw_no_extra_points.info['dig'] = new_dig

    write_raw_bids(raw_no_extra_points, bids_path, events_data=events,
                   event_id=event_id, overwrite=True)

    meg_json_path = Path(
        _find_matching_sidecar(
            bids_path=bids_path.copy().update(
                root=bids_root, datatype='meg'
            ),
            suffix='meg',
            extension='.json'
        )
    )
    meg_json = json.loads(meg_json_path.read_text(encoding='utf-8'))

    assert meg_json['DigitizedHeadPoints'] is False
    assert 'SoftwareFilters' in meg_json
    software_filters = meg_json['SoftwareFilters']
    assert 'SpatialCompensation' in software_filters
    assert 'GradientOrder' in software_filters['SpatialCompensation']
    assert (software_filters['SpatialCompensation']['GradientOrder'] ==
            raw.compensation_grade)


@pytest.mark.parametrize('format', ('fif_no_chpi', 'fif', 'ctf', 'kit'))
@pytest.mark.filterwarnings(warning_str['maxshield'])
def test_chpi(_bids_validate, tmp_path, format):
    """Test writing of cHPI information."""
    data_path = testing.data_path()
    kit_data_path = op.join(base_path, 'kit', 'tests', 'data')

    if format == 'fif_no_chpi':
        fif_raw_fname = op.join(data_path, 'MEG', 'sample',
                                'sample_audvis_trunc_raw.fif')
        raw = _read_raw_fif(fif_raw_fname)
    elif format == 'fif':
        fif_raw_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
        raw = _read_raw_fif(fif_raw_fname, allow_maxshield=True)
    elif format == 'ctf':
        ctf_raw_fname = op.join(data_path, 'CTF', 'testdata_ctf.ds')
        raw = _read_raw_ctf(ctf_raw_fname)
    elif format == 'kit':
        kit_raw_fname = op.join(kit_data_path, 'test.sqd')
        kit_hpi_fname = op.join(kit_data_path, 'test_mrk.sqd')
        kit_electrode_fname = op.join(kit_data_path, 'test.elp')
        kit_headshape_fname = op.join(kit_data_path, 'test.hsp')
        raw = _read_raw_kit(kit_raw_fname, mrk=kit_hpi_fname,
                            elp=kit_electrode_fname, hsp=kit_headshape_fname)

    bids_root = tmp_path / 'bids'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')

    write_raw_bids(raw, bids_path)
    _bids_validate(bids_path.root)

    meg_json = bids_path.copy().update(suffix='meg', extension='.json')
    meg_json_data = json.loads(meg_json.fpath.read_text(encoding='utf-8'))

    if parse_version(mne.__version__) <= parse_version('0.23'):
        assert 'ContinuousHeadLocalization' not in meg_json_data
        assert 'HeadCoilFrequency' not in meg_json_data
    elif format in ['fif_no_chpi', 'kit']:
        # no cHPI info is contained in the sample data
        assert meg_json_data['ContinuousHeadLocalization'] is False
        assert meg_json_data['HeadCoilFrequency'] == []
    elif format == 'fif':
        assert meg_json_data['ContinuousHeadLocalization'] is True
        assert_array_almost_equal(meg_json_data['HeadCoilFrequency'],
                                  [83., 143., 203., 263., 323.])
    elif format == 'ctf':
        assert meg_json_data['ContinuousHeadLocalization'] is True
        assert_array_equal(meg_json_data['HeadCoilFrequency'],
                           np.array([]))


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_fif_dtype(_bids_validate, tmp_path):
    """Test functionality of the write_raw_bids conversion for fif."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    desired_fmt = 'int'
    raw = _read_raw_fif(raw_fname)

    # Fiddle with raw.orig_format -- this should never be done in "real-life",
    # but we do it here to test whether write_raw_bids() will actually stick
    # to the format that's specified in that attribute.
    assert raw.orig_format != desired_fmt  # We're actually changing something
    raw.orig_format = desired_fmt

    write_raw_bids(raw, bids_path, overwrite=False)
    raw = read_raw_bids(bids_path)
    assert raw.orig_format == desired_fmt


def test_fif_anonymize(_bids_validate, tmp_path):
    """Test write_raw_bids() with anonymization fif."""
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root)
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    # test keyword mne-bids anonymize
    raw = _read_raw_fif(raw_fname)
    with pytest.raises(ValueError, match='`daysback` argument required'):
        write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                       anonymize=dict(), overwrite=True)

    bids_root = tmp_path / 'bids2'
    bids_path.update(root=bids_root)
    raw = _read_raw_fif(raw_fname)
    with pytest.warns(RuntimeWarning, match='daysback` is too small'):
        write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                       anonymize=dict(daysback=400), overwrite=False)

    bids_root = tmp_path / 'bids3'
    bids_path.update(root=bids_root)
    raw = _read_raw_fif(raw_fname)
    with pytest.raises(ValueError, match='`daysback` exceeds maximum value'):
        write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                       anonymize=dict(daysback=40000), overwrite=False)

    bids_root = tmp_path / 'bids4'
    bids_path.update(root=bids_root)
    raw = _read_raw_fif(raw_fname)
    write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                   anonymize=dict(daysback=30000, keep_his=True),
                   overwrite=False)
    scans_tsv = BIDSPath(
        subject=subject_id, session=session_id,
        suffix='scans', extension='.tsv',
        root=bids_root)
    data = _from_tsv(scans_tsv)

    # anonymize using MNE manually
    anonymized_info = anonymize_info(info=raw.info, daysback=30000,
                                     keep_his=True)
    anon_date = anonymized_info['meas_date'].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    assert data['acq_time'][0] == anon_date
    _bids_validate(bids_root)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_fif_ias(tmp_path):
    """Test writing FIF files with internal active shielding."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)

    raw.set_channel_types({raw.ch_names[0]: 'ias'})

    data_path = BIDSPath(subject='sample', task='task', root=tmp_path)

    write_raw_bids(raw, data_path)
    raw = read_raw_bids(data_path)
    assert raw.info['chs'][0]['kind'] == FIFF.FIFFV_IAS_CH


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_fif_exci(tmp_path):
    """Test writing FIF files with excitation channel."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)

    raw.set_channel_types({raw.ch_names[0]: 'exci'})
    data_path = BIDSPath(subject='sample', task='task', root=tmp_path)

    write_raw_bids(raw, data_path)
    raw = read_raw_bids(data_path)
    assert raw.info['chs'][0]['kind'] == FIFF.FIFFV_EXCI_CH


def test_kit(_bids_validate, tmp_path):
    """Test functionality of the write_raw_bids conversion for KIT data."""
    bids_root = tmp_path / 'bids'
    data_path = op.join(base_path, 'kit', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.sqd')
    events_fname = op.join(data_path, 'test-eve.txt')
    hpi_fname = op.join(data_path, 'test_mrk.sqd')
    hpi_pre_fname = op.join(data_path, 'test_mrk_pre.sqd')
    hpi_post_fname = op.join(data_path, 'test_mrk_post.sqd')
    electrode_fname = op.join(data_path, 'test.elp')
    headshape_fname = op.join(data_path, 'test.hsp')
    event_id = dict(cond=128)

    kit_bids_path = _bids_path.copy().update(acquisition=None,
                                             root=bids_root,
                                             suffix='meg')

    raw = _read_raw_kit(
        raw_fname, mrk=hpi_fname, elp=electrode_fname,
        hsp=headshape_fname)
    write_raw_bids(raw, kit_bids_path,
                   events_data=events_fname,
                   event_id=event_id, overwrite=False)

    _bids_validate(bids_root)
    assert op.exists(bids_root / 'participants.tsv')
    read_raw_bids(bids_path=kit_bids_path)

    # ensure the marker file is produced in the right place
    marker_fname = BIDSPath(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='markers', extension='.sqd',
        root=bids_root)
    assert op.exists(marker_fname)

    # test anonymize
    output_path = _test_anonymize(
        tmp_path / 'tmp1', raw, kit_bids_path, events_fname, event_id
    )
    _bids_validate(output_path)

    # ensure the channels file has no STI 014 channel:
    channels_tsv = marker_fname.copy().update(datatype='meg',
                                              suffix='channels',
                                              extension='.tsv')
    data = _from_tsv(channels_tsv)
    assert 'STI 014' not in data['name']

    # ensure the marker file is produced in the right place
    assert op.exists(marker_fname)

    # test attempts at writing invalid event data
    event_data = np.loadtxt(events_fname)
    # make the data the wrong number of dimensions
    event_data_3d = np.atleast_3d(event_data)
    other_output_path = tmp_path / 'tmp2'
    bids_path = _bids_path.copy().update(root=other_output_path)
    with pytest.raises(ValueError, match='two dimensions'):
        write_raw_bids(raw, bids_path, events_data=event_data_3d,
                       event_id=event_id, overwrite=True)
    # remove 3rd column
    event_data = event_data[:, :2]
    with pytest.raises(ValueError, match='second dimension'):
        write_raw_bids(raw, bids_path, events_data=event_data,
                       event_id=event_id, overwrite=True)
    # test correct naming of marker files
    raw = _read_raw_kit(
        raw_fname, mrk=[hpi_pre_fname, hpi_post_fname], elp=electrode_fname,
        hsp=headshape_fname)
    kit_bids_path.update(subject=subject_id2)
    write_raw_bids(raw, kit_bids_path, events_data=events_fname,
                   event_id=event_id, overwrite=False)

    _bids_validate(bids_root)
    # ensure the marker files are renamed correctly
    marker_fname.update(acquisition='pre', subject=subject_id2)
    info = get_kit_info(marker_fname, False)[0]
    assert info['meas_date'] == get_kit_info(hpi_pre_fname,
                                             False)[0]['meas_date']
    marker_fname.update(acquisition='post')
    info = get_kit_info(marker_fname, False)[0]
    assert info['meas_date'] == get_kit_info(hpi_post_fname,
                                             False)[0]['meas_date']

    # check that providing markers in the wrong order raises an error
    raw = _read_raw_kit(
        raw_fname, mrk=[hpi_post_fname, hpi_pre_fname], elp=electrode_fname,
        hsp=headshape_fname)
    with pytest.raises(ValueError, match='Markers'):
        write_raw_bids(raw, kit_bids_path.update(subject=subject_id2),
                       events_data=events_fname, event_id=event_id,
                       overwrite=True)

    # check that everything works with MRK markers, and CON files
    data_path = op.join(testing.data_path(download=False), 'KIT')
    raw_fname = op.join(data_path, 'data_berlin.con')
    hpi_fname = op.join(data_path, 'MQKIT_125.mrk')
    electrode_fname = op.join(data_path, 'MQKIT_125.elp')
    headshape_fname = op.join(data_path, 'MQKIT_125.hsp')
    bids_root = tmp_path / 'bids_kit_mrk'
    kit_bids_path = _bids_path.copy().update(acquisition=None,
                                             root=bids_root,
                                             suffix='meg')
    raw = _read_raw_kit(
        raw_fname, mrk=hpi_fname, elp=electrode_fname,
        hsp=headshape_fname)
    write_raw_bids(raw, kit_bids_path)

    _bids_validate(bids_root)
    assert op.exists(bids_root / 'participants.tsv')
    read_raw_bids(bids_path=kit_bids_path)

    # Check that we can successfully write even when elp, hsp, and mrk are not
    # supplied
    raw = _read_raw_kit(raw_fname)
    bids_root = tmp_path / 'no_elp_hsp_mrk'
    kit_bids_path = kit_bids_path.copy().update(root=bids_root)
    write_raw_bids(raw=raw, bids_path=kit_bids_path)
    _bids_validate(bids_root)


@pytest.mark.filterwarnings(warning_str['meas_date_set_to_none'])
def test_ctf(_bids_validate, tmp_path):
    """Test functionality of the write_raw_bids conversion for CTF data."""
    data_path = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(data_path, 'testdata_ctf.ds')
    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg')

    raw = _read_raw_ctf(raw_fname)
    raw.info['line_freq'] = 60
    write_raw_bids(raw, bids_path)
    write_raw_bids(raw, bids_path, overwrite=True)  # test overwrite

    _bids_validate(tmp_path)
    with pytest.warns(RuntimeWarning, match='Did not find any events'):
        raw = read_raw_bids(bids_path=bids_path,
                            extra_params=dict(clean_names=False))

    # test to check that running again with overwrite == False raises an error
    with pytest.raises(FileExistsError, match="already exists"):  # noqa: F821
        write_raw_bids(raw, bids_path)

    assert op.exists(tmp_path / 'participants.tsv')

    # test anonymize
    raw = _read_raw_ctf(raw_fname)
    with pytest.warns(RuntimeWarning,
                      match='Converting to FIF for anonymization'):
        output_path = _test_anonymize(tmp_path / 'tmp', raw, bids_path)
    _bids_validate(output_path)

    raw.set_meas_date(None)
    raw.anonymize()
    with pytest.raises(ValueError, match='All measurement dates are None'):
        get_anonymization_daysback(raw)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_bti(_bids_validate, tmp_path):
    """Test functionality of the write_raw_bids conversion for BTi data."""
    data_path = op.join(base_path, 'bti', 'tests', 'data')
    raw_fname = op.join(data_path, 'test_pdf_linux')
    config_fname = op.join(data_path, 'test_config_linux')
    headshape_fname = op.join(data_path, 'test_hs_linux')

    raw = _read_raw_bti(raw_fname, config_fname=config_fname,
                        head_shape_fname=headshape_fname)

    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg')

    # write the BIDS dataset description, then write BIDS files
    make_dataset_description(tmp_path, name="BTi data")
    write_raw_bids(raw, bids_path, verbose=True)

    assert op.exists(tmp_path / 'participants.tsv')
    _bids_validate(tmp_path)

    raw = read_raw_bids(bids_path=bids_path)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_path=bids_path, extra_params=dict(foo='bar'))

    # test anonymize
    raw = _read_raw_bti(raw_fname, config_fname=config_fname,
                        head_shape_fname=headshape_fname)
    with pytest.warns(RuntimeWarning,
                      match='Converting to FIF for anonymization'):
        output_path = _test_anonymize(tmp_path / 'tmp', raw, bids_path)
    _bids_validate(output_path)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'],
                            warning_str['unraisable_exception'])
def test_vhdr(_bids_validate, tmp_path):
    """Test write_raw_bids conversion for BrainVision data."""
    bids_root = tmp_path / 'bids1'
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')

    raw = _read_raw_brainvision(raw_fname)

    # inject a bad channel
    assert not raw.info['bads']
    injected_bad = ['FP1']
    raw.info['bads'] = injected_bad

    bids_path = _bids_path.copy().update(root=bids_root)
    bids_path_minimal = _bids_path_minimal.copy().update(root=bids_root,
                                                         datatype='eeg')

    # write with injected bad channels
    write_raw_bids(raw, bids_path_minimal, overwrite=False)
    _bids_validate(bids_root)

    # read and also get the bad channels
    raw = read_raw_bids(bids_path=bids_path_minimal)
    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_path=bids_path_minimal,
                      extra_params=dict(foo='bar'))

    # Check that injected bad channel shows up in raw after reading
    np.testing.assert_array_equal(np.asarray(raw.info['bads']),
                                  np.asarray(injected_bad))

    # Test that correct channel units are written ... and that bad channel
    # is in channels.tsv
    suffix, ext = 'channels', '.tsv'
    channels_tsv_name = bids_path_minimal.copy().update(
        suffix=suffix, extension=ext)

    data = _from_tsv(channels_tsv_name)
    assert data['units'][data['name'].index('FP1')] == 'µV'
    assert data['units'][data['name'].index('CP5')] == 'n/a'
    assert data['status'][data['name'].index(injected_bad[0])] == 'bad'
    status_description = data['status_description']
    assert status_description[data['name'].index(injected_bad[0])] == 'n/a'

    # check events.tsv is written
    events_tsv_fname = channels_tsv_name.update(suffix='events')
    assert op.exists(events_tsv_fname)

    # test anonymize and convert
    if check_version('pybv', '0.6'):
        raw = _read_raw_brainvision(raw_fname)
        output_path = _test_anonymize(tmp_path / 'tmp', raw, bids_path)
        _bids_validate(output_path)

    # Also cover iEEG
    # We use the same data and pretend that eeg channels are ecog
    raw = _read_raw_brainvision(raw_fname)
    raw.set_channel_types({raw.ch_names[i]: 'ecog'
                           for i in mne.pick_types(raw.info, eeg=True)})
    bids_root = tmp_path / 'bids2'
    bids_path.update(root=bids_root, datatype='ieeg')
    write_raw_bids(raw, bids_path, overwrite=False)
    _bids_validate(bids_root)

    # Now let's test that the same works for new channel type 'dbs'
    raw = _read_raw_brainvision(raw_fname)
    raw.set_channel_types({raw.ch_names[i]: 'dbs'
                           for i in mne.pick_types(raw.info, eeg=True)})
    bids_root = tmp_path / 'bids_dbs'
    bids_path.update(root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=False)
    _bids_validate(bids_root)

    # Test coords and impedance writing
    # first read the data and set a montage
    data_path = op.join(testing.data_path(), 'montage')
    fname_vhdr = op.join(data_path, 'bv_dig_test.vhdr')
    raw = _read_raw_brainvision(fname_vhdr, preload=False)
    raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    fname_bvct = op.join(data_path, 'captrak_coords.bvct')
    montage = mne.channels.read_dig_captrak(fname_bvct)
    raw.set_montage(montage)

    # convert to BIDS
    bids_root = tmp_path / 'bids3'
    bids_path.update(root=bids_root, datatype='eeg')
    write_raw_bids(raw, bids_path)

    # check impedances
    electrodes_fpath = _find_matching_sidecar(
        bids_path.copy().update(root=bids_root),
        suffix='electrodes', extension='.tsv')
    tsv = _from_tsv(electrodes_fpath)
    assert len(tsv.get('impedance', {})) > 0
    assert tsv['impedance'][-3:] == ['n/a', 'n/a', 'n/a']
    assert tsv['impedance'][:3] == ['5.0', '2.0', '4.0']

    # check coordsystem
    coordsystem_fpath = _find_matching_sidecar(
        bids_path.copy().update(root=bids_root),
        suffix='coordsystem', extension='.json')
    with open(coordsystem_fpath, 'r') as fin:
        coordsys_data = json.load(fin)
        descr = coordsys_data.get("EEGCoordinateSystemDescription", "")
        assert descr == BIDS_COORD_FRAME_DESCRIPTIONS["captrak"]

    # electrodes file path should only contain
    # sub/ses/acq/space at most
    entities = get_entities_from_fname(electrodes_fpath)
    assert all([entity is None for key, entity in entities.items()
                if key not in ['subject', 'session',
                               'acquisition', 'space']])


@pytest.mark.parametrize('dir_name, fname, reader', test_eegieeg_data)
@pytest.mark.filterwarnings(
    warning_str['nasion_not_found'],
    warning_str['brainvision_unit'],
    warning_str['channel_unit_changed']
)
def test_eegieeg(dir_name, fname, reader, _bids_validate, tmp_path):
    """Test write_raw_bids conversion for EEG/iEEG data formats."""
    bids_root = tmp_path / 'bids1'
    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    # the BIDS path for test datasets to get written to
    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg')

    raw = reader(raw_fname)
    events, _ = mne.events_from_annotations(raw, event_id=None)
    kwargs = dict(raw=raw, bids_path=bids_path, overwrite=True)
    if dir_name == 'EDF':
        bids_output_path = write_raw_bids(**kwargs)
    elif dir_name == 'NihonKohden':
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "short" format'):
            bids_output_path = write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            bids_output_path = write_raw_bids(**kwargs)

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

    with pytest.raises(RuntimeError,
                       match='You passed events_data, but no event_id '):
        write_raw_bids(raw, bids_path, events_data=events)

    with pytest.raises(RuntimeError,
                       match='You passed event_id, but no events_data'):
        write_raw_bids(raw, bids_path, event_id=event_id)

    # check events.tsv is written
    events_tsv_fname = bids_output_path.copy().update(suffix='events',
                                                      extension='.tsv')
    if events.size == 0:
        assert not events_tsv_fname.fpath.exists()
    else:
        assert events_tsv_fname.fpath.exists()

    raw2 = read_raw_bids(bids_path=bids_output_path)
    events2, _ = mne.events_from_annotations(raw2)
    assert_array_equal(events2[:, 0], events[:, 0])
    del raw2, events2

    # alter some channels manually
    raw.rename_channels({raw.ch_names[0]: 'EOGtest'})
    raw.info['chs'][0]['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR
    raw.rename_channels({raw.ch_names[1]: 'EMG'})
    raw.set_channel_types({'EMG': 'emg'})

    # Test we can overwrite dataset_description.json
    kwargs = dict(raw=raw, bids_path=bids_path, overwrite=True)
    if dir_name == 'EDF':
        write_raw_bids(**kwargs)
    elif dir_name == 'NihonKohden':
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "short" format'):
            write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            write_raw_bids(**kwargs)

    make_dataset_description(bids_root, name="test",
                             authors=["test1", "test2"], overwrite=True)
    dataset_description_fpath = op.join(bids_root, "dataset_description.json")
    with open(dataset_description_fpath, 'r', encoding='utf-8') as f:
        dataset_description_json = json.load(f)
        assert dataset_description_json["Authors"] == ["test1", "test2"]

    # After writing the entire dataset again, dataset_description.json should
    # contain the default values.
    kwargs = dict(raw=raw, bids_path=bids_path, overwrite=True)
    if dir_name == 'EDF':
        write_raw_bids(**kwargs)
    elif dir_name == 'NihonKohden':
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "short" format'):
            write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            write_raw_bids(**kwargs)

    # dataset_description.json files should not be overwritten inside
    # write_raw_bids calls
    with open(dataset_description_fpath, 'r', encoding='utf-8') as f:
        dataset_description_json = json.load(f)
        assert dataset_description_json["Authors"] == ["test1", "test2"]

    # Reading the file back should still work, even though we've renamed
    # some channels (there's now a mismatch between BIDS and Raw channel
    # names, and BIDS should take precedence)
    raw_read = read_raw_bids(bids_path=bids_path)
    assert raw_read.ch_names[0] == 'EOGtest'
    assert raw_read.ch_names[1] == 'EMG'

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_path=bids_path, extra_params=dict(foo='bar'))

    bids_path = bids_path.copy().update(run=run2)
    # add data in as a montage, but .set_montage only works for some
    # channel types, so make a specific selection
    ch_names = [ch_name
                for ch_name, ch_type in
                zip(raw.ch_names, raw.get_channel_types())
                if ch_type in ['eeg', 'seeg', 'ecog', 'dbs', 'fnirs']]
    elec_locs = np.random.random((len(ch_names), 3))

    # test what happens if there is some nan entries
    elec_locs[-1, :] = [np.nan, np.nan, np.nan]
    ch_pos = dict(zip(ch_names, elec_locs.tolist()))
    eeg_montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                                coord_frame='head')
    raw.set_montage(eeg_montage)
    # electrodes are not written w/o landmarks
    with pytest.warns(RuntimeWarning, match='Skipping EEG electrodes.tsv... '
                                            'Setting montage not possible'):
        write_raw_bids(raw, bids_path, overwrite=True)

    electrodes_fpath = _find_matching_sidecar(bids_path,
                                              suffix='electrodes',
                                              extension='.tsv',
                                              on_error='ignore')
    assert electrodes_fpath is None

    # with landmarks, eeg montage is written
    eeg_montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                                coord_frame='head',
                                                nasion=[1, 0, 0],
                                                lpa=[0, 1, 0],
                                                rpa=[0, 0, 1])
    raw.set_montage(eeg_montage)
    kwargs = dict(raw=raw, bids_path=bids_path, overwrite=True)
    if dir_name == 'EDF':
        write_raw_bids(**kwargs)
    elif dir_name == 'NihonKohden':
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "short" format'):
            write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            write_raw_bids(**kwargs)

    electrodes_fpath = _find_matching_sidecar(bids_path,
                                              suffix='electrodes',
                                              extension='.tsv')
    assert op.exists(electrodes_fpath)
    _bids_validate(bids_root)

    # ensure there is an EMG channel in the channels.tsv:
    channels_tsv = BIDSPath(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels', extension='.tsv', acquisition=acq,
        root=bids_root, datatype='eeg')
    data = _from_tsv(channels_tsv)
    assert 'ElectroMyoGram' in data['description']

    # check that the scans list contains two scans
    scans_tsv = BIDSPath(
        subject=subject_id, session=session_id,
        suffix='scans', extension='.tsv',
        root=bids_root)
    data = _from_tsv(scans_tsv)
    assert len(list(data.values())[0]) == 2

    # check that scans list is properly converted to brainvision
    if check_version('pybv', '0.6') or dir_name == 'EDF':
        daysback_min, daysback_max = _get_anonymization_daysback(raw)
        daysback = (daysback_min + daysback_max) // 2

        kwargs = dict(raw=raw, bids_path=bids_path,
                      anonymize=dict(daysback=daysback), overwrite=True)
        if dir_name == 'EDF':
            match = r"^EDF\/EDF\+\/BDF files contain two fields .*"
            with pytest.warns(RuntimeWarning, match=match):
                write_raw_bids(**kwargs)
        elif dir_name == 'Persyst':
            with pytest.warns(RuntimeWarning,
                              match='Encountered data in "double" format'):
                write_raw_bids(**kwargs)
        else:
            with pytest.warns(RuntimeWarning,
                              match='Encountered data in "short" format'):
                write_raw_bids(**kwargs)

        data = _from_tsv(scans_tsv)
        bids_path = bids_path.copy()
        if dir_name != 'EDF':
            bids_path = bids_path.update(suffix='eeg', extension='.vhdr')
        assert any([bids_path.basename in fname
                    for fname in data['filename']])

    # Also cover iEEG
    # We use the same data and pretend that eeg channels are ecog
    ieeg_raw = raw.copy()

    # remove the old "EEG" montage, to test iEEG functionality
    ieeg_raw.set_montage(None)

    # convert channel types to ECoG and write BIDS
    eeg_picks = mne.pick_types(ieeg_raw.info, eeg=True)
    ieeg_raw.set_channel_types({raw.ch_names[i]: 'ecog'
                                for i in eeg_picks})
    bids_root = tmp_path / 'bids2'
    bids_path.update(root=bids_root, datatype='ieeg')
    kwargs = dict(raw=ieeg_raw, bids_path=bids_path, overwrite=True)
    if dir_name == 'EDF':
        write_raw_bids(**kwargs)
    elif dir_name == 'NihonKohden':
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "short" format'):
            write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            write_raw_bids(**kwargs)

    _bids_validate(bids_root)

    # assert README has references in it
    readme = op.join(bids_root, 'README')
    with open(readme, 'r', encoding='utf-8-sig') as fid:
        text = fid.read()
        assert REFERENCES['ieeg'] in text
        assert REFERENCES['meg'] not in text
        assert REFERENCES['eeg'] not in text

    # test writing electrode coordinates (.tsv)
    # and coordinate system (.json)
    # .set_montage only works for some channel types -> specific selection
    ch_names = [ch_name
                for ch_name, ch_type in
                zip(ieeg_raw.ch_names, ieeg_raw.get_channel_types())
                if ch_type in ['eeg', 'seeg', 'ecog', 'dbs', 'fnirs']]

    elec_locs = np.random.random((len(ch_names), 3)).tolist()
    ch_pos = dict(zip(ch_names, elec_locs))
    ecog_montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                                 coord_frame='mni_tal')
    ieeg_raw.set_montage(ecog_montage)
    bids_root = tmp_path / 'bids3'
    bids_path.update(root=bids_root, datatype='ieeg')
    kwargs = dict(raw=ieeg_raw, bids_path=bids_path, overwrite=True)
    if dir_name == 'EDF':
        write_raw_bids(**kwargs)
    elif dir_name == 'NihonKohden':
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "short" format'):
            write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            write_raw_bids(**kwargs)

    _bids_validate(bids_root)

    # XXX: Should be improved with additional coordinate system descriptions
    # iEEG montages written from mne-python end up as "Other"
    bids_path.update(root=bids_root)
    electrodes_path = bids_path.copy().update(
        suffix='electrodes', extension='.tsv',
        space='fsaverage', task=None, run=None
    ).fpath
    coordsystem_path = bids_path.copy().update(
        suffix='coordsystem', extension='.json',
        space='fsaverage', task=None, run=None
    ).fpath

    assert electrodes_path.exists()
    assert coordsystem_path.exists()

    # Test we get the correct sidecar via _find_matching_sidecar()
    electrodes_fname = _find_matching_sidecar(bids_path,
                                              suffix='electrodes',
                                              extension='.tsv')
    coordsystem_fname = _find_matching_sidecar(bids_path,
                                               suffix='coordsystem',
                                               extension='.json')
    electrodes_fname == str(electrodes_fpath)
    coordsystem_fname == str(coordsystem_path)

    coordsystem_json = json.loads(coordsystem_path.read_text(encoding='utf-8'))
    assert coordsystem_json['iEEGCoordinateSystem'] == 'fsaverage'

    # test writing to ACPC
    ecog_montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                                 coord_frame='mri')
    bids_root = tmp_path / 'bids4'
    bids_path.update(root=bids_root, datatype='ieeg')
    # test works if ACPC-aligned is specified
    kwargs.update(montage=ecog_montage, acpc_aligned=True)
    if dir_name == 'EDF':
        write_raw_bids(**kwargs)
    elif dir_name == 'NihonKohden':
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "short" format'):
            write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            write_raw_bids(**kwargs)

    _bids_validate(bids_root)

    bids_path.update(root=bids_root)
    electrodes_path = bids_path.copy().update(
        suffix='electrodes', extension='.tsv', space='ACPC',
        task=None, run=None
    ).fpath
    coordsystem_path = bids_path.copy().update(
        suffix='coordsystem', extension='.json', space='ACPC',
        task=None, run=None
    ).fpath

    assert electrodes_path.exists()
    assert coordsystem_path.exists()

    # Test we get the correct sidecar via _find_matching_sidecar()
    electrodes_fname = _find_matching_sidecar(bids_path,
                                              suffix='electrodes',
                                              extension='.tsv')
    coordsystem_fname = _find_matching_sidecar(bids_path,
                                               suffix='coordsystem',
                                               extension='.json')
    electrodes_fname == str(electrodes_fpath)
    coordsystem_fname == str(coordsystem_path)

    coordsystem_json = json.loads(coordsystem_path.read_text(encoding='utf-8'))
    assert coordsystem_json['iEEGCoordinateSystem'] == 'ACPC'

    kwargs.update(acpc_aligned=False)
    with pytest.raises(RuntimeError, match='`acpc_aligned` is False'):
        write_raw_bids(**kwargs)

    # test anonymize and convert
    if check_version('pybv', '0.6') or dir_name == 'EDF':
        raw = reader(raw_fname)
        bids_path.update(root=bids_root, datatype='eeg')
        kwargs = dict(raw=raw, bids_path=bids_path, overwrite=True)
        if dir_name == 'NihonKohden':
            with pytest.warns(RuntimeWarning,
                              match='Encountered data in "short" format'):
                write_raw_bids(**kwargs)
                output_path = _test_anonymize(tmp_path / 'a', raw, bids_path)
        elif dir_name == 'EDF':
            match = r"^EDF\/EDF\+\/BDF files contain two fields .*"
            with pytest.warns(RuntimeWarning, match=match):
                write_raw_bids(**kwargs)  # Just copies.
                output_path = _test_anonymize(tmp_path / 'b', raw, bids_path)
        else:
            with pytest.warns(RuntimeWarning,
                              match='Encountered data in "double" format'):
                write_raw_bids(**kwargs)  # Converts.
                output_path = _test_anonymize(tmp_path / 'c', raw, bids_path)
        _bids_validate(output_path)


def test_bdf(_bids_validate, tmp_path):
    """Test write_raw_bids conversion for Biosemi data."""
    data_path = op.join(base_path, 'edf', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.bdf')

    bids_path = _bids_path.copy().update(root=tmp_path, datatype='eeg')

    raw = _read_raw_bdf(raw_fname)
    raw.info['line_freq'] = 60
    write_raw_bids(raw, bids_path, overwrite=False)
    _bids_validate(tmp_path)

    # assert README has references in it
    readme = op.join(tmp_path, 'README')
    with open(readme, 'r', encoding='utf-8-sig') as fid:
        text = fid.read()
        assert REFERENCES['eeg'] in text
        assert REFERENCES['meg'] not in text
        assert REFERENCES['ieeg'] not in text

    # Test also the reading of channel types from channels.tsv
    # the first channel in the raw data is not MISC right now
    test_ch_idx = 0
    assert coil_type(raw.info, test_ch_idx) != 'misc'

    # we will change the channel type to MISC and overwrite the channels file
    bids_fname = bids_path.copy().update(suffix='eeg',
                                         extension='.bdf')
    channels_fname = _find_matching_sidecar(bids_fname,
                                            suffix='channels',
                                            extension='.tsv')
    channels_dict = _from_tsv(channels_fname)
    channels_dict['type'][test_ch_idx] = 'MISC'
    _to_tsv(channels_dict, channels_fname)

    # Now read the raw data back from BIDS, with the tampered TSV, to show
    # that the channels.tsv truly influences how read_raw_bids sets ch_types
    # in the raw data object
    raw = read_raw_bids(bids_path=bids_path)
    assert coil_type(raw.info, test_ch_idx) == 'misc'
    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_path=bids_path, extra_params=dict(foo='bar'))

    # Test errors for modified raw.times
    raw = _read_raw_bdf(raw_fname)

    with pytest.raises(ValueError, match='fewer time points'):
        write_raw_bids(raw.copy().crop(0, raw.times[-2]), bids_path,
                       overwrite=True)

    with pytest.raises(ValueError, match='more time points'):
        write_raw_bids(mne.concatenate_raws([raw.copy(), raw]), bids_path,
                       overwrite=True)

    if hasattr(raw.info, '_unlock'):
        with raw.info._unlock():
            raw.info['sfreq'] -= 10  # change raw.times, but retain shape
    elif parse_version(mne.__version__) >= parse_version('0.23'):
        raw.info['sfreq'] -= 10
    else:
        raw._times = raw._times / 5

    with pytest.raises(ValueError, match='raw.times has changed'):
        write_raw_bids(raw, bids_path, overwrite=True)

    # test anonymize and convert
    raw = _read_raw_bdf(raw_fname)
    match = r"^EDF\/EDF\+\/BDF files contain two fields .*"
    with pytest.warns(RuntimeWarning, match=match):
        output_path = _test_anonymize(tmp_path / 'tmp', raw, bids_path)
    _bids_validate(output_path)


@pytest.mark.filterwarnings(warning_str['meas_date_set_to_none'])
def test_set(_bids_validate, tmp_path):
    """Test write_raw_bids conversion for EEGLAB data."""
    # standalone .set file with associated .fdt
    bids_root = tmp_path / 'bids1'
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw.set')
    raw = _read_raw_eeglab(raw_fname)
    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg')

    # proceed with the actual test for EEGLAB data
    write_raw_bids(raw, bids_path, overwrite=False)
    read_raw_bids(bids_path=bids_path)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_path=bids_path, extra_params=dict(foo='bar'))

    with pytest.raises(FileExistsError, match="already exists"):  # noqa: F821
        write_raw_bids(raw, bids_path, overwrite=False)
    _bids_validate(bids_root)

    # check events.tsv is written
    # XXX: only from 0.18 onwards because events_from_annotations
    # is broken for earlier versions
    events_tsv_fname = op.join(bids_root, 'sub-' + subject_id,
                               'ses-' + session_id, 'eeg',
                               bids_path.basename + '_events.tsv')
    assert op.exists(events_tsv_fname)

    # Also cover iEEG
    # We use the same data and pretend that eeg channels are ecog
    raw.set_channel_types({raw.ch_names[i]: 'ecog'
                           for i in mne.pick_types(raw.info, eeg=True)})
    bids_root = tmp_path / 'bids2'
    bids_path.update(root=bids_root, datatype='ieeg')
    write_raw_bids(raw, bids_path)
    _bids_validate(bids_root)

    # test anonymize and convert
    if check_version('pybv', '0.6'):
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            output_path = _test_anonymize(tmp_path / 'tmp', raw, bids_path)
        _bids_validate(output_path)


def _check_anat_json(bids_path):
    json_path = bids_path.copy().update(extension='.json')
    # Validate that matching sidecar file is as expected
    assert op.exists(json_path.fpath)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)

    # We only should have AnatomicalLandmarkCoordinates as key
    np.testing.assert_array_equal(list(json_dict.keys()),
                                  ['AnatomicalLandmarkCoordinates'])
    # And within AnatomicalLandmarkCoordinates only LPA, NAS, RPA in that order
    anat_dict = json_dict['AnatomicalLandmarkCoordinates']
    point_list = ['LPA', 'NAS', 'RPA']
    np.testing.assert_array_equal(list(anat_dict.keys()),
                                  point_list)
    # test the actual values of the voxels (no floating points)
    for i, point in enumerate([(66, 51, 46), (41, 32, 74), (17, 53, 47)]):
        coords = anat_dict[point_list[i]]
        np.testing.assert_array_equal(np.asarray(coords, dtype=int),
                                      point)


def test_get_anat_landmarks():
    """Test getting anatomical landmarks in image space."""
    data_path = testing.data_path()
    # Get the T1 weighted MRI data file
    # Needs to be converted to Nifti because we only have mgh in our test base
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    fs_subjects_dir = op.join(data_path, 'subjects')
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)
    # Write some MRI data and supply a `trans`
    trans_fname = raw_fname.replace('_raw.fif', '-trans.fif')
    trans = mne.read_trans(trans_fname)

    # define some keyword arguments to simplify testing
    kwargs = dict(image=t1w_mgh, info=raw.info, trans=trans,
                  fs_subject='sample', fs_subjects_dir=fs_subjects_dir)

    # trans has a wrong type
    wrong_type = 1
    match = f'trans must be an instance of .*, got {type(wrong_type)} '
    ex = TypeError

    with pytest.raises(ex, match=match):
        get_anat_landmarks(**dict(kwargs, trans=wrong_type))

    # trans is a str, but file does not exist
    wrong_fname = 'not_a_trans'
    match = 'trans file "{}" not found'.format(wrong_fname)
    with pytest.raises(IOError, match=match):
        get_anat_landmarks(**dict(kwargs, trans=wrong_fname))

    # However, reading trans if it is a string pointing to trans is fine
    get_anat_landmarks(**dict(kwargs, trans=trans_fname))

    # test unsupported coord_frame
    fail_info = raw.info.copy()
    fail_info['dig'][0]['coord_frame'] = 3
    fail_info['dig'][1]['coord_frame'] = 3
    fail_info['dig'][2]['coord_frame'] = 3

    with pytest.raises(ValueError, match='must be in the head'):
        get_anat_landmarks(**dict(kwargs, info=fail_info))

    # test bad freesurfer directory
    with pytest.raises(ValueError, match='subject folder is incorrect'):
        get_anat_landmarks(**dict(kwargs, fs_subject='bad'))

    # test _get_fid_coords
    fail_landmarks = mne.channels.make_dig_montage(
        lpa=[66.08580, 51.33362, 46.52982],
        coord_frame='mri_voxel')

    with pytest.raises(ValueError, match='Some fiducial points are missing'):
        _get_fid_coords(fail_landmarks.dig, raise_error=True)

    fail_landmarks = mne.channels.make_dig_montage(
        lpa=[66.08580, 51.33362, 46.52982],
        nasion=[41.87363, 32.24694, 74.55314],
        rpa=[17.23812, 53.08294, 47.01789],
        coord_frame='mri_voxel')
    fail_landmarks.dig[2]['coord_frame'] = 99

    with pytest.raises(ValueError, match='must be in the same coordinate'):
        _get_fid_coords(fail_landmarks.dig, raise_error=True)

    # test main
    mri_voxel_landmarks = mne.channels.make_dig_montage(
        lpa=[66.08580, 51.33362, 46.52982],
        nasion=[41.87363, 32.24694, 74.55314],
        rpa=[17.23812, 53.08294, 47.01789],
        coord_frame='mri_voxel')
    coords_dict, mri_voxel_coord_frame = _get_fid_coords(
        mri_voxel_landmarks.dig)
    mri_voxel_landmarks = np.asarray((coords_dict['lpa'],
                                      coords_dict['nasion'],
                                      coords_dict['rpa']))
    landmarks = get_anat_landmarks(**kwargs)
    coords_dict2, coord_frame = _get_fid_coords(landmarks.dig)
    landmarks = np.asarray((coords_dict2['lpa'],
                            coords_dict2['nasion'],
                            coords_dict2['rpa']))
    assert mri_voxel_coord_frame == coord_frame
    np.testing.assert_array_almost_equal(
        mri_voxel_landmarks, landmarks, decimal=5)


@requires_nibabel()
def test_write_anat(_bids_validate, tmp_path):
    """Test writing anatomical data."""
    # Get the MNE testing sample data
    import nibabel as nib
    bids_root = tmp_path / 'bids1'
    data_path = testing.data_path()

    # Get the T1 weighted MRI data file
    # Needs to be converted to Nifti because we only have mgh in our test base
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')

    # define hard-coded landmark locations in voxel and scanner RAS
    mri_voxel_landmarks = mne.channels.make_dig_montage(
        lpa=[66.08580, 51.33362, 46.52982],
        nasion=[41.87363, 32.24694, 74.55314],
        rpa=[17.23812, 53.08294, 47.01789],
        coord_frame='mri_voxel')

    mri_scanner_ras_landmarks = mne.channels.make_dig_montage(
        lpa=[-0.07453101, 0.01962855, -0.05228882],
        nasion=[-0.00189453, 0.1036985, 0.00497122],
        rpa=[0.07201203, 0.02109275, -0.05753678],
        coord_frame='ras')

    # write base bids directory
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    raw = _read_raw_fif(raw_fname)
    bids_path = _bids_path.copy().update(root=bids_root)
    write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                   overwrite=False)

    # define some keyword arguments to simplify testing
    kwargs = dict(bids_path=bids_path, landmarks=mri_voxel_landmarks,
                  deface=True, verbose=True, overwrite=True)

    # test writing with no sidecar
    bids_path = write_anat(t1w_mgh, **kwargs)
    anat_dir = bids_path.directory
    _bids_validate(bids_root)
    assert op.exists(op.join(anat_dir, 'sub-01_ses-01_acq-01_T1w.nii.gz'))

    # Validate that files are as expected
    _check_anat_json(bids_path)

    # Now try some anat writing that will fail
    # We already have some MRI data there
    with pytest.raises(IOError, match='`overwrite` is set to False'):
        write_anat(t1w_mgh, **dict(kwargs, overwrite=False))

    # check overwrite no JSON
    with pytest.raises(IOError, match='it already exists'):
        write_anat(t1w_mgh, bids_path=bids_path, verbose=True,
                   overwrite=False)

    # pass some invalid type as T1 MRI
    with pytest.raises(ValueError, match='must be a path to an MRI'):
        write_anat(9999999999999, **kwargs)

    # Return without writing sidecar
    sh.rmtree(anat_dir)
    write_anat(t1w_mgh, bids_path=bids_path)
    # Assert that we truly cannot find a sidecar
    with pytest.raises(RuntimeError, match='Did not find any'):
        _find_matching_sidecar(bids_path,
                               suffix='T1w', extension='.json')

    # Writing without a session does NOT yield "ses-None" anywhere
    bids_path.update(session=None, acquisition=None)
    kwargs.update(bids_path=bids_path)
    bids_path = write_anat(t1w_mgh, bids_path=bids_path)
    anat_dir2 = bids_path.directory
    assert 'ses-None' not in anat_dir2.as_posix()
    assert op.exists(op.join(anat_dir2, 'sub-01_T1w.nii.gz'))

    # test deface
    bids_path = write_anat(t1w_mgh, **kwargs)
    anat_dir = bids_path.directory
    t1w = nib.load(op.join(anat_dir, 'sub-01_T1w.nii.gz'))
    vox_sum = t1w.get_fdata().sum()

    _check_anat_json(bids_path)

    # Check that increasing inset leads to more voxels at 0
    bids_path = write_anat(t1w_mgh, **dict(kwargs, deface=dict(inset=25.)))
    anat_dir2 = bids_path.directory
    t1w2 = nib.load(op.join(anat_dir2, 'sub-01_T1w.nii.gz'))
    vox_sum2 = t1w2.get_fdata().sum()

    _check_anat_json(bids_path)

    assert vox_sum > vox_sum2

    # Check that increasing theta leads to more voxels at 0
    bids_path = write_anat(t1w_mgh, **dict(kwargs, deface=dict(theta=45)))
    anat_dir3 = bids_path.directory
    t1w3 = nib.load(op.join(anat_dir3, 'sub-01_T1w.nii.gz'))
    vox_sum3 = t1w3.get_fdata().sum()

    assert vox_sum > vox_sum3

    with pytest.raises(ValueError, match='must be provided to deface'):
        write_anat(t1w_mgh, bids_path=bids_path, deface=True,
                   verbose=True, overwrite=True)

    with pytest.raises(ValueError, match='inset must be numeric'):
        write_anat(t1w_mgh, **dict(kwargs, deface=dict(inset='small')))

    with pytest.raises(ValueError, match='inset should be positive'):
        write_anat(t1w_mgh, **dict(kwargs, deface=dict(inset=-2.)))

    with pytest.raises(ValueError, match='theta must be numeric'):
        write_anat(t1w_mgh, **dict(kwargs, deface=dict(theta='big')))

    with pytest.raises(ValueError, match='theta should be between 0 and 90'):
        write_anat(t1w_mgh, **dict(kwargs, deface=dict(theta=100)))

    # test using landmarks
    bids_path.update(acquisition=acq)

    # test unsupported coord_frame
    fail_landmarks = mri_voxel_landmarks.copy()
    fail_landmarks.dig[0]['coord_frame'] = 3
    fail_landmarks.dig[1]['coord_frame'] = 3
    fail_landmarks.dig[2]['coord_frame'] = 3

    with pytest.raises(ValueError, match='Coordinate frame not supported'):
        write_anat(t1w_mgh, **dict(kwargs, landmarks=fail_landmarks))

    # Test now using FLASH
    flash_mgh = \
        op.join(data_path, 'subjects', 'sample', 'mri', 'flash', 'mef05.mgz')
    trans_fname = raw_fname.replace('_raw.fif', '-trans.fif')
    landmarks = get_anat_landmarks(flash_mgh, raw.info, trans_fname, 'sample',
                                   op.join(data_path, 'subjects'))
    bids_path = BIDSPath(subject=subject_id, session=session_id,
                         suffix='FLASH', root=bids_root)
    kwargs.update(bids_path=bids_path, landmarks=landmarks)

    bids_path = write_anat(flash_mgh, **kwargs)
    anat_dir = bids_path.directory
    assert op.exists(op.join(anat_dir, 'sub-01_ses-01_FLASH.nii.gz'))
    _bids_validate(bids_root)

    flash1 = nib.load(op.join(anat_dir, 'sub-01_ses-01_FLASH.nii.gz'))
    fvox1 = flash1.get_fdata()

    # test landmarks in scanner RAS coordinates
    bids_path = write_anat(
        flash_mgh, **dict(kwargs, landmarks=mri_scanner_ras_landmarks))
    anat_dir = bids_path.directory
    flash2 = nib.load(op.join(anat_dir, 'sub-01_ses-01_FLASH.nii.gz'))
    fvox2 = flash2.get_fdata()
    assert_array_equal(fvox1, fvox2)

    # test that we can now use a BIDSPath to use the landmarks
    landmarks = get_anat_landmarks(bids_path, raw.info, trans_fname,
                                   'sample', op.join(data_path, 'subjects'))


def test_write_raw_pathlike(tmp_path):
    data_path = Path(testing.data_path())
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32,
                'unknown': 0}
    raw = _read_raw_fif(raw_fname)

    bids_root = tmp_path
    events_fname = (data_path / 'MEG' / 'sample' /
                    'sample_audvis_trunc_raw-eve.fif')
    bids_path = _bids_path.copy().update(root=bids_root)
    bids_path_ = write_raw_bids(raw=raw, bids_path=bids_path,
                                events_data=events_fname,
                                event_id=event_id, overwrite=False)

    # write_raw_bids() should return a string.
    assert isinstance(bids_path_, BIDSPath)
    assert bids_path_.root == bids_root


def test_write_raw_no_dig(tmp_path):
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)
    bids_root = tmp_path
    bids_path = _bids_path.copy().update(root=bids_root)
    bids_path_ = write_raw_bids(raw=raw, bids_path=bids_path,
                                overwrite=True)
    assert bids_path_.root == bids_root
    with raw.info._unlock():
        raw.info['dig'] = None
    raw.save(str(bids_root / 'tmp_raw.fif'))
    raw = _read_raw_fif(bids_root / 'tmp_raw.fif')
    bids_path_ = write_raw_bids(raw=raw, bids_path=bids_path,
                                overwrite=True)
    assert bids_path_.root == bids_root
    assert bids_path_.suffix == 'meg'
    assert bids_path_.extension == '.fif'


@requires_nibabel()
def test_write_anat_pathlike(tmp_path):
    """Test writing anatomical data with pathlib.Paths."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    trans_fname = raw_fname.replace('_raw.fif', '-trans.fif')
    raw = _read_raw_fif(raw_fname)
    trans = mne.read_trans(trans_fname)

    bids_root = tmp_path
    t1w_mgh_fname = Path(data_path) / 'subjects' / 'sample' / 'mri' / 'T1.mgz'
    bids_path = BIDSPath(subject=subject_id, session=session_id,
                         acquisition=acq, root=bids_root)
    landmarks = get_anat_landmarks(
        t1w_mgh_fname, raw.info, trans, 'sample',
        fs_subjects_dir=op.join(data_path, 'subjects'))
    bids_path = write_anat(t1w_mgh_fname, bids_path=bids_path,
                           landmarks=landmarks, deface=True,
                           verbose=True, overwrite=True)

    # write_anat() should return a BIDSPath.
    assert isinstance(bids_path, BIDSPath)


def test_write_does_not_alter_events_inplace(tmp_path):
    """Test that writing does not modify the passed events array."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw = _read_raw_fif(raw_fname)
    events = mne.read_events(events_fname)
    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    events_orig = events.copy()
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

    bids_path = _bids_path.copy().update(root=tmp_path)
    write_raw_bids(raw=raw, bids_path=bids_path,
                   events_data=events, event_id=event_id, overwrite=True)

    assert np.array_equal(events, events_orig)


def _ensure_list(x):
    """Return a list representation of the input."""
    if isinstance(x, str):
        return [x]
    elif x is None:
        return []
    else:
        return list(x)


@pytest.mark.parametrize(
    'ch_names, descriptions, drop_status_col, drop_description_col, '
    'existing_ch_names, existing_descriptions',
    [
        # Only mark channels, do not set descriptions.
        (['MEG 0112', 'MEG 0131', 'EEG 053'], None, False, False, [], []),
        ('MEG 0112', None, False, False, [], []),
        ('nonsense', None, False, False, [], []),
        # Now also set descriptions.
        (['MEG 0112', 'MEG 0131'], ['Really bad!', 'Even worse.'], False,
         False, [], []),
        ('MEG 0112', 'Really bad!', False, False, [], []),
        # Should raise.
        (['MEG 0112', 'MEG 0131'], ['Really bad!'], False, False, [], []),
        # `datatype='meg`
        (['MEG 0112'], ['Really bad!'], False, False, [], []),
        # Enure we create missing columns.
        ('MEG 0112', 'Really bad!', True, True, [], []),
    ])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_mark_channels(_bids_validate,
                       ch_names, descriptions,
                       drop_status_col, drop_description_col,
                       existing_ch_names, existing_descriptions,
                       tmp_path):
    """Test marking channels of an existing BIDS dataset as "bad"."""
    # Setup: Create a fresh BIDS dataset.
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg',
                                         suffix='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    raw = _read_raw_fif(raw_fname, verbose=False)
    raw.info['bads'] = []
    write_raw_bids(raw, bids_path=bids_path, events_data=events,
                   event_id=event_id, verbose=False)

    channels_fname = _find_matching_sidecar(bids_path, suffix='channels',
                                            extension='.tsv')

    if drop_status_col:
        # Remove `status` column from the sidecare TSV file.
        tsv_data = _from_tsv(channels_fname)
        del tsv_data['status']
        _to_tsv(tsv_data, channels_fname)

    if drop_description_col:
        # Remove `status_description` column from the sidecare TSV file.
        tsv_data = _from_tsv(channels_fname)
        del tsv_data['status_description']
        _to_tsv(tsv_data, channels_fname)

    # Test that we raise if number of channels doesn't match number of
    # descriptions.
    if (descriptions is not None and
            len(_ensure_list(ch_names)) != len(_ensure_list(descriptions))):
        with pytest.raises(ValueError, match='must match'):
            mark_channels(ch_names=ch_names, descriptions=descriptions,
                          bids_path=bids_path, status='bad',
                          verbose=False)
        return

    # Test that we raise if we encounter an unknown channel name.
    if any([ch_name not in raw.ch_names
            for ch_name in _ensure_list(ch_names)]):
        with pytest.raises(ValueError, match='not found in dataset'):
            mark_channels(ch_names=ch_names, descriptions=descriptions,
                          bids_path=bids_path, status='bad', verbose=False)
        return

    # Mark `existing_ch_names` as bad in raw and sidecar TSV before we
    # begin our actual tests, which should then add additional channels
    # to the list of bads, retaining the ones we're specifying here.
    mark_channels(ch_names=[],
                  bids_path=bids_path, status='good',
                  verbose=False)
    _bids_validate(bids_root)
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    # Order is not preserved
    assert set(existing_ch_names) == set(raw.info['bads'])
    del raw

    mark_channels(ch_names=ch_names, descriptions=descriptions,
                  bids_path=bids_path, status='bad', verbose=False)
    _bids_validate(bids_root)
    raw = read_raw_bids(bids_path=bids_path, verbose=False)

    # expected bad channels and descriptions just get appended
    expected_bads = (_ensure_list(ch_names) +
                     _ensure_list(existing_ch_names))
    expected_descriptions = (_ensure_list(descriptions) +
                             _ensure_list(existing_descriptions))

    # Order is not preserved
    assert len(expected_bads) == len(raw.info['bads'])
    assert set(expected_bads) == set(raw.info['bads'])

    # Descriptions are not mapped to Raw, so let's check the TSV contents
    # directly.
    tsv_data = _from_tsv(channels_fname)
    assert 'status' in tsv_data
    assert 'status_description' in tsv_data
    for description in expected_descriptions:
        assert description in tsv_data['status_description']


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_mark_channel_roundtrip(tmp_path):
    """Test marking channels fulfills roundtrip."""
    # Setup: Create a fresh BIDS dataset.
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg',
                                         suffix='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, events_data=events,
                   event_id=event_id, verbose=False)
    channels_fname = _find_matching_sidecar(bids_path, suffix='channels',
                                            extension='.tsv')

    ch_names = raw.ch_names
    # first mark all channels as good
    mark_channels(bids_path, ch_names=[], status='good', verbose=False)
    tsv_data = _from_tsv(channels_fname)
    assert all(status == 'good' for status in tsv_data['status'])

    # now mark some bad channels
    mark_channels(bids_path, ch_names=ch_names[:5], status='bad',
                  verbose=False)
    tsv_data = _from_tsv(channels_fname)
    status = tsv_data['status']
    assert all(status_ == 'bad' for status_ in status[:5])
    assert all(status_ == 'good' for status_ in status[5:])

    # now mark them good again
    mark_channels(bids_path, ch_names=ch_names[:5], status='good',
                  verbose=False)
    tsv_data = _from_tsv(channels_fname)
    assert all(status == 'good' for status in tsv_data['status'])


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_error_mark_channels(tmp_path):
    """Test errors when marking channels."""
    # Setup: Create a fresh BIDS dataset.
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg',
                                         suffix='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, events_data=events,
                   event_id=event_id, verbose=False)

    ch_names = raw.ch_names

    with pytest.raises(ValueError, match='Setting the status'):
        mark_channels(ch_names=ch_names, bids_path=bids_path,
                      status='test')


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_mark_channels_files(tmp_path):
    """Test validity of bad channel writing."""
    # BV
    bids_root = tmp_path / 'bids1'
    data_path = op.join(testing.data_path(), 'montage')
    raw_fname = op.join(data_path, 'bv_dig_test.vhdr')

    raw = _read_raw_brainvision(raw_fname)
    raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})

    # inject a bad channel
    assert not raw.info['bads']
    injected_bad = ['Fp1']
    raw.info['bads'] = injected_bad

    bids_path = _bids_path.copy().update(root=bids_root)

    # write with injected bad channels
    write_raw_bids(raw, bids_path, overwrite=True)

    # mark bad channels that get stored as uV in write_brain_vision
    bads = ['CP5', 'CP6']
    mark_channels(bids_path=bids_path, ch_names=bads, status='bad')
    raw.info['bads'].extend(bads)

    # the raw data should match if you drop the bads
    raw_2 = read_raw_bids(bids_path)
    raw.drop_channels(raw.info['bads'])
    raw_2.drop_channels(raw_2.info['bads'])
    assert_array_almost_equal(raw.get_data(), raw_2.get_data())

    # test EDF too
    dir_name = 'EDF'
    fname = 'test_reduced.edf'
    bids_root = tmp_path / 'bids2'
    bids_path = _bids_path.copy().update(root=bids_root)
    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)
    raw = _read_raw_edf(raw_fname)
    write_raw_bids(raw, bids_path, overwrite=True)
    mark_channels(bids_path=bids_path, ch_names=raw.ch_names[0],
                  status='bad')


def test_write_meg_calibration(_bids_validate, tmp_path):
    """Test writing of the Elekta/Neuromag fine-calibration file."""
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root)

    data_path = Path(testing.data_path())

    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, verbose=False)

    fine_cal_fname = data_path / 'SSS' / 'sss_cal_mgh.dat'

    # Test passing a filename.
    write_meg_calibration(calibration=fine_cal_fname,
                          bids_path=bids_path)
    _bids_validate(bids_root)

    # Test passing a dict.
    calibration = mne.preprocessing.read_fine_calibration(fine_cal_fname)
    write_meg_calibration(calibration=calibration,
                          bids_path=bids_path)
    _bids_validate(bids_root)

    # Test passing in incompatible dict.
    calibration = mne.preprocessing.read_fine_calibration(fine_cal_fname)
    del calibration['locs']
    with pytest.raises(ValueError, match='not .* proper fine-calibration'):
        write_meg_calibration(calibration=calibration,
                              bids_path=bids_path)

    # subject not set.
    bids_path = bids_path.copy().update(root=bids_root, subject=None)
    with pytest.raises(ValueError, match='must have root and subject set'):
        write_meg_calibration(fine_cal_fname, bids_path)

    # root not set.
    bids_path = bids_path.copy().update(subject='01', root=None)
    with pytest.raises(ValueError, match='must have root and subject set'):
        write_meg_calibration(fine_cal_fname, bids_path)

    # datatype is not 'meg.
    bids_path = bids_path.copy().update(subject='01', root=bids_root,
                                        datatype='eeg')
    with pytest.raises(ValueError, match='Can only write .* for MEG'):
        write_meg_calibration(fine_cal_fname, bids_path)


def test_write_meg_crosstalk(_bids_validate, tmp_path):
    """Test writing of the Elekta/Neuromag fine-calibration file."""
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root)
    data_path = Path(testing.data_path())

    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, verbose=False)

    crosstalk_fname = data_path / 'SSS' / 'ct_sparse.fif'

    write_meg_crosstalk(fname=crosstalk_fname, bids_path=bids_path)
    _bids_validate(bids_root)

    # subject not set.
    bids_path = bids_path.copy().update(root=bids_root, subject=None)
    with pytest.raises(ValueError, match='must have root and subject set'):
        write_meg_crosstalk(crosstalk_fname, bids_path)

    # root not set.
    bids_path = bids_path.copy().update(subject='01', root=None)
    with pytest.raises(ValueError, match='must have root and subject set'):
        write_meg_crosstalk(crosstalk_fname, bids_path)

    # datatype is not 'meg'.
    bids_path = bids_path.copy().update(subject='01', root=bids_root,
                                        datatype='eeg')
    with pytest.raises(ValueError, match='Can only write .* for MEG'):
        write_meg_crosstalk(crosstalk_fname, bids_path)


@pytest.mark.parametrize(
    'bad_segments',
    [False, 'add', 'only']
)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_annotations(_bids_validate, bad_segments, tmp_path):
    """Test that Annotations are stored as events."""
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    events = mne.read_events(events_fname)
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    event_desc = dict(zip(event_id.values(), event_id.keys()))

    raw = _read_raw_fif(raw_fname)
    annotations = mne.annotations_from_events(
        events=events, sfreq=raw.info['sfreq'], event_desc=event_desc,
        orig_time=raw.info['meas_date']
    )
    if bad_segments:
        bad_annots = mne.Annotations(
            # Try to avoid rounding errors.
            onset=(annotations.onset[0] + 1 / raw.info['sfreq'] * 600,
                   annotations.onset[0] + 1 / raw.info['sfreq'] * 3000),
            duration=(1 / raw.info['sfreq'] * 750,
                      1 / raw.info['sfreq'] * 550),
            description=('BAD_segment', 'BAD_segment'),
            orig_time=annotations.orig_time)

        if bad_segments == 'add':
            annotations += bad_annots
        elif bad_segments == 'only':
            annotations = bad_annots
        else:
            raise ValueError('Unknown `bad_segments` test parameter passed.')
        del bad_annots

    raw.set_annotations(annotations)
    write_raw_bids(raw, bids_path, events_data=None, event_id=None,
                   overwrite=False)

    annotations_read = read_raw_bids(bids_path=bids_path).annotations
    assert_array_almost_equal(annotations.onset, annotations_read.onset)
    assert_array_almost_equal(annotations.duration, annotations_read.duration)
    assert_array_equal(annotations.description, annotations_read.description)
    assert annotations.orig_time == annotations_read.orig_time
    _bids_validate(bids_root)


@pytest.mark.parametrize(
    'drop_undescribed_events',
    [True, False]
)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_undescribed_events(_bids_validate, drop_undescribed_events, tmp_path):
    """Test we're behaving correctly if event descriptions are missing."""
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    events = mne.read_events(events_fname)
    if drop_undescribed_events:
        mask = events[:, 2] != 0
        assert sum(mask) > 0  # Make sure we're actually about to drop sth.!
        events = events[mask]
        del mask

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

    raw = _read_raw_fif(raw_fname)
    raw.set_annotations(None)  # Make sure it's clean.
    kwargs = dict(raw=raw, bids_path=bids_path, events_data=events,
                  event_id=event_id, overwrite=False)

    if not drop_undescribed_events:
        with pytest.raises(ValueError, match='No description was specified'):
            write_raw_bids(**kwargs)
        return
    else:
        write_raw_bids(**kwargs)

    raw_read = read_raw_bids(bids_path=bids_path)
    events_read, event_id_read = mne.events_from_annotations(
        raw=raw_read, event_id=event_id, regexp=None
    )

    assert_array_equal(events, events_read)
    assert event_id == event_id_read
    _bids_validate(bids_root)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_event_storage(tmp_path):
    """Test we're retaining the original event IDs when storing events."""
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')
    events_tsv_fname = (bids_path.copy()
                        .update(suffix='events', extension='.tsv'))

    events = mne.read_events(events_fname)
    events = events[events[:, -1] != 0]  # Drop unused events
    # Change an event ID
    idx = np.where(events[:, -1] == 1)[0]
    events[idx, -1] = 123

    event_id = {'Auditory/Left': 123, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

    raw = _read_raw_fif(raw_fname)
    write_raw_bids(raw=raw, bids_path=bids_path, events_data=events,
                   event_id=event_id, overwrite=False)

    events_tsv = _from_tsv(events_tsv_fname)
    assert set(int(e) for e in events_tsv['value']) == set(event_id.values())


@pytest.mark.parametrize(
    'dir_name, fname, reader, datatype, coord_frame', [
        ('EDF', 'test_reduced.edf', _read_raw_edf, 'ieeg', 'mni_tal'),
        ('EDF', 'test_reduced.edf', _read_raw_edf, 'ieeg', 'mri'),
        ('EDF', 'test_reduced.edf', _read_raw_edf, 'eeg', 'head'),
        ('EDF', 'test_reduced.edf', _read_raw_edf, 'eeg', 'mri'),
        ('EDF', 'test_reduced.edf', _read_raw_edf, 'eeg', 'unknown'),
        ('CTF', 'testdata_ctf.ds', _read_raw_ctf, 'meg', ''),
        ('MEG', 'sample/sample_audvis_trunc_raw.fif', _read_raw_fif, 'meg', ''),  # noqa
    ]
)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
@pytest.mark.filterwarnings(warning_str['encountered_data_in'])
@pytest.mark.filterwarnings(warning_str['nasion_not_found'])
def test_coordsystem_json_compliance(
        dir_name, fname, reader, datatype, coord_frame, tmp_path):
    """Tests that coordsystem.json contents are written correctly.

    Tests multiple manufacturer data formats and MEG, EEG, and iEEG.
    """
    bids_root = tmp_path / 'bids1'
    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    # the BIDS path for test datasets to get written to
    bids_path = _bids_path.copy().update(root=bids_root,
                                         datatype=datatype)

    raw = reader(raw_fname)

    # when passed as a montage, these are ignored so that MNE does
    # not transform back to "head" as it does for internal consistency
    landmarks = dict(nasion=[1, 0, 0], lpa=[0, 1, 0], rpa=[0, 0, 1])

    if datatype == 'eeg':
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    elif datatype == 'ieeg':
        raw.set_channel_types({ch: 'seeg' for ch in raw.ch_names})

    if datatype == 'meg':
        montage = None
    else:
        # alter some channels manually with electrodes to write
        ch_names = raw.ch_names
        elec_locs = np.random.random((len(ch_names), 3)).tolist()
        ch_pos = dict(zip(ch_names, elec_locs))
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                                coord_frame=coord_frame,
                                                **landmarks)
        if datatype == 'eeg':
            raw.set_montage(montage)
            montage = None

    # clean all events for this test
    kwargs = dict(raw=raw, bids_path=bids_path, acpc_aligned=True,
                  montage=montage, overwrite=True, verbose=False)
    # write to BIDS and then check the coordsystem files
    bids_output_path = write_raw_bids(**kwargs)
    coordsystem_fname = _find_matching_sidecar(bids_output_path,
                                               suffix='coordsystem',
                                               extension='.json')
    with open(coordsystem_fname, 'r', encoding='utf-8') as fin:
        coordsystem_json = json.load(fin)

    # writing twice should work as long as the coordsystem
    # contents have not changed
    kwargs.update(bids_path=bids_path.copy().update(run='02'),
                  overwrite=False)
    write_raw_bids(**kwargs)

    datatype_ = {'meg': 'MEG', 'eeg': 'EEG', 'ieeg': 'iEEG'}[datatype]
    # if there is a change in the underlying
    # coordsystem.json file, then an error will occur.
    # upon changing coordsystem contents, and overwrite not True
    # this will fail
    new_coordsystem_json = coordsystem_json.copy()
    new_coordsystem_json[f'{datatype_}CoordinateSystem'] = 'blah'
    _write_json(coordsystem_fname, new_coordsystem_json, overwrite=True)
    kwargs.update(bids_path=bids_path.copy().update(run='03'))
    with pytest.raises(RuntimeError,
                       match='Trying to write coordsystem.json, '
                             'but it already exists'):
        write_raw_bids(**kwargs)
    _write_json(coordsystem_fname, coordsystem_json, overwrite=True)

    if datatype != 'meg':
        electrodes_fname = _find_matching_sidecar(bids_output_path,
                                                  suffix='electrodes',
                                                  extension='.tsv')
        elecs_tsv = _from_tsv(electrodes_fname)

        # electrodes.tsv file, then an error will occur.
        # upon changing electrodes contents, and overwrite not True
        # this will fail
        new_elecs_tsv = elecs_tsv.copy()
        new_elecs_tsv['name'][0] = 'blah'
        _to_tsv(new_elecs_tsv, electrodes_fname)
        kwargs.update(bids_path=bids_path.copy().update(run='04'))
        with pytest.raises(
                RuntimeError, match='Trying to write electrodes.tsv, '
                                    'but it already exists'):
            write_raw_bids(**kwargs)

    # perform checks on the coordsystem.json file itself
    if datatype == 'eeg' and coord_frame == 'head':
        assert coordsystem_json['EEGCoordinateSystem'] == 'CapTrak'
        assert coordsystem_json['EEGCoordinateSystemDescription'] == \
            BIDS_COORD_FRAME_DESCRIPTIONS['captrak']
    elif datatype == 'eeg' and coord_frame == 'unknown':
        assert coordsystem_json['EEGCoordinateSystem'] == 'CapTrak'
        assert coordsystem_json['EEGCoordinateSystemDescription'] == \
            BIDS_COORD_FRAME_DESCRIPTIONS['captrak']
    elif datatype == 'ieeg' and coord_frame == 'mni_tal':
        assert 'space-fsaverage' in coordsystem_fname
        assert coordsystem_json['iEEGCoordinateSystem'] == 'fsaverage'
        assert coordsystem_json['iEEGCoordinateSystemDescription'] == \
            BIDS_COORD_FRAME_DESCRIPTIONS['fsaverage']
    elif datatype == 'ieeg' and coord_frame == 'mri':
        assert 'space-ACPC' in coordsystem_fname
        assert coordsystem_json['iEEGCoordinateSystem'] == 'ACPC'
        assert coordsystem_json['iEEGCoordinateSystemDescription'] == \
            BIDS_COORD_FRAME_DESCRIPTIONS['acpc']
    elif datatype == 'ieeg' and coord_frame == 'unknown':
        assert coordsystem_json['iEEGCoordinateSystem'] == 'Other'
        assert coordsystem_json['iEEGCoordinateSystemDescription'] == 'n/a'
    elif datatype == 'meg' and dir_name == 'CTF':
        assert coordsystem_json['MEGCoordinateSystem'] == 'CTF'
        assert coordsystem_json['MEGCoordinateSystemDescription'] == \
            BIDS_COORD_FRAME_DESCRIPTIONS['ctf']
    elif datatype == 'meg' and dir_name == 'MEG':
        assert coordsystem_json['MEGCoordinateSystem'] == 'ElektaNeuromag'
        assert coordsystem_json['MEGCoordinateSystemDescription'] == \
            BIDS_COORD_FRAME_DESCRIPTIONS['elektaneuromag']


@pytest.mark.parametrize(
    'subject, dir_name, fname, reader', [
        ('01', 'EDF', 'test_reduced.edf', _read_raw_edf),
        ('02', 'Persyst', 'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay', _read_raw_persyst),  # noqa
        ('03', 'NihonKohden', 'MB0400FU.EEG', _read_raw_nihon),
        ('emptyroom', 'MEG/sample',
         'sample_audvis_trunc_raw.fif', _read_raw_fif),
    ]
)
@pytest.mark.filterwarnings(
    warning_str['encountered_data_in'],
    warning_str['channel_unit_changed'],
    warning_str['edf_warning'],
    warning_str['brainvision_unit']
)
def test_anonymize(subject, dir_name, fname, reader, tmp_path, _bids_validate):
    """Test writing anonymized EDF data."""
    data_path = testing.data_path()

    raw_fname = op.join(data_path, dir_name, fname)

    bids_root = tmp_path / 'bids1'
    raw = reader(raw_fname)
    raw_date = raw.info['meas_date'].strftime('%Y%m%d')

    bids_path = BIDSPath(subject=subject, root=bids_root)

    # handle different edge cases
    if subject == 'emptyroom':
        bids_path.update(task='noise', session=raw_date,
                         suffix='meg', datatype='meg')
    else:
        bids_path.update(task='task', suffix='eeg', datatype='eeg')
    daysback_min, daysback_max = get_anonymization_daysback(raw)
    anonymize = dict(daysback=daysback_min + 1)
    orig_bids_path = bids_path.copy()
    bids_path = \
        write_raw_bids(raw, bids_path, overwrite=True,
                       anonymize=anonymize, verbose=False)
    # emptyroom recordings' session should match the recording date
    if subject == 'emptyroom':
        assert (
            bids_path.session ==
            (raw.info['meas_date'] -
             timedelta(days=anonymize['daysback'])).strftime('%Y%m%d')
        )

    raw2 = read_raw_bids(bids_path, verbose=False)
    if raw_fname.endswith('.edf'):
        _raw = reader(bids_path)
        assert _raw.info['meas_date'].year == 1985
        assert _raw.info['meas_date'].month == 1
        assert _raw.info['meas_date'].day == 1
    assert raw2.info['meas_date'].year < 1925

    # write without source
    scans_fname = BIDSPath(subject=bids_path.subject,
                           session=bids_path.session,
                           suffix='scans', extension='.tsv',
                           root=bids_path.root)
    anonymize['keep_source'] = False
    bids_path = \
        write_raw_bids(raw, orig_bids_path, overwrite=True,
                       anonymize=anonymize, verbose=False)
    scans_tsv = _from_tsv(scans_fname)
    assert 'source' not in scans_tsv.keys()

    # Write with source this time get the scans tsv
    bids_path = write_raw_bids(
        raw, orig_bids_path, overwrite=True,
        anonymize=dict(daysback=daysback_min, keep_source=True),
        verbose=False)
    scans_fname = BIDSPath(subject=bids_path.subject,
                           session=bids_path.session,
                           suffix='scans', extension='.tsv',
                           root=bids_path.root)
    scans_tsv = _from_tsv(scans_fname)
    assert scans_tsv['source'] == [
        Path(f).name for f in raw.filenames
    ]
    _bids_validate(bids_path.root)

    # update the scans sidecar JSON with information
    scans_json_fpath = scans_fname.copy().update(extension='.json')
    with open(scans_json_fpath, 'r') as fin:
        scans_json = json.load(fin)
    scans_json['test'] = 'New stuff...'
    update_sidecar_json(scans_json_fpath, scans_json)

    # write again and make sure scans json was not altered
    bids_path = write_raw_bids(
        raw, orig_bids_path, overwrite=True,
        anonymize=dict(daysback=daysback_min, keep_source=True),
        verbose=False)
    with open(scans_json_fpath, 'r') as fin:
        scans_json = json.load(fin)
    assert 'test' in scans_json


@pytest.mark.parametrize('dir_name, fname', [
    ['EDF', 'test_reduced.edf'],
    ['BDF', 'test_bdf_stim_channel.bdf']
])
def test_write_uppercase_edfbdf(tmp_path, dir_name, fname):
    """Test writing uppercase EDF/BDF ext results in lowercase."""
    subject = 'cap'
    if dir_name == 'EDF':
        read_func = _read_raw_edf
    elif dir_name == 'BDF':
        read_func = _read_raw_bdf

    data_path = testing.data_path()
    raw_fname = op.join(data_path, dir_name, fname)

    # capitalize the extension file
    lower_case_ext = f'.{dir_name.lower()}'
    upper_case_ext = f'.{dir_name.upper()}'
    new_basename = (op.basename(raw_fname).split(lower_case_ext)[0] +
                    upper_case_ext)
    new_raw_fname = tmp_path / new_basename
    sh.copyfile(raw_fname, new_raw_fname)
    raw_fname = new_raw_fname.as_posix()

    # now read in the file and write to BIDS
    bids_root = tmp_path / 'bids1'
    raw = read_func(raw_fname)
    bids_path = BIDSPath(subject=subject, task=task, root=bids_root)
    bids_path = write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    # the final output file should have lower case EDF extension
    assert bids_path.extension == lower_case_ext


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_sidecar_encoding(_bids_validate, tmp_path):
    """Test we're properly encoding text as UTF8."""
    bids_root = tmp_path / 'bids1'
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw = _read_raw_fif(raw_fname)
    events = mne.read_events(events_fname)
    event_desc = {1: 'döner', 2: 'bøfsandwich'}
    annotations = mne.annotations_from_events(
        events=events, sfreq=raw.info['sfreq'], event_desc=event_desc,
        orig_time=raw.info['meas_date']
    )

    raw.set_annotations(annotations)
    write_raw_bids(raw, bids_path=bids_path, verbose=False)
    _bids_validate(bids_root)

    # TSV files should be written with a BOM
    for tsv_file in bids_path.root.rglob('*.tsv'):
        with open(tsv_file, 'r', encoding='utf-8') as f:
            x = f.read()
        assert x[0] == codecs.BOM_UTF8.decode('utf-8')

    # Readme should be written with a BOM
    with open(bids_path.root / 'README', 'r', encoding='utf-8') as f:
        x = f.read()
    assert x[0] == codecs.BOM_UTF8.decode('utf-8')

    # JSON files should be written without a BOM
    for json_file in bids_path.root.rglob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            x = f.read()
        assert x[0] != codecs.BOM_UTF8.decode('utf-8')

    # Unicode event names should be written correctly
    events_tsv_fname = (bids_path.copy()
                        .update(suffix='events', extension='.tsv')
                        .match()[0])
    with open(str(events_tsv_fname), 'r', encoding='utf-8-sig') as f:
        x = f.read()
    assert 'döner' in x
    assert 'bøfsandwich' in x

    # Read back the data
    raw_read = read_raw_bids(bids_path)
    assert_array_equal(raw.annotations.description,
                       raw_read.annotations.description)


@requires_version('mne', '0.24')
@requires_version('pybv', '0.6')
@pytest.mark.parametrize(
    'dir_name, format, fname, reader', test_converteeg_data)
@pytest.mark.filterwarnings(
    warning_str['channel_unit_changed'], warning_str['edfblocks'])
def test_convert_eeg_formats(dir_name, format, fname, reader, tmp_path):
    """Test conversion of EEG/iEEG manufacturer fmt to BrainVision/EDF."""
    bids_root = tmp_path / format
    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    # the BIDS path for test datasets to get written to
    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg')

    raw = reader(raw_fname)
    # drop 'misc' type channels when exporting
    raw = raw.pick_types(eeg=True)
    kwargs = dict(raw=raw, format=format, bids_path=bids_path, overwrite=True,
                  verbose=False)

    # test formatting to BrainVision, EDF, or auto (BrainVision)
    if format in ['BrainVision', 'auto']:
        if dir_name == 'NihonKohden':
            with pytest.warns(RuntimeWarning,
                              match='Encountered data in "short" format'):
                bids_output_path = write_raw_bids(**kwargs)
        else:
            with pytest.warns(RuntimeWarning,
                              match='Encountered data in "double" format'):
                bids_output_path = write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Converting data files to EDF format'):
            bids_output_path = write_raw_bids(**kwargs)

    # channel units should stay the same
    raw2 = read_raw_bids(bids_output_path)
    assert all([ch1['unit'] == ch2['unit'] for ch1, ch2 in
                zip(raw.info['chs'], raw2.info['chs'])])
    assert raw2.info['chs'][0]['unit'] == FIFF.FIFF_UNIT_V

    # load channels.tsv; the unit should be Volts
    channels_fname = bids_output_path.copy().update(
        suffix='channels', extension='.tsv')
    channels_tsv = _from_tsv(channels_fname)
    assert channels_tsv['units'][0] == 'V'

    if format == 'BrainVision':
        assert raw2.filenames[0].endswith('.eeg')
        assert bids_output_path.extension == '.vhdr'
    elif format == 'EDF':
        assert raw2.filenames[0].endswith('.edf')
        assert bids_output_path.extension == '.edf'

    orig_len = len(raw)
    assert_allclose(raw.times, raw2.times[:orig_len], atol=1e-5, rtol=0)
    assert_array_equal(raw.ch_names, raw2.ch_names)
    assert raw.get_channel_types() == raw2.get_channel_types()

    # writing to EDF is not 100% lossless, as the resolution is determined
    # by the physical min/max. The precision is to 0.09 uV.
    assert_array_almost_equal(
        raw.get_data(), raw2.get_data()[:, :orig_len], decimal=6)


@requires_version('mne', '0.22')
@pytest.mark.parametrize(
    'dir_name, format, fname, reader', test_converteeg_data)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_error_write_meg_as_eeg(dir_name, format, fname, reader, tmp_path):
    """Test error writing as BrainVision EEG data for MEG."""
    bids_root = tmp_path / 'bids1'
    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg',
                                         extension='.vhdr')
    raw = reader(raw_fname)
    kwargs = dict(raw=raw, format='auto',
                  bids_path=bids_path.update(datatype='meg'))

    # if we accidentally add MEG channels, then an error will occur
    raw.set_channel_types({raw.info['ch_names'][0]: 'mag'})
    with pytest.raises(ValueError, match='Got file extension .*'
                                         'for MEG data'):
        write_raw_bids(**kwargs)


@pytest.mark.parametrize(
    'dir_name, format, fname, reader', test_convertmeg_data)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_convert_meg_formats(dir_name, format, fname, reader, tmp_path):
    """Test conversion of MEG manufacturer format to FIF."""
    bids_root = tmp_path / format
    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    # the BIDS path for test datasets to get written to
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')

    raw = reader(raw_fname)
    kwargs = dict(raw=raw, format=format, bids_path=bids_path, overwrite=True,
                  verbose=False)

    # test formatting to FIF, or auto (FIF)
    bids_output_path = write_raw_bids(**kwargs)

    # channel units should stay the same
    raw2 = read_raw_bids(bids_output_path)

    if format == 'FIF':
        assert raw2.filenames[0].endswith('.fif')
        assert bids_output_path.extension == '.fif'

    orig_len = len(raw)
    assert_allclose(raw.times, raw2.times[:orig_len], atol=1e-5, rtol=0)
    assert_array_equal(raw.ch_names, raw2.ch_names)
    assert raw.get_channel_types() == raw2.get_channel_types()
    assert_array_almost_equal(
        raw.get_data(), raw2.get_data()[:, :orig_len], decimal=3)


@pytest.mark.parametrize('dir_name, fname, reader', test_convert_data)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_convert_raw_errors(dir_name, fname, reader, tmp_path):
    """Test errors when converting raw file formats."""
    bids_root = tmp_path / 'bids_1'

    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    # the BIDS path for test datasets to get written to
    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg')

    # test conversion to BrainVision/FIF
    raw = reader(raw_fname)
    kwargs = dict(raw=raw, bids_path=bids_path, overwrite=True)

    # only accepted keywords will work for the 'format' parameter
    with pytest.raises(ValueError, match='The input "format" .* is '
                                         'not an accepted input format for '
                                         '`write_raw_bids`'):
        kwargs['format'] = 'blah'
        write_raw_bids(**kwargs)

    # write should fail when trying to convert to wrong data format for
    # the datatype inside the file (e.g. EEG -> 'FIF' or MEG -> 'BrainVision')
    with pytest.raises(ValueError, match='The input "format" .* is not an '
                                         'accepted input format for '
                                         '.* datatype.'):
        if dir_name == 'CTF':
            new_format = 'BrainVision'
        else:
            new_format = 'FIF'
        kwargs['format'] = new_format
        write_raw_bids(**kwargs)


def test_write_fif_triux(tmp_path):
    """Test writing Triux files."""
    data_path = testing.data_path()
    triux_path = op.join(data_path, 'SSS', 'TRIUX')
    tri_fname = op.join(triux_path, 'triux_bmlhus_erm_raw.fif')
    raw = mne.io.read_raw_fif(tri_fname)
    bids_path = BIDSPath(
        subject="01", task="task", session="01", run="01", datatype="meg",
        root=tmp_path
    )
    write_raw_bids(raw, bids_path=bids_path, overwrite=True)


@pytest.mark.filterwarnings(warning_str['nasion_not_found'])
@pytest.mark.parametrize('datatype', ['eeg', 'ieeg'])
def test_write_extension_case_insensitive(_bids_validate, tmp_path, datatype):
    """Test writing files is case insensitive."""
    dir_name, fname, reader = 'EDF', 'test_reduced.edf', _read_raw_edf

    bids_root = tmp_path / 'bids1'
    source_path = Path(bids_root) / 'sourcedata'
    data_path = op.join(testing.data_path(), dir_name)
    sh.copytree(data_path, source_path)
    data_path = source_path

    # rename extension to upper-case
    _fname, ext = _parse_ext(fname)
    new_fname = _fname + ext.upper()

    # rename the file's extension
    raw_fname = op.join(data_path, fname)
    new_raw_fname = op.join(data_path, new_fname)
    os.rename(raw_fname, new_raw_fname)

    # the BIDS path for test datasets to get written to
    raw = reader(new_raw_fname)
    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg')
    write_raw_bids(raw, bids_path)
    raw.set_channel_types({raw.ch_names[i]: 'ecog'
                           for i in mne.pick_types(raw.info, eeg=True)})
    bids_path = _bids_path.copy().update(root=bids_root, datatype='ieeg')
    write_raw_bids(raw, bids_path)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_symlink(tmp_path):
    """Test creation of symbolic links."""
    testing_data_path = Path(testing.data_path())
    raw_trunc_path = (testing_data_path / 'MEG' / 'sample' /
                      'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_trunc_path)
    root = tmp_path / 'symlink'
    bids_path = _bids_path.copy().update(root=root, datatype='meg')
    kwargs = dict(raw=raw, bids_path=bids_path, symlink=True)

    # We currently don't support windows
    if sys.platform in ('win32', 'cygwin'):
        with pytest.raises(NotImplementedError, match='not supported'):
            write_raw_bids(**kwargs)
        return

    # Symlinks & anonymization don't go together
    with pytest.raises(ValueError, match='Cannot create symlinks'):
        write_raw_bids(anonymize=dict(daysback=123), **kwargs)

    # We currently only support FIFF
    raw_eeglab_path = testing_data_path / 'EEGLAB' / 'test_raw.set'
    raw_eeglab = _read_raw_eeglab(raw_eeglab_path)
    bids_path_eeglab = _bids_path.copy().update(root=root, datatype='eeg')
    with pytest.raises(NotImplementedError, match='only.*for FIFF'):
        write_raw_bids(raw=raw_eeglab, bids_path=bids_path_eeglab,
                       symlink=True)

    p = write_raw_bids(raw=raw, bids_path=bids_path, symlink=True)
    assert p.fpath.is_symlink()
    assert p.fpath.resolve() == raw_trunc_path
    read_raw_bids(p)

    # test with split files
    # prepare the split files
    sample_data_path = Path(mne.datasets.sample.data_path())
    raw_path = sample_data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = _read_raw_fif(raw_path).crop(0, 10)

    split_raw_path = tmp_path / 'raw' / 'sample_audivis_raw.fif'
    split_raw_path.parent.mkdir()
    raw.save(split_raw_path, split_size='10MB', split_naming='neuromag')
    raw = _read_raw_fif(split_raw_path)
    assert len(raw.filenames) == 2

    # now actually test the I/O roundtrip
    root = tmp_path / 'symlink-split'
    bids_path = _bids_path.copy().update(root=root, datatype='meg')
    p = write_raw_bids(raw=raw, bids_path=bids_path, symlink=True)
    raw = read_raw_bids(p)
    assert len(raw.filenames) == 2


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_write_associated_emptyroom(_bids_validate, tmp_path):
    """Test functionality of the write_raw_bids conversion for fif."""
    bids_root = tmp_path / 'bids1'
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)
    meas_date = datetime(year=2020, month=1, day=10, tzinfo=timezone.utc)
    raw.set_meas_date(meas_date)

    # First write "empty-room" data
    bids_path_er = BIDSPath(subject='emptyroom', session='20200110',
                            task='noise', root=bids_root, datatype='meg',
                            suffix='meg', extension='.fif')
    write_raw_bids(raw, bids_path=bids_path_er)

    # Now we write experimental data and associate it with the empty-room
    # recording
    bids_path = bids_path_er.copy().update(subject='01', session=None,
                                           task='task')
    write_raw_bids(raw, bids_path=bids_path, empty_room=bids_path_er)
    _bids_validate(bids_path.root)

    meg_json_path = bids_path.copy().update(extension='.json')
    with open(meg_json_path, 'r') as fin:
        meg_json_data = json.load(fin)

    assert 'AssociatedEmptyRoom' in meg_json_data
    assert (bids_path_er.fpath
            .as_posix()  # make test work on Windows, too
            .endswith(meg_json_data['AssociatedEmptyRoom']))
    assert meg_json_data['AssociatedEmptyRoom'].startswith('/')


def test_preload(_bids_validate, tmp_path):
    """Test writing custom preloaded raw objects"""
    bids_root = tmp_path / 'bids'
    bids_path = _bids_path.copy().update(root=bids_root)
    sfreq, n_points = 1024., int(1e6)
    info = mne.create_info(['ch1', 'ch2', 'ch3', 'ch4', 'ch5'], sfreq,
                           ['eeg'] * 5)
    rng = np.random.RandomState(99)
    raw = mne.io.RawArray(rng.random((5, n_points)) * 1e-6, info)
    raw.orig_format = 'single'
    raw.info['line_freq'] = 60

    # reject preloaded by default
    with pytest.raises(ValueError, match='allow_preload'):
        write_raw_bids(raw, bids_path, verbose=False, overwrite=True)

    # preloaded raw must specify format
    with pytest.raises(ValueError, match='format'):
        write_raw_bids(raw, bids_path, allow_preload=True,
                       verbose=False, overwrite=True)

    write_raw_bids(raw, bids_path, allow_preload=True, format='BrainVision',
                   verbose=False, overwrite=True)
    _bids_validate(bids_root)


@pytest.mark.parametrize(
    'dir_name', ('tsv_test', 'json_test')
)
def test_write_raw_special_paths(tmp_path, dir_name):
    """Test writing to locations containing strings with special meaning."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = _read_raw_fif(raw_fname)

    root = tmp_path / dir_name
    bids_path = _bids_path.copy().update(root=root)
    write_raw_bids(raw=raw, bids_path=bids_path)


@requires_nibabel()
def test_anonymize_dataset(_bids_validate, tmpdir):
    """Test creating an anonymized copy of a dataset."""
    # Create a non-anonymized dataset
    bids_root = tmpdir / 'bids'
    bids_path = _bids_path.copy().update(
        root=bids_root, subject='testparticipant', extension='.fif',
        datatype='meg'
    )
    bids_path_er = bids_path.copy().update(
        subject='emptyroom', task='noise', session='20021203', run=None,
        acquisition=None
    )
    bids_path_anat = bids_path.copy().update(
        datatype='anat', suffix='T1w', extension='.nii.gz'
    )

    data_path = Path(testing.data_path())
    raw_path = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    raw_er_path = data_path / 'MEG' / 'sample' / 'ernoise_raw.fif'
    fine_cal_path = data_path / 'SSS' / 'sss_cal_mgh.dat'
    crosstalk_path = data_path / 'SSS' / 'ct_sparse_mgh.fif'
    t1w_path = data_path / 'subjects' / 'sample' / 'mri' / 'T1.mgz'
    mri_landmarks = mne.channels.make_dig_montage(
        lpa=[66.08580, 51.33362, 46.52982],
        nasion=[41.87363, 32.24694, 74.55314],
        rpa=[17.23812, 53.08294, 47.01789],
        coord_frame='mri_voxel'
    )
    events_path = (data_path / 'MEG' / 'sample' /
                   'sample_audvis_trunc_raw-eve.fif')
    event_id = {
        'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
        'Visual/Right': 4, 'Smiley': 5, 'Button': 32,
        'unknown': 0
    }

    raw = _read_raw_fif(raw_path, verbose=False)
    raw_er = _read_raw_fif(raw_er_path, verbose=False)

    write_raw_bids(raw_er, bids_path=bids_path_er)
    write_raw_bids(
        raw, bids_path=bids_path, empty_room=bids_path_er,
        events_data=events_path, event_id=event_id, verbose=False
    )
    write_meg_crosstalk(
        fname=crosstalk_path, bids_path=bids_path, verbose=False
    )
    write_meg_calibration(
        calibration=fine_cal_path, bids_path=bids_path, verbose=False
    )
    write_anat(
        image=t1w_path, bids_path=bids_path_anat, landmarks=mri_landmarks,
        verbose=False
    )
    _bids_validate(bids_root)

    # Now run the actual anonymization
    bids_root_anon = tmpdir / 'bids-anonymized'
    anonymize_dataset(
        bids_root_in=bids_root,
        bids_root_out=bids_root_anon,
        random_state=42
    )
    _bids_validate(bids_root_anon)
    meg_dir = bids_root_anon / 'sub-1' / 'ses-01' / 'meg'
    assert (meg_dir /
            'sub-1_ses-01_task-testing_acq-01_run-01_meg.fif').exists()
    assert (meg_dir / 'sub-1_ses-01_acq-crosstalk_meg.fif').exists()
    assert (meg_dir / 'sub-1_ses-01_acq-calibration_meg.dat').exists()
    assert (bids_root_anon / 'sub-1' / 'ses-01' / 'anat' /
            'sub-1_ses-01_acq-01_T1w.nii.gz').exists()
    assert (bids_root_anon / 'sub-emptyroom' / 'ses-19221211' / 'meg' /
            'sub-emptyroom_ses-19221211_task-noise_meg.fif').exists()

    events_tsv_orig_bp = bids_path.copy().update(
        suffix='events', extension='.tsv'
    )
    events_tsv_anonymized_bp = events_tsv_orig_bp.copy().update(
        subject='1', root=bids_root_anon
    )
    events_tsv_orig = _from_tsv(events_tsv_orig_bp)
    events_tsv_anonymized = _from_tsv(events_tsv_anonymized_bp)
    assert events_tsv_orig == events_tsv_anonymized

    # Explicitly specify multiple data types
    bids_root_anon = tmpdir / 'bids-anonymized-1'
    anonymize_dataset(
        bids_root_in=bids_root,
        bids_root_out=bids_root_anon,
        datatypes=['meg', 'anat'],
        random_state=42
    )
    _bids_validate(bids_root_anon)
    assert (bids_root_anon / 'sub-1' / 'ses-01' / 'meg').exists()
    assert (bids_root_anon / 'sub-1' / 'ses-01' / 'anat').exists()
    assert (bids_root_anon / 'sub-emptyroom').exists()

    # One data type, daysback, subject mapping
    bids_root_anon = tmpdir / 'bids-anonymized-2'
    anonymize_dataset(
        bids_root_in=bids_root,
        bids_root_out=bids_root_anon,
        daysback=10,
        datatypes='meg',
        subject_mapping={
            'testparticipant': '123',
            'emptyroom': 'emptyroom'
        }
    )
    _bids_validate(bids_root_anon)
    assert (bids_root_anon / 'sub-123' / 'ses-01' / 'meg').exists()
    assert not (bids_root_anon / 'sub-123' / 'ses-01' / 'anat').exists()
    assert (bids_root_anon / 'sub-emptyroom' / 'ses-20021123').exists()

    # Unknown subject in subject_mapping
    bids_root_anon = tmpdir / 'bids-anonymized-3'
    with pytest.raises(IndexError, match='does not contain an entry for'):
        anonymize_dataset(
            bids_root_in=bids_root,
            bids_root_out=bids_root_anon,
            subject_mapping={
                'foobar': '123',
                'emptyroom': 'emptyroom'
            }
        )

    # Duplicated entries in subject_mapping
    bids_root_anon = tmpdir / 'bids-anonymized-4'
    with pytest.raises(ValueError, match='dictionary contains duplicated'):
        anonymize_dataset(
            bids_root_in=bids_root,
            bids_root_out=bids_root_anon,
            subject_mapping={
                'testparticipant': '123',
                'foobar': '123',
                'emptyroom': 'emptyroom'
            }
        )

    # bids_root_in does not exist
    bids_root_anon = tmpdir / 'bids-anonymized-5'
    with pytest.raises(FileNotFoundError, match='directory does not exist'):
        anonymize_dataset(
            bids_root_in='/foobar',
            bids_root_out=bids_root_anon
        )

    # input dir == output dir
    with pytest.raises(ValueError, match='directory must differ'):
        anonymize_dataset(
            bids_root_in=bids_root,
            bids_root_out=bids_root
        )

    # bids_root_out exists
    bids_root_anon = tmpdir / 'bids-anonymized-6'
    bids_root_anon.mkdir()
    with pytest.raises(FileExistsError, match='directory already exists'):
        anonymize_dataset(
            bids_root_in=bids_root,
            bids_root_out=bids_root_anon
        )

    # Unsupported data type
    bids_root_anon = tmpdir / 'bids-anonymized-7'
    with pytest.raises(ValueError, match='Unsupported data type'):
        anonymize_dataset(
            bids_root_in=bids_root,
            bids_root_out=bids_root_anon,
            datatypes='func'
        )

    # subject_mapping None
    bids_root_anon = tmpdir / 'bids-anonymized-8'
    anonymize_dataset(
        bids_root_in=bids_root,
        bids_root_out=bids_root_anon,
        datatypes='meg',
        subject_mapping=None
    )
    _bids_validate(bids_root_anon)
    assert (bids_root_anon / 'sub-testparticipant').exists()
    assert (bids_root_anon / 'sub-emptyroom').exists()

    # subject_mapping callable
    bids_root_anon = tmpdir / 'bids-anonymized-9'
    anonymize_dataset(
        bids_root_in=bids_root,
        bids_root_out=bids_root_anon,
        datatypes='meg',
        subject_mapping=lambda x: {
            'testparticipant': '123', 'emptyroom': 'emptyroom'
        }
    )
    _bids_validate(bids_root_anon)
    assert (bids_root_anon / 'sub-123').exists()
    assert (bids_root_anon / 'sub-emptyroom').exists()

    # Rename emptyroom
    bids_root_anon = tmpdir / 'bids-anonymized-10'
    with pytest.warns(
        RuntimeWarning,
        match='requested to change the "emptyroom" subject ID'
    ):
        anonymize_dataset(
            bids_root_in=bids_root,
            bids_root_out=bids_root_anon,
            datatypes='meg',
            subject_mapping={
                'testparticipant': 'testparticipant',
                'emptyroom': 'emptiestroom'
            }
        )
    _bids_validate(bids_root)
    assert (bids_root_anon / 'sub-testparticipant').exists()
    assert (bids_root_anon / 'sub-emptiestroom').exists()

    # Only anat data
    bids_root_anon = tmpdir / 'bids-anonymized-11'
    anonymize_dataset(
        bids_root_in=bids_root,
        bids_root_out=bids_root_anon,
        datatypes='anat'
    )
    _bids_validate(bids_root_anon)
    assert (bids_root_anon / 'sub-1' / 'ses-01' / 'anat').exists()
    assert not (bids_root_anon / 'sub-1' / 'ses-01' / 'meg').exists()

    # Ensure that additional JSON sidecar fields are transferred if they are
    # "safe", and are omitted if they are not whitelisted
    bids_path.datatype = 'meg'
    meg_json_path = bids_path.copy().update(extension='.json')
    meg_json = json.loads(meg_json_path.fpath.read_text(encoding='utf-8'))
    assert 'Instructions' not in meg_json  # ensure following test makes sense
    meg_json['Instructions'] = 'Foo'
    meg_json['UnknownKey'] = 'Bar'
    meg_json_path.fpath.write_text(
        data=json.dumps(meg_json),
        encoding='utf-8'
    )

    # After anonymization, "Instructions" should be there and "UnknownKey"
    # should be gone.
    bids_root_anon = tmpdir / 'bids-anonymized-12'
    anonymize_dataset(
        bids_root_in=bids_root,
        bids_root_out=bids_root_anon,
        datatypes='meg'
    )
    path = (bids_root_anon / 'sub-1' / 'ses-01' / 'meg' /
            'sub-1_ses-01_task-testing_acq-01_run-01_meg.json')
    meg_json = json.loads(path.read_text(encoding='utf=8'))
    assert 'Instructions' in meg_json
    assert 'UnknownKey' not in meg_json


def test_anonymize_dataset_daysback(tmpdir):
    """Test some bits of _get_daysback, which doesn't have a public API."""
    # Check progress bar output
    from mne_bids.write import _get_daysback

    bids_root = tmpdir / 'bids'
    bids_path = _bids_path.copy().update(
        root=bids_root, subject='testparticipant', datatype='meg'
    )
    data_path = Path(testing.data_path())
    raw_path = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    raw = _read_raw_fif(raw_path, verbose=False)
    write_raw_bids(raw, bids_path=bids_path)

    _get_daysback(
        bids_paths=[bids_path],
        rng=np.random.default_rng(),
        show_progress_thresh=1
    )

    # Multiple runs
    _get_daysback(
        bids_paths=[
            bids_path.copy().update(run='01'),
            bids_path.copy().update(run='02')
        ],
        rng=np.random.default_rng(),
        show_progress_thresh=20
    )

    # Multiple sessions
    bids_root = tmpdir / 'bids-multisession'
    bids_path = _bids_path.copy().update(
        root=bids_root, subject='testparticipant', datatype='meg'
    )
    write_raw_bids(raw, bids_path=bids_path.copy().update(session='01'))
    write_raw_bids(raw, bids_path=bids_path.copy().update(session='02'))

    _get_daysback(
        bids_paths=[
            bids_path.copy().update(session='01'),
            bids_path.copy().update(session='02')
        ],
        rng=np.random.default_rng(),
        show_progress_thresh=20
    )
