"""Testing utilities for file io."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
import json
import os
import os.path as op
import pathlib
from datetime import datetime, timezone

import pytest
import shutil as sh
import numpy as np
from numpy.testing import assert_almost_equal

import mne
from mne.io.constants import FIFF
from mne.utils import requires_nibabel, object_diff, requires_version
from mne.utils import assert_dig_allclose
from mne.datasets import testing, somato

from mne_bids import BIDSPath
from mne_bids.config import MNE_STR_TO_FRAME
from mne_bids.read import (read_raw_bids,
                           _read_raw, get_head_mri_trans,
                           _handle_events_reading)
from mne_bids.tsv_handler import _to_tsv, _from_tsv
from mne_bids.utils import (_write_json)
from mne_bids.sidecar_updates import _update_sidecar
from mne_bids.path import _find_matching_sidecar
from mne_bids.write import write_anat, write_raw_bids, get_anat_landmarks

subject_id = '01'
session_id = '01'
run = '01'
acq = '01'
task = 'testing'

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)

_bids_path_minimal = BIDSPath(subject=subject_id, task=task)

# Get the MNE testing sample data - USA
data_path = testing.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_raw.fif')

# Get the MNE somato data - EU
somato_path = somato.data_path()
somato_raw_fname = op.join(somato_path, 'sub-01', 'meg',
                           'sub-01_task-somato_meg.fif')

# Data with cHPI info
raw_fname_chpi = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')

warning_str = dict(
    channel_unit_changed='ignore:The unit for chann*.:RuntimeWarning:mne',
    meas_date_set_to_none="ignore:.*'meas_date' set to None:RuntimeWarning:"
                          "mne",
    nasion_not_found='ignore:.*nasion not found:RuntimeWarning:mne',
    maxshield='ignore:.*Internal Active Shielding:RuntimeWarning:mne'
)


def _wrap_read_raw(read_raw):
    def fn(fname, *args, **kwargs):
        raw = read_raw(fname, *args, **kwargs)
        raw.info['line_freq'] = 60
        return raw
    return fn


_read_raw_fif = _wrap_read_raw(mne.io.read_raw_fif)
_read_raw_ctf = _wrap_read_raw(mne.io.read_raw_ctf)
_read_raw_edf = _wrap_read_raw(mne.io.read_raw_edf)


def test_read_raw():
    """Test the raw reading."""
    # Use a file ending that does not exist
    f = 'file.bogus'
    with pytest.raises(ValueError, match='file name extension must be one of'):
        _read_raw(f)


def test_not_implemented(tmp_path):
    """Test the not yet implemented data formats raise an adequate error."""
    for not_implemented_ext in ['.mef', '.nwb']:
        raw_fname = tmp_path / f'test{not_implemented_ext}'
        with open(raw_fname, 'w', encoding='utf-8'):
            pass
        with pytest.raises(ValueError, match=('there is no IO support for '
                                              'this file format yet')):
            _read_raw(raw_fname)


def test_read_correct_inputs():
    """Test that inputs of read functions are correct."""
    bids_path = 'sub-01_ses-01_meg.fif'
    with pytest.raises(RuntimeError, match='"bids_path" must be a '
                                           'BIDSPath object'):
        read_raw_bids(bids_path)

    with pytest.raises(RuntimeError, match='"bids_path" must be a '
                                           'BIDSPath object'):
        get_head_mri_trans(bids_path)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_participants_data(tmp_path):
    """Test reading information from a BIDS sidecar.json file."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg')
    raw = _read_raw_fif(raw_fname, verbose=False)

    # if subject info was set, we don't roundtrip birthday
    # due to possible anonymization in mne-bids
    subject_info = {
        'hand': 1,
        'sex': 2,
    }
    raw.info['subject_info'] = subject_info
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    raw = read_raw_bids(bids_path=bids_path)
    print(raw.info['subject_info'])
    assert raw.info['subject_info']['hand'] == 1
    assert raw.info['subject_info']['sex'] == 2
    assert raw.info['subject_info'].get('birthday', None) is None
    assert raw.info['subject_info']['his_id'] == f'sub-{bids_path.subject}'
    assert 'participant_id' not in raw.info['subject_info']

    # if modifying participants tsv, then read_raw_bids reflects that
    participants_tsv_fpath = tmp_path / 'participants.tsv'
    participants_tsv = _from_tsv(participants_tsv_fpath)
    participants_tsv['hand'][0] = 'n/a'
    _to_tsv(participants_tsv, participants_tsv_fpath)
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info['subject_info']['hand'] == 0
    assert raw.info['subject_info']['sex'] == 2
    assert raw.info['subject_info'].get('birthday', None) is None

    # make sure things are read even if the entries don't make sense
    participants_tsv = _from_tsv(participants_tsv_fpath)
    participants_tsv['hand'][0] = 'righty'
    participants_tsv['sex'][0] = 'malesy'
    _to_tsv(participants_tsv, participants_tsv_fpath)
    with pytest.warns(RuntimeWarning, match='Unable to map'):
        raw = read_raw_bids(bids_path=bids_path)
        assert raw.info['subject_info']['hand'] is None
        assert raw.info['subject_info']['sex'] is None

    # make sure to read in if no participants file
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    os.remove(participants_tsv_fpath)
    with pytest.warns(RuntimeWarning, match='participants.tsv file not found'):
        raw = read_raw_bids(bids_path=bids_path)
        assert raw.info['subject_info'] is None


@pytest.mark.parametrize(
    ('hand_bids', 'hand_mne', 'sex_bids', 'sex_mne'),
    [('Right', 1, 'Female', 2),
     ('RIGHT', 1, 'FEMALE', 2),
     ('R', 1, 'F', 2),
     ('left', 2, 'male', 1),
     ('l', 2, 'm', 1)]
)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_participants_handedness_and_sex_mapping(hand_bids, hand_mne,
                                                      sex_bids, sex_mne,
                                                      tmp_path):
    """Test we're correctly mapping handedness and sex between BIDS and MNE."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg')
    participants_tsv_fpath = tmp_path / 'participants.tsv'
    raw = _read_raw_fif(raw_fname, verbose=False)

    # Avoid that we end up with subject information stored in the raw data.
    raw.info['subject_info'] = {}
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    participants_tsv = _from_tsv(participants_tsv_fpath)
    participants_tsv['hand'][0] = hand_bids
    participants_tsv['sex'][0] = sex_bids
    _to_tsv(participants_tsv, participants_tsv_fpath)

    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info['subject_info']['hand'] is hand_mne
    assert raw.info['subject_info']['sex'] is sex_mne


@requires_nibabel()
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_get_head_mri_trans(tmp_path):
    """Test getting a trans object from BIDS data."""
    import nibabel as nib

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')
    subjects_dir = op.join(data_path, 'subjects')

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    # Write it to BIDS
    raw = _read_raw_fif(raw_fname)
    bids_path = _bids_path.copy().update(root=tmp_path)
    write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                   overwrite=False)

    # We cannot recover trans if no MRI has yet been written
    with pytest.raises(FileNotFoundError, match='Did not find'):
        estimated_trans = get_head_mri_trans(
            bids_path=bids_path, fs_subject='sample',
            fs_subjects_dir=subjects_dir)

    # Write some MRI data and supply a `trans` so that a sidecar gets written
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    t1w_mgh = nib.load(t1w_mgh)

    landmarks = get_anat_landmarks(
        t1w_mgh, raw.info, trans, fs_subject='sample',
        fs_subjects_dir=subjects_dir)
    t1w_bids_path = write_anat(
        t1w_mgh, bids_path=bids_path, landmarks=landmarks, verbose=True)
    anat_dir = bids_path.directory

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(
        bids_path=bids_path, fs_subject='sample', fs_subjects_dir=subjects_dir)

    assert trans['from'] == estimated_trans['from']
    assert trans['to'] == estimated_trans['to']
    assert_almost_equal(trans['trans'], estimated_trans['trans'])

    # provoke an error by introducing NaNs into MEG coords
    raw.info['dig'][0]['r'] = np.full(3, np.nan)
    sh.rmtree(anat_dir)
    bad_landmarks = get_anat_landmarks(t1w_mgh, raw.info, trans, 'sample',
                                       op.join(data_path, 'subjects'))
    write_anat(t1w_mgh, bids_path=t1w_bids_path, landmarks=bad_landmarks)
    with pytest.raises(RuntimeError, match='AnatomicalLandmarkCoordinates'):
        estimated_trans = get_head_mri_trans(bids_path=t1w_bids_path,
                                             fs_subject='sample',
                                             fs_subjects_dir=subjects_dir)

    # test we are permissive for different casings of landmark names in the
    # sidecar, and also accept "nasion" instead of just "NAS"
    raw = _read_raw_fif(raw_fname)
    write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                   overwrite=True)  # overwrite with new acq
    t1w_bids_path = write_anat(t1w_mgh, bids_path=bids_path,
                               landmarks=landmarks, overwrite=True)

    t1w_json_fpath = t1w_bids_path.copy().update(extension='.json').fpath
    with t1w_json_fpath.open('r', encoding='utf-8') as f:
        t1w_json = json.load(f)

    coords = t1w_json['AnatomicalLandmarkCoordinates']
    coords['lpa'] = coords['LPA']
    coords['Rpa'] = coords['RPA']
    coords['Nasion'] = coords['NAS']
    del coords['LPA'], coords['RPA'], coords['NAS']

    _write_json(t1w_json_fpath, t1w_json, overwrite=True)

    estimated_trans = get_head_mri_trans(
        bids_path=bids_path,
        fs_subject='sample', fs_subjects_dir=subjects_dir)
    assert_almost_equal(trans['trans'], estimated_trans['trans'])

    # Test t1_bids_path parameter
    #
    # Case 1: different BIDS roots
    meg_bids_path = _bids_path.copy().update(root=tmp_path / 'meg_root')
    t1_bids_path = _bids_path.copy().update(
        root=tmp_path / 'mri_root', task=None, run=None
    )
    raw = _read_raw_fif(raw_fname)

    write_raw_bids(raw, bids_path=meg_bids_path)
    landmarks = get_anat_landmarks(
        t1w_mgh, raw.info, trans, fs_subject='sample',
        fs_subjects_dir=subjects_dir)
    write_anat(t1w_mgh, bids_path=t1_bids_path, landmarks=landmarks)
    read_trans = get_head_mri_trans(
        bids_path=meg_bids_path, t1_bids_path=t1_bids_path,
        fs_subject='sample', fs_subjects_dir=subjects_dir)
    assert np.allclose(trans['trans'], read_trans['trans'])

    # Case 2: different sessions
    raw = _read_raw_fif(raw_fname)
    meg_bids_path = _bids_path.copy().update(root=tmp_path / 'session_test',
                                             session='01')
    t1_bids_path = meg_bids_path.copy().update(
        session='02', task=None, run=None
    )

    write_raw_bids(raw, bids_path=meg_bids_path)
    write_anat(t1w_mgh, bids_path=t1_bids_path, landmarks=landmarks)
    read_trans = get_head_mri_trans(
        bids_path=meg_bids_path, t1_bids_path=t1_bids_path,
        fs_subject='sample', fs_subjects_dir=subjects_dir)
    assert np.allclose(trans['trans'], read_trans['trans'])

    # Test that incorrect subject directory throws error
    with pytest.raises(ValueError, match='Could not find'):
        estimated_trans = get_head_mri_trans(
            bids_path=bids_path, fs_subject='bad',
            fs_subjects_dir=subjects_dir)


def test_handle_events_reading(tmp_path):
    """Test reading events from a BIDS events.tsv file."""
    # We can use any `raw` for this
    raw = _read_raw_fif(raw_fname)

    # Create an arbitrary events.tsv file, to test we can deal with 'n/a'
    # make sure we can deal w/ "#" characters
    events = {'onset': [11, 12, 'n/a'],
              'duration': ['n/a', 'n/a', 'n/a'],
              'trial_type': ["rec start", "trial #1", "trial #2!"]}
    events_fname = tmp_path / 'bids1' / 'sub-01_task-test_events.json'
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    raw = _handle_events_reading(events_fname, raw)
    events, event_id = mne.events_from_annotations(raw)

    # Test with a `stim_type` column instead of `trial_type`.
    events = {'onset': [11, 12, 'n/a'],
              'duration': ['n/a', 'n/a', 'n/a'],
              'stim_type': ["rec start", "trial #1", "trial #2!"]}
    events_fname = tmp_path / 'bids2' / 'sub-01_task-test_events.json'
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    with pytest.warns(RuntimeWarning, match='This column should be renamed'):
        raw = _handle_events_reading(events_fname, raw)
    events, event_id = mne.events_from_annotations(raw)

    # Test with same `trial_type` referring to different `value`
    events = {'onset': [11, 12, 13],
              'duration': ['n/a', 'n/a', 'n/a'],
              'trial_type': ["event1", "event1", "event2"],
              'value': [1, 2, 3]}
    events_fname = tmp_path / 'bids3' / 'sub-01_task-test_events.json'
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    raw = _handle_events_reading(events_fname, raw)
    events, event_id = mne.events_from_annotations(raw)

    assert len(events) == 3
    assert 'event1/1' in event_id
    assert 'event1/2' in event_id
    # The event with unique value mapping should not be renamed
    assert 'event2' in event_id

    # Test without any kind of event description.
    events = {'onset': [11, 12, 'n/a'],
              'duration': ['n/a', 'n/a', 'n/a']}
    events_fname = tmp_path / 'bids4' / 'sub-01_task-test_events.json'
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    raw = _handle_events_reading(events_fname, raw)
    events, event_id = mne.events_from_annotations(raw)
    ids = list(event_id.keys())
    assert len(ids) == 1
    assert ids == ['n/a']


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_keep_essential_annotations(tmp_path):
    """Test that essential Annotations are not omitted during I/O roundtrip."""
    raw = _read_raw_fif(raw_fname)
    annotations = mne.Annotations(onset=[raw.times[0]], duration=[1],
                                  description=['BAD_ACQ_SKIP'])
    raw.set_annotations(annotations)

    # Write data, remove events.tsv, then try to read again
    bids_path = BIDSPath(subject='01', task='task', datatype='meg',
                         root=tmp_path)
    with pytest.warns(RuntimeWarning, match='Acquisition skips detected'):
        write_raw_bids(raw, bids_path, overwrite=True)

    bids_path.copy().update(suffix='events', extension='.tsv').fpath.unlink()
    raw_read = read_raw_bids(bids_path)

    assert len(raw_read.annotations) == len(raw.annotations) == 1
    assert (raw_read.annotations[0]['description'] ==
            raw.annotations[0]['description'])


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_scans_reading(tmp_path):
    """Test reading data from a BIDS scans.tsv file."""
    raw = _read_raw_fif(raw_fname)
    suffix = "meg"

    # write copy of raw with line freq of 60
    # bids basename and fname
    bids_path = BIDSPath(subject='01', session='01',
                         task='audiovisual', run='01',
                         datatype=suffix,
                         root=tmp_path)
    bids_path = write_raw_bids(raw, bids_path, overwrite=True)
    raw_01 = read_raw_bids(bids_path)

    # find sidecar scans.tsv file and alter the
    # acquisition time to not have the optional microseconds
    scans_path = BIDSPath(subject=bids_path.subject,
                          session=bids_path.session,
                          root=tmp_path,
                          suffix='scans', extension='.tsv')
    scans_tsv = _from_tsv(scans_path)
    acq_time_str = scans_tsv['acq_time'][0]
    acq_time = datetime.strptime(acq_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    acq_time = acq_time.replace(tzinfo=timezone.utc)
    new_acq_time = acq_time_str.split('.')[0]
    assert acq_time == raw_01.info['meas_date']
    scans_tsv['acq_time'][0] = new_acq_time
    _to_tsv(scans_tsv, scans_path)

    # now re-load the data and it should be different
    # from the original date and the same as the newly altered date
    raw_02 = read_raw_bids(bids_path)
    new_acq_time += '.0Z'
    new_acq_time = datetime.strptime(new_acq_time,
                                     '%Y-%m-%dT%H:%M:%S.%fZ')
    new_acq_time = new_acq_time.replace(tzinfo=timezone.utc)
    assert raw_02.info['meas_date'] == new_acq_time
    assert new_acq_time != raw_01.info['meas_date']


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_info_reading(tmp_path):
    """Test reading information from a BIDS sidecar JSON file."""
    # read in USA dataset, so it should find 50 Hz
    raw = _read_raw_fif(raw_fname)

    # write copy of raw with line freq of 60
    # bids basename and fname
    bids_path = BIDSPath(subject='01', session='01',
                         task='audiovisual', run='01',
                         root=tmp_path)
    suffix = "meg"
    bids_fname = bids_path.copy().update(suffix=suffix,
                                         extension='.fif')
    write_raw_bids(raw, bids_path, overwrite=True)

    # find sidecar JSON fname
    bids_fname.update(datatype=suffix)
    sidecar_fname = _find_matching_sidecar(bids_fname, suffix=suffix,
                                           extension='.json')
    sidecar_fname = pathlib.Path(sidecar_fname)

    # assert that we get the same line frequency set
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info['line_freq'] == 60

    # setting line_freq to None should produce 'n/a' in the JSON sidecar
    raw.info['line_freq'] = None
    write_raw_bids(raw, bids_path, overwrite=True)
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info['line_freq'] is None

    sidecar_json = json.loads(sidecar_fname.read_text(encoding='utf-8'))
    assert sidecar_json["PowerLineFrequency"] == 'n/a'

    # 2. if line frequency is not set in raw file, then ValueError
    del raw.info['line_freq']
    with pytest.raises(ValueError, match="PowerLineFrequency .* required"):
        write_raw_bids(raw, bids_path, overwrite=True)

    # check whether there are "Extra points" in raw.info['dig'] if
    # DigitizedHeadPoints is set to True and not otherwise
    n_dig_points = 0
    for dig_point in raw.info['dig']:
        if dig_point['kind'] == FIFF.FIFFV_POINT_EXTRA:
            n_dig_points += 1
    if sidecar_json['DigitizedHeadPoints']:
        assert n_dig_points > 0
    else:
        assert n_dig_points == 0

    # check whether any of NAS/LPA/RPA are present in raw.info['dig']
    # DigitizedLandmark is set to True, and False otherwise
    landmark_present = False
    for dig_point in raw.info['dig']:
        if dig_point['kind'] in [FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_RPA,
                                 FIFF.FIFFV_POINT_NASION]:
            landmark_present = True
            break
    if landmark_present:
        assert sidecar_json['DigitizedLandmarks'] is True
    else:
        assert sidecar_json['DigitizedLandmarks'] is False

    # make a copy of the sidecar in "derivatives/"
    # to check that we make sure we always get the right sidecar
    # in addition, it should not break the sidecar reading
    # in `read_raw_bids`
    raw.info['line_freq'] = 60
    write_raw_bids(raw, bids_path, overwrite=True)
    deriv_dir = tmp_path / 'derivatives'
    deriv_dir.mkdir()
    sidecar_copy = deriv_dir / op.basename(sidecar_fname)
    sidecar_json = json.loads(sidecar_fname.read_text(encoding='utf-8'))
    sidecar_json["PowerLineFrequency"] = 45
    _write_json(sidecar_copy, sidecar_json)
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info['line_freq'] == 60

    # 3. assert that we get an error when sidecar json doesn't match
    _update_sidecar(sidecar_fname, "PowerLineFrequency", 55)
    with pytest.warns(RuntimeWarning, match="Defaulting to .* sidecar JSON"):
        raw = read_raw_bids(bids_path=bids_path)
        assert raw.info['line_freq'] == 55


@requires_version('mne', '0.24')
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
@pytest.mark.filterwarnings(warning_str['maxshield'])
def test_handle_chpi_reading(tmp_path):
    """Test reading of cHPI information."""
    raw = _read_raw_fif(raw_fname_chpi, allow_maxshield=True)
    root = tmp_path / 'chpi'
    root.mkdir()
    bids_path = BIDSPath(subject='01', session='01',
                         task='audiovisual', run='01',
                         root=root, datatype='meg')
    bids_path = write_raw_bids(raw, bids_path)

    raw_read = read_raw_bids(bids_path)
    assert raw_read.info['hpi_subsystem'] is not None

    # cause conflicts between cHPI info in sidecar and raw data
    meg_json_path = bids_path.copy().update(suffix='meg', extension='.json')
    with open(meg_json_path, 'r', encoding='utf-8') as f:
        meg_json_data = json.load(f)

    # cHPI frequency mismatch
    meg_json_data_freq_mismatch = meg_json_data.copy()
    meg_json_data_freq_mismatch['HeadCoilFrequency'][0] = 123
    _write_json(meg_json_path, meg_json_data_freq_mismatch, overwrite=True)

    with pytest.warns(RuntimeWarning, match='Defaulting to .* mne.Raw object'):
        raw_read = read_raw_bids(bids_path)

    # cHPI "off" according to sidecar, but present in the data
    meg_json_data_chpi_mismatch = meg_json_data.copy()
    meg_json_data_chpi_mismatch['ContinuousHeadLocalization'] = False
    _write_json(meg_json_path, meg_json_data_chpi_mismatch, overwrite=True)

    raw_read = read_raw_bids(bids_path)
    assert raw_read.info['hpi_subsystem'] is None
    assert raw_read.info['hpi_meas'] == []


@pytest.mark.filterwarnings(warning_str['nasion_not_found'])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_eeg_coords_reading(tmp_path):
    """Test reading iEEG coordinates from BIDS files."""
    bids_path = BIDSPath(
        subject=subject_id, session=session_id, run=run, acquisition=acq,
        task=task, root=tmp_path)

    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')
    raw = _read_raw_edf(raw_fname)

    # ensure we are writing 'eeg' data
    raw.set_channel_types({ch: 'eeg'
                           for ch in raw.ch_names})

    # set a `random` montage
    ch_names = raw.ch_names
    elec_locs = np.random.random((len(ch_names), 3)).astype(float)
    ch_pos = dict(zip(ch_names, elec_locs))

    # # create montage in 'unknown' coordinate frame
    # # and assert coordsystem/electrodes sidecar tsv don't exist
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                            coord_frame="unknown")
    raw.set_montage(montage)
    with pytest.warns(RuntimeWarning, match="Skipping EEG electrodes.tsv"):
        write_raw_bids(raw, bids_path, overwrite=True)

    bids_path.update(root=tmp_path)
    coordsystem_fname = _find_matching_sidecar(bids_path,
                                               suffix='coordsystem',
                                               extension='.json',
                                               on_error='warn')
    electrodes_fname = _find_matching_sidecar(bids_path,
                                              suffix='electrodes',
                                              extension='.tsv',
                                              on_error='warn')
    assert coordsystem_fname is None
    assert electrodes_fname is None

    # create montage in head frame and set should result in
    # warning if landmarks not set
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                            coord_frame="head")
    raw.set_montage(montage)
    with pytest.warns(RuntimeWarning, match='Setting montage not possible '
                                            'if anatomical landmarks'):
        write_raw_bids(raw, bids_path, overwrite=True)

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                            coord_frame="head",
                                            nasion=[1, 0, 0],
                                            lpa=[0, 1, 0],
                                            rpa=[0, 0, 1])
    raw.set_montage(montage)
    write_raw_bids(raw, bids_path, overwrite=True)

    # obtain the sensor positions and assert ch_coords are same
    raw_test = read_raw_bids(bids_path, verbose=True)
    assert not object_diff(raw.info['chs'], raw_test.info['chs'])

    # modify coordinate frame to not-captrak
    coordsystem_fname = _find_matching_sidecar(bids_path,
                                               suffix='coordsystem',
                                               extension='.json')
    _update_sidecar(coordsystem_fname, 'EEGCoordinateSystem', 'besa')
    with pytest.warns(RuntimeWarning, match='EEG Coordinate frame is not '
                                            'accepted BIDS keyword'):
        raw_test = read_raw_bids(bids_path)
        assert raw_test.info['dig'] is None


@pytest.mark.parametrize('bids_path',
                         [_bids_path, _bids_path_minimal])
@pytest.mark.filterwarnings(warning_str['nasion_not_found'])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_ieeg_coords_reading(bids_path, tmp_path):
    """Test reading iEEG coordinates from BIDS files."""
    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')
    bids_fname = bids_path.copy().update(datatype='ieeg',
                                         suffix='ieeg',
                                         extension='.edf',
                                         root=tmp_path)
    raw = _read_raw_edf(raw_fname)

    # ensure we are writing 'ecog'/'ieeg' data
    raw.set_channel_types({ch: 'ecog'
                           for ch in raw.ch_names})

    # coordinate frames in mne-python should all map correctly
    # set a `random` montage
    ch_names = raw.ch_names
    elec_locs = np.random.random((len(ch_names), 3)).astype(float)
    ch_pos = dict(zip(ch_names, elec_locs))
    coordinate_frames = ['mni_tal']
    for coord_frame in coordinate_frames:
        # XXX: mne-bids doesn't support multiple electrodes.tsv files
        sh.rmtree(tmp_path)
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                                coord_frame=coord_frame)
        raw.set_montage(montage)
        write_raw_bids(raw, bids_fname,
                       overwrite=True, verbose=False)
        # read in raw file w/ updated coordinate frame
        # and make sure all digpoints are correct coordinate frames
        raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
        coord_frame_int = MNE_STR_TO_FRAME[coord_frame]
        for digpoint in raw_test.info['dig']:
            assert digpoint['coord_frame'] == coord_frame_int

    # start w/ new bids root
    sh.rmtree(tmp_path)
    write_raw_bids(raw, bids_fname, overwrite=True, verbose=False)

    # obtain the sensor positions and assert ch_coords are same
    raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
    orig_locs = raw.info['dig'][1]
    test_locs = raw_test.info['dig'][1]
    assert orig_locs == test_locs
    assert not object_diff(raw.info['chs'], raw_test.info['chs'])

    # read in the data and assert montage is the same
    # regardless of 'm', 'cm', 'mm', or 'pixel'
    scalings = {'m': 1, 'cm': 100, 'mm': 1000}
    bids_fname.update(root=tmp_path)
    coordsystem_fname = _find_matching_sidecar(bids_fname,
                                               suffix='coordsystem',
                                               extension='.json')
    electrodes_fname = _find_matching_sidecar(bids_fname,
                                              suffix='electrodes',
                                              extension='.tsv')
    orig_electrodes_dict = _from_tsv(electrodes_fname,
                                     [str, float, float, float, str])

    # not BIDS specified should not be read
    coord_unit = 'km'
    scaling = 0.001
    _update_sidecar(coordsystem_fname, 'iEEGCoordinateUnits', coord_unit)
    electrodes_dict = _from_tsv(electrodes_fname,
                                [str, float, float, float, str])
    for axis in ['x', 'y', 'z']:
        electrodes_dict[axis] = \
            np.multiply(orig_electrodes_dict[axis], scaling)
    _to_tsv(electrodes_dict, electrodes_fname)
    with pytest.warns(RuntimeWarning, match='Coordinate unit is not '
                                            'an accepted BIDS unit'):
        raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)

    # correct BIDS units should scale to meters properly
    for coord_unit, scaling in scalings.items():
        # update coordinate SI units
        _update_sidecar(coordsystem_fname, 'iEEGCoordinateUnits', coord_unit)
        electrodes_dict = _from_tsv(electrodes_fname,
                                    [str, float, float, float, str])
        for axis in ['x', 'y', 'z']:
            electrodes_dict[axis] = \
                np.multiply(orig_electrodes_dict[axis], scaling)
        _to_tsv(electrodes_dict, electrodes_fname)

        # read in raw file w/ updated montage
        raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)

        # obtain the sensor positions and make sure they're the same
        assert_dig_allclose(raw.info, raw_test.info)

    # XXX: Improve by changing names to 'unknown' coordframe (needs mne PR)
    # check that coordinate systems other coordinate systems should be named
    # in the file and not the CoordinateSystem, which is reserved for keywords
    coordinate_frames = ['Other']
    for coord_frame in coordinate_frames:
        # update coordinate units
        _update_sidecar(coordsystem_fname, 'iEEGCoordinateSystem', coord_frame)
        # read in raw file w/ updated coordinate frame
        # and make sure all digpoints are MRI coordinate frame
        with pytest.warns(RuntimeWarning, match="Defaulting coordinate "
                                                "frame to unknown"):
            raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
            assert raw_test.info['dig'] is not None

    # check that standard template identifiers that are unsupported in
    # mne-python coordinate frames, still get read in, but produce a warning
    coordinate_frames = ['individual', 'fsnative', 'scanner',
                         'ICBM452AirSpace', 'NIHPD']
    for coord_frame in coordinate_frames:
        # update coordinate units
        _update_sidecar(coordsystem_fname, 'iEEGCoordinateSystem', coord_frame)
        # read in raw file w/ updated coordinate frame
        # and make sure all digpoints are MRI coordinate frame
        with pytest.warns(
                RuntimeWarning, match=f"iEEG Coordinate frame {coord_frame} "
                                      f"is not a readable BIDS keyword "):
            raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
            assert raw_test.info['dig'] is not None

    # ACPC should be read in as RAS for iEEG
    _update_sidecar(coordsystem_fname, 'iEEGCoordinateSystem', 'ACPC')
    raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
    coord_frame_int = MNE_STR_TO_FRAME['mri']
    for digpoint in raw_test.info['dig']:
        assert digpoint['coord_frame'] == coord_frame_int

    # if we delete the coordsystem.json file, an error will be raised
    os.remove(coordsystem_fname)
    with pytest.raises(RuntimeError, match='BIDS mandates that '
                                           'the coordsystem.json'):
        raw = read_raw_bids(bids_path=bids_fname, verbose=False)

    # test error message if electrodes is not a subset of Raw
    bids_path.update(root=tmp_path)
    write_raw_bids(raw, bids_path, overwrite=True)
    electrodes_dict = _from_tsv(electrodes_fname)
    # pop off 5 channels
    for key in electrodes_dict.keys():
        for i in range(5):
            electrodes_dict[key].pop()
    _to_tsv(electrodes_dict, electrodes_fname)
    # popping off channels should not result in an error
    # however, a warning will be raised through mne-python
    with pytest.warns(RuntimeWarning, match='DigMontage is '
                                            'only a subset of info'):
        read_raw_bids(bids_path=bids_fname, verbose=False)

    # make sure montage is set if there are coordinates w/ 'n/a'
    raw.info['bads'] = []
    write_raw_bids(raw, bids_path,
                   overwrite=True, verbose=False)
    electrodes_dict = _from_tsv(electrodes_fname)
    for axis in ['x', 'y', 'z']:
        electrodes_dict[axis][0] = 'n/a'
        electrodes_dict[axis][3] = 'n/a'
    _to_tsv(electrodes_dict, electrodes_fname)

    # test if montage is correctly set via mne-bids
    # electrode coordinates should be nan
    # when coordinate is 'n/a'
    nan_chs = [electrodes_dict['name'][i] for i in [0, 3]]
    with pytest.warns(RuntimeWarning, match='There are channels '
                                            'without locations'):
        raw = read_raw_bids(bids_path=bids_fname, verbose=False)
        for idx, ch in enumerate(raw.info['chs']):
            if ch['ch_name'] in nan_chs:
                assert all(np.isnan(ch['loc'][:3]))
            else:
                assert not any(np.isnan(ch['loc'][:3]))
            assert ch['ch_name'] not in raw.info['bads']


@requires_nibabel()
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
@pytest.mark.parametrize('fname', ['testdata_ctf.ds', 'catch-alp-good-f.ds'])
def test_get_head_mri_trans_ctf(fname, tmp_path):
    """Test getting a trans object from BIDS data in CTF."""
    import nibabel as nib

    ctf_data_path = op.join(testing.data_path(), 'CTF')
    raw_ctf_fname = op.join(ctf_data_path, fname)
    raw_ctf = _read_raw_ctf(raw_ctf_fname, clean_names=True)
    bids_path = _bids_path.copy().update(root=tmp_path)
    write_raw_bids(raw_ctf, bids_path, overwrite=False)

    # Take a fake trans
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    t1w_mgh = nib.load(t1w_mgh)

    t1w_bids_path = BIDSPath(subject=subject_id, session=session_id,
                             acquisition=acq, root=tmp_path)
    landmarks = get_anat_landmarks(
        t1w_mgh, raw_ctf.info, trans, fs_subject='sample',
        fs_subjects_dir=op.join(data_path, 'subjects'))
    write_anat(t1w_mgh, bids_path=t1w_bids_path, landmarks=landmarks)

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(
        bids_path=bids_path, extra_params=dict(clean_names=True),
        fs_subject='sample', fs_subjects_dir=op.join(data_path, 'subjects'))

    assert_almost_equal(trans['trans'], estimated_trans['trans'])


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_raw_bids_pathlike(tmp_path):
    """Test that read_raw_bids() can handle a Path-like bids_root."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg')
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    raw = read_raw_bids(bids_path=bids_path)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_raw_datatype(tmp_path):
    """Test that read_raw_bids() can infer the str_suffix if need be."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg')
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    raw_1 = read_raw_bids(bids_path=bids_path)
    bids_path.update(datatype=None)
    raw_2 = read_raw_bids(bids_path=bids_path)
    raw_3 = read_raw_bids(bids_path=bids_path)

    raw_1.crop(0, 2).load_data()
    raw_2.crop(0, 2).load_data()
    raw_3.crop(0, 2).load_data()

    assert raw_1 == raw_2
    assert raw_1 == raw_3


def test_handle_channel_type_casing(tmp_path):
    """Test that non-uppercase entries in the `type` column are accepted."""
    bids_path = _bids_path.copy().update(root=tmp_path)
    raw = _read_raw_fif(raw_fname, verbose=False)

    write_raw_bids(raw, bids_path, overwrite=True,
                   verbose=False)

    ch_path = bids_path.copy().update(root=tmp_path,
                                      datatype='meg',
                                      suffix='channels',
                                      extension='.tsv')
    bids_channels_fname = ch_path.fpath

    # Convert all channel type entries to lowercase.
    channels_data = _from_tsv(bids_channels_fname)
    channels_data['type'] = [t.lower() for t in channels_data['type']]
    _to_tsv(channels_data, bids_channels_fname)

    with pytest.warns(RuntimeWarning, match='lowercase spelling'):
        read_raw_bids(bids_path)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_bads_reading(tmp_path):
    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg')
    bads_raw = ['MEG 0112', 'MEG 0113']
    bads_sidecar = ['EEG 053', 'MEG 2443']

    # Produce conflicting information between raw and sidecar file.
    raw = _read_raw_fif(raw_fname, verbose=False)
    raw.info['bads'] = bads_sidecar
    write_raw_bids(raw, bids_path, verbose=False)

    raw = _read_raw(bids_path.copy().update(extension='.fif').fpath,
                    preload=True)
    raw.info['bads'] = bads_raw
    raw.save(raw.filenames[0], overwrite=True)

    # Upon reading the data, only the sidecar info should be present.
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    assert len(raw.info['bads']) == len(bads_sidecar)
    assert set(raw.info['bads']) == set(bads_sidecar)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_write_read_fif_split_file(tmp_path):
    """Test split files are read correctly."""
    # load raw test file, extend it to be larger than 2gb, and save it
    bids_root = tmp_path / 'bids'
    tmp_dir = tmp_path / 'tmp'
    tmp_dir.mkdir()

    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    raw = _read_raw_fif(raw_fname, verbose=False)
    bids_path.update(acquisition=None)
    write_raw_bids(raw, bids_path, verbose=False)
    bids_path.update(acquisition='01')
    n_channels = len(raw.ch_names)
    n_times = int(2.2e9 / (n_channels * 4))  # enough to produce a split
    data = np.empty((n_channels, n_times), dtype=np.float32)
    raw = mne.io.RawArray(data, raw.info)
    big_fif_fname = pathlib.Path(tmp_dir) / 'test_raw.fif'
    raw.save(big_fif_fname)
    raw = _read_raw_fif(big_fif_fname, verbose=False)
    write_raw_bids(raw, bids_path, verbose=False)

    # test whether split raw files were read correctly
    raw1 = read_raw_bids(bids_path=bids_path)
    assert 'split-01' in str(bids_path.fpath)
    bids_path.update(split='01')
    raw2 = read_raw_bids(bids_path=bids_path)
    bids_path.update(split='02')
    raw3 = read_raw_bids(bids_path=bids_path)
    assert len(raw) == len(raw1)
    assert len(raw) == len(raw2)
    assert len(raw) > len(raw3)

    # check that split files both appear in scans.tsv
    scans_tsv = BIDSPath(
        subject=subject_id, session=session_id,
        suffix='scans', extension='.tsv',
        root=bids_root)
    scan_data = _from_tsv(scans_tsv)
    scan_fnames = scan_data['filename']
    scan_acqtime = scan_data['acq_time']

    assert len(scan_fnames) == 3
    assert 'split-01' in scan_fnames[0] and 'split-02' in scan_fnames[1]
    # check that the acq_times in scans.tsv are the same
    assert scan_acqtime[0] == scan_acqtime[1]
    # check the recordings are in the correct order
    assert raw2.first_time < raw3.first_time

    # check whether non-matching acq_times are caught
    scan_data['acq_time'][0] = scan_acqtime[0].split('.')[0]
    _to_tsv(scan_data, scans_tsv)
    with pytest.raises(ValueError,
                       match='Split files must have the same acq_time.'):
        read_raw_bids(bids_path)

    # reset scans.tsv file for downstream tests
    scan_data['acq_time'][0] = scan_data['acq_time'][1]
    _to_tsv(scan_data, scans_tsv)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_ignore_exclude_param(tmp_path):
    """Test that extra_params=dict(exclude=...) is being ignored."""
    bids_path = _bids_path.copy().update(root=tmp_path)
    ch_name = 'EEG 001'
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

    raw = read_raw_bids(bids_path=bids_path, verbose=False,
                        extra_params=dict(exclude=[ch_name]))
    assert ch_name in raw.ch_names


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_channels_tsv_raw_mismatch(tmp_path):
    """Test behavior when channels.tsv contains channels not found in raw."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype='meg',
                                         task='rest')

    # Remove one channel from the raw data without updating channels.tsv
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

    raw_path = bids_path.copy().update(extension='.fif').fpath
    raw = _read_raw(raw_path, preload=True)
    raw.drop_channels(ch_names=raw.ch_names[-1])
    raw.load_data()
    raw.save(raw_path, overwrite=True)

    with pytest.warns(
        RuntimeWarning,
        match='number of channels in the channels.tsv sidecar .* '
              'does not match the number of channels in the raw data'
    ):
        read_raw_bids(bids_path)

    # Remame a channel in the raw data without updating channels.tsv
    # (number of channels in channels.tsv and raw remains different)
    ch_name_orig = raw.ch_names[-1]
    ch_name_new = 'MEGtest'
    raw.rename_channels({ch_name_orig: ch_name_new})
    raw.save(raw_path, overwrite=True)

    with pytest.warns(
        RuntimeWarning,
        match=f'Cannot set channel type for the following channels, as they '
              f'are missing in the raw data: {ch_name_orig}'
    ):
        read_raw_bids(bids_path)

    # Mark channel as bad in channels.tsv and remove it from the raw data
    raw = _read_raw_fif(raw_fname, verbose=False)
    ch_name_orig = raw.ch_names[-1]
    ch_name_new = 'MEGtest'

    raw.info['bads'] = [ch_name_orig]
    write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

    raw.drop_channels(raw.ch_names[-2])
    raw.rename_channels({ch_name_orig: ch_name_new})
    raw.save(raw_path, overwrite=True)

    with pytest.warns(
        RuntimeWarning,
        match=f'Cannot set "bad" status for the following channels, as '
              f'they are missing in the raw data: {ch_name_orig}'
    ):
        read_raw_bids(bids_path)


def test_file_not_found(tmp_path):
    """Check behavior if the requested file cannot be found."""
    # First a path with a filename extension.
    bp = BIDSPath(
        root=tmp_path, subject='foo', task='bar', datatype='eeg', suffix='eeg',
        extension='.fif'
    )
    bp.fpath.parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match='File does not exist'):
        read_raw_bids(bids_path=bp)

    # Now without an extension
    bp.extension = None
    with pytest.raises(FileNotFoundError, match='File does not exist'):
        read_raw_bids(bids_path=bp)
