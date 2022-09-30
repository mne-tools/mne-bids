"""Test for the MNE BIDS updating of BIDS datasets."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD-3-Clause
import json
import os.path as op
from pathlib import Path

import pytest
import numpy as np

import mne
from mne.io.constants import FIFF
from mne.datasets import testing
from mne.utils import requires_nibabel

from mne_bids import (BIDSPath, write_raw_bids,
                      write_meg_calibration, write_meg_crosstalk,
                      get_anat_landmarks, update_sidecar_json, write_anat,
                      update_anat_landmarks)
from mne_bids.path import _mkdir_p
from mne_bids.utils import _write_json

subject_id = '01'
session_id = '01'
run = '01'
acq = None
task = 'testing'

data_path = testing.data_path(download=False)
bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)


@pytest.fixture(scope='session')
def _get_bids_test_dir(tmp_path_factory):
    """Return path to a written test BIDS dir."""
    bids_root = str(tmp_path_factory.mktemp('mnebids_utils_test_bids_ds'))
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')
    cal_fname = op.join(data_path, 'SSS', 'sss_cal_mgh.dat')
    crosstalk_fname = op.join(data_path, 'SSS', 'ct_sparse.fif')

    raw = mne.io.read_raw_fif(raw_fname)
    raw.info['line_freq'] = 60

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    bids_path.update(root=bids_root)
    # Write multiple runs for test_purposes
    for run_idx in [run, '02']:
        name = bids_path.copy().update(run=run_idx)
        write_raw_bids(raw, name, events=events,
                       event_id=event_id, overwrite=True)

    write_meg_calibration(cal_fname, bids_path=bids_path)
    write_meg_crosstalk(crosstalk_fname, bids_path=bids_path)
    return bids_root


@pytest.fixture(scope='session')
def _get_sidecar_json_update_file(_get_bids_test_dir):
    """Return path to a sidecar JSON updating file."""
    bids_root = _get_bids_test_dir
    sample_scripts = op.join(bids_root, 'sourcedata')
    sidecar_fpath = op.join(sample_scripts, 'sidecarjson_update.json')
    _mkdir_p(sample_scripts)

    update_json = {
        'InstitutionName': 'mne-bids',
        'InstitutionAddress': 'Internet',
        'MEGChannelCount': 300,
        'MEGREFChannelCount': 6,
        'SEEGChannelCount': 0,
    }
    _write_json(sidecar_fpath, update_json, overwrite=True)

    return sidecar_fpath


@testing.requires_testing_data
def test_update_sidecar_jsons(_get_bids_test_dir, _bids_validate,
                              _get_sidecar_json_update_file):
    """Test updating sidecar JSON files."""
    bids_path = BIDSPath(
        subject=subject_id, session=session_id, run=run, acquisition=acq,
        task=task, suffix='meg', root=_get_bids_test_dir)

    # expected key, original value, and expected value after update
    # Fields that are not `None` already are expected to exist
    # in this sidecar file. Fields that are `None` will get
    # written with the sidecar json value when update is called.
    expected_checks = [('InstitutionName', None, 'mne-bids'),
                       ('InstitutionAddress', None, 'Internet'),
                       ('MEGChannelCount', 306, 300),
                       ('MEGREFChannelCount', 0, 6),
                       ('ECGChannelCount', 0, 0),
                       ('SEEGChannelCount', None, 0)]

    # get the sidecar json
    sidecar_path = bids_path.copy().update(extension='.json', datatype='meg')
    sidecar_fpath = sidecar_path.fpath
    with open(sidecar_fpath, 'r', encoding='utf-8') as fin:
        sidecar_json = json.load(fin)
    for key, val, _ in expected_checks:
        assert sidecar_json.get(key) == val
    _bids_validate(bids_path.root)

    # update sidecars
    update_sidecar_json(sidecar_path, _get_sidecar_json_update_file)
    with open(sidecar_fpath, 'r', encoding='utf-8') as fin:
        sidecar_json = json.load(fin)
    for key, _, val in expected_checks:
        assert sidecar_json.get(key) == val
    _bids_validate(bids_path.root)

    # should result in error if you don't explicitly say
    # its a json file
    with pytest.raises(RuntimeError, match='Only works for ".json"'):
        update_sidecar_json(sidecar_path.copy().update(
            extension=None), _get_sidecar_json_update_file)

    # error should raise if the file path doesn't exist
    error_bids_path = sidecar_path.copy().update(subject='02')
    with pytest.raises(RuntimeError, match='Sidecar file '
                                           'does not exist.'):
        update_sidecar_json(
            error_bids_path, _get_sidecar_json_update_file)


@requires_nibabel()
@testing.requires_testing_data
def test_update_anat_landmarks(tmp_path):
    """Test updating the anatomical landmarks of an MRI scan."""
    raw_path = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    trans_path = Path(str(raw_path).replace('_raw.fif', '-trans.fif'))
    t1_path = data_path / 'subjects' / 'sample' / 'mri' / 'T1.mgz'
    fs_subject = 'sample'
    fs_subjects_dir = data_path / 'subjects'
    bids_root = tmp_path
    bids_path_mri = BIDSPath(subject=subject_id, session=session_id,
                             acquisition=acq, root=bids_root, datatype='anat',
                             suffix='T1w')

    # First, write the MRI scan to BIDS, including the anatomical landmarks
    info = mne.io.read_info(raw_path)
    trans = mne.read_trans(trans_path)
    landmarks = get_anat_landmarks(
        image=t1_path, info=info, trans=trans, fs_subject=fs_subject,
        fs_subjects_dir=fs_subjects_dir
    )
    bids_path_mri = write_anat(image=t1_path, bids_path=bids_path_mri,
                               landmarks=landmarks, deface=False)
    bids_path_mri_json = bids_path_mri.copy().update(extension='.json')

    # Modify the landmarks
    # Move the nasion a bit
    landmarks_new = landmarks.copy()
    landmarks_new.dig[1]['r'] *= 0.9
    update_anat_landmarks(bids_path=bids_path_mri, landmarks=landmarks_new)

    with bids_path_mri_json.fpath.open(encoding='utf-8') as f:
        mri_json = json.load(f)

    assert np.allclose(
        landmarks_new.dig[1]['r'],
        mri_json['AnatomicalLandmarkCoordinates']['NAS']
    )

    # Remove JSON sidecar; updating the anatomical landmarks should re-create
    # the file unless `on_missing` is `'raise'`
    bids_path_mri_json.fpath.unlink()
    with pytest.raises(
        KeyError,
        match='No AnatomicalLandmarkCoordinates section found'
    ):
        update_anat_landmarks(bids_path=bids_path_mri, landmarks=landmarks_new)

    update_anat_landmarks(
        bids_path=bids_path_mri, landmarks=landmarks_new, on_missing='ignore'
    )

    with pytest.raises(KeyError, match='landmark not found'):
        update_anat_landmarks(
            bids_path=bids_path_mri, landmarks=landmarks_new, kind='ses-1'
        )
    update_anat_landmarks(
        bids_path=bids_path_mri, landmarks=landmarks_new, kind='ses-1',
        on_missing='ignore'
    )

    mri_json = json.loads(bids_path_mri_json.fpath.read_text(encoding='utf-8'))
    assert 'NAS' in mri_json['AnatomicalLandmarkCoordinates']
    assert 'NAS_ses-1' in mri_json['AnatomicalLandmarkCoordinates']

    assert np.allclose(
        landmarks_new.dig[1]['r'],
        mri_json['AnatomicalLandmarkCoordinates']['NAS']
    )

    # Check without extension provided
    bids_path_mri_no_ext = bids_path_mri.copy().update(extension=None)
    update_anat_landmarks(bids_path=bids_path_mri_no_ext,
                          landmarks=landmarks_new)

    # Check handling of invalid input
    bids_path_invalid = bids_path_mri.copy().update(datatype='meg')
    with pytest.raises(ValueError, match='Can only operate on "anat"'):
        update_anat_landmarks(bids_path=bids_path_invalid, landmarks=landmarks)

    bids_path_invalid = bids_path_mri.copy().update(suffix=None)
    with pytest.raises(ValueError, match='lease specify the "suffix"'):
        update_anat_landmarks(bids_path=bids_path_invalid, landmarks=landmarks)

    bids_path_invalid = bids_path_mri.copy().update(suffix='meg')
    with pytest.raises(ValueError,
                       match='Can only operate on "T1w" and "FLASH"'):
        update_anat_landmarks(bids_path=bids_path_invalid, landmarks=landmarks)

    bids_path_invalid = bids_path_mri.copy().update(subject='invalid')
    with pytest.raises(ValueError, match='Could not find an MRI scan'):
        update_anat_landmarks(bids_path=bids_path_invalid, landmarks=landmarks)

    # Unsupported coordinate frame
    landmarks_invalid = landmarks.copy()
    for digpoint in landmarks_invalid.dig:
        digpoint['coord_frame'] = FIFF.FIFFV_MNE_COORD_RAS

    with pytest.raises(ValueError, match='must be specified in MRI voxel'):
        update_anat_landmarks(bids_path=bids_path_mri,
                              landmarks=landmarks_invalid)

    # Missing cardinal point
    landmarks_invalid = landmarks.copy()
    del landmarks_invalid.dig[0]
    with pytest.raises(ValueError,
                       match='did not contain all required cardinal points'):
        update_anat_landmarks(bids_path=bids_path_mri,
                              landmarks=landmarks_invalid)

    # Test with path-like landmarks
    fiducials_path = (data_path / 'subjects' / 'sample' / 'bem' /
                      'sample-fiducials.fif')

    update_anat_landmarks(
        bids_path=bids_path_mri,
        landmarks=fiducials_path,
        fs_subject='sample',
        fs_subjects_dir=data_path / 'subjects'
    )
    expected_coords_in_voxels = np.array(
        [[68.38202,  45.24057,  43.439808],  # noqa: E241
         [42.27006,  30.758774, 74.09837 ],  # noqa: E202, E241
         [17.044853, 46.586075, 42.618504]]
    )
    mri_json = json.loads(
        bids_path_mri_json.fpath.read_text(encoding='utf-8')
    )
    for landmark, expected_coords in zip(
        ('LPA', 'NAS', 'RPA'),
        expected_coords_in_voxels
    ):
        assert np.allclose(
            mri_json['AnatomicalLandmarkCoordinates'][landmark],
            expected_coords
        )
