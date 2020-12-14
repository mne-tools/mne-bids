"""Testing utilities for file io."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import json
import os
import os.path as op
import pathlib

import pytest
import shutil as sh

import numpy as np
from numpy.testing import assert_almost_equal

# This is here to handle mne-python <0.20
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne

from mne.utils import _TempDir, requires_nibabel, object_diff
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
from mne_bids.write import write_anat, write_raw_bids

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

warning_str = dict(
    channel_unit_changed='ignore:The unit for chann*.:RuntimeWarning:mne',
    meas_date_set_to_none="ignore:.*'meas_date' set to None:RuntimeWarning:"
                          "mne",
    nasion_not_found='ignore:.*nasion not found:RuntimeWarning:mne',
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


def test_not_implemented():
    """Test the not yet implemented data formats raise an adequate error."""
    for not_implemented_ext in ['.mef', '.nwb']:
        data_path = _TempDir()
        raw_fname = op.join(data_path, 'test' + not_implemented_ext)
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
def test_read_participants_data():
    """Test reading information from a BIDS sidecar.json file."""
    bids_root = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
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

    # if modifying participants tsv, then read_raw_bids reflects that
    participants_tsv_fpath = op.join(bids_root, 'participants.tsv')
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
    with pytest.warns(RuntimeWarning, match='Participants file not found'):
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
                                                      sex_bids, sex_mne):
    """Test we're correctly mapping handedness and sex between BIDS and MNE."""
    bids_root = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    participants_tsv_fpath = op.join(bids_root, 'participants.tsv')
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
def test_get_head_mri_trans():
    """Test getting a trans object from BIDS data."""
    import nibabel as nib

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    # Write it to BIDS
    raw = _read_raw_fif(raw_fname)
    bids_root = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root)
    write_raw_bids(raw, bids_path, events_data=events, event_id=event_id,
                   overwrite=False)

    # We cannot recover trans, if no MRI has yet been written
    with pytest.raises(RuntimeError):
        estimated_trans = get_head_mri_trans(bids_path=bids_path)

    # Write some MRI data and supply a `trans` so that a sidecar gets written
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    t1w_mgh = nib.load(t1w_mgh)

    t1w_bidspath = BIDSPath(subject=subject_id, session=session_id,
                            acquisition=acq, root=bids_root)
    t1w_bidspath = write_anat(t1w_mgh, bids_path=t1w_bidspath,
                              raw=raw, trans=trans, verbose=True)
    anat_dir = t1w_bidspath.directory

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(bids_path=bids_path)

    assert trans['from'] == estimated_trans['from']
    assert trans['to'] == estimated_trans['to']
    assert_almost_equal(trans['trans'], estimated_trans['trans'])
    print(trans)
    print(estimated_trans)

    # provoke an error by introducing NaNs into MEG coords
    with pytest.raises(RuntimeError, match='AnatomicalLandmarkCoordinates'):
        raw.info['dig'][0]['r'] = np.ones(3) * np.nan
        sh.rmtree(anat_dir)
        bids_path = write_anat(t1w_mgh, bids_path=t1w_bidspath,
                               raw=raw, trans=trans, verbose=True)
        estimated_trans = get_head_mri_trans(bids_path=bids_path)


def test_handle_events_reading():
    """Test reading events from a BIDS events.tsv file."""
    # We can use any `raw` for this
    raw = _read_raw_fif(raw_fname)

    # Create an arbitrary events.tsv file, to test we can deal with 'n/a'
    # make sure we can deal w/ "#" characters
    events = {'onset': [11, 12, 'n/a'],
              'duration': ['n/a', 'n/a', 'n/a'],
              'trial_type': ["rec start", "trial #1", "trial #2!"]
              }
    tmp_dir = _TempDir()
    events_fname = op.join(tmp_dir, 'sub-01_task-test_events.json')
    _to_tsv(events, events_fname)

    raw = _handle_events_reading(events_fname, raw)
    events, event_id = mne.events_from_annotations(raw)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_info_reading():
    """Test reading information from a BIDS sidecar.json file."""
    bids_root = _TempDir()

    # read in USA dataset, so it should find 50 Hz
    raw = _read_raw_fif(raw_fname)

    # write copy of raw with line freq of 60
    # bids basename and fname
    bids_path = BIDSPath(subject='01', session='01',
                         task='audiovisual', run='01',
                         root=bids_root)
    suffix = "meg"
    bids_fname = bids_path.copy().update(suffix=suffix,
                                         extension='.fif')
    write_raw_bids(raw, bids_path, overwrite=True)

    # find sidecar JSON fname
    bids_fname.update(datatype=suffix)
    sidecar_fname = _find_matching_sidecar(bids_fname, suffix=suffix,
                                           extension='.json')

    # assert that we get the same line frequency set
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info['line_freq'] == 60

    # 2. if line frequency is not set in raw file, then ValueError
    raw.info['line_freq'] = None
    with pytest.raises(ValueError, match="PowerLineFrequency .* required"):
        write_raw_bids(raw, bids_path, overwrite=True)

    # make a copy of the sidecar in "derivatives/"
    # to check that we make sure we always get the right sidecar
    # in addition, it should not break the sidecar reading
    # in `read_raw_bids`
    deriv_dir = op.join(bids_root, "derivatives")
    sidecar_copy = op.join(deriv_dir, op.basename(sidecar_fname))
    os.mkdir(deriv_dir)
    with open(sidecar_fname, "r", encoding='utf-8') as fin:
        sidecar_json = json.load(fin)
        sidecar_json["PowerLineFrequency"] = 45
    _write_json(sidecar_copy, sidecar_json)
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info['line_freq'] == 60

    # 3. assert that we get an error when sidecar json doesn't match
    _update_sidecar(sidecar_fname, "PowerLineFrequency", 55)
    with pytest.raises(ValueError, match="Line frequency in sidecar json"):
        raw = read_raw_bids(bids_path=bids_path)
        assert raw.info['line_freq'] == 55


@pytest.mark.filterwarnings(warning_str['nasion_not_found'])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_eeg_coords_reading():
    """Test reading iEEG coordinates from BIDS files."""
    bids_root = _TempDir()

    bids_path = BIDSPath(
        subject=subject_id, session=session_id, run=run, acquisition=acq,
        task=task, root=bids_root)

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

    bids_path.update(root=bids_root)
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
def test_handle_ieeg_coords_reading(bids_path):
    """Test reading iEEG coordinates from BIDS files."""
    bids_root = _TempDir()

    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')
    bids_fname = bids_path.copy().update(datatype='ieeg',
                                         suffix='ieeg',
                                         extension='.edf',
                                         root=bids_root)
    raw = _read_raw_edf(raw_fname)

    # ensure we are writing 'ecog'/'ieeg' data
    raw.set_channel_types({ch: 'ecog'
                           for ch in raw.ch_names})

    # coordinate frames in mne-python should all map correctly
    # set a `random` montage
    ch_names = raw.ch_names
    elec_locs = np.random.random((len(ch_names), 3)).astype(float)
    ch_pos = dict(zip(ch_names, elec_locs))
    coordinate_frames = ['mri', 'ras']
    for coord_frame in coordinate_frames:
        # XXX: mne-bids doesn't support multiple electrodes.tsv files
        sh.rmtree(bids_root)
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
    sh.rmtree(bids_root)
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
    bids_fname.update(root=bids_root)
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
    coordinate_frames = ['lia', 'ria', 'lip', 'rip', 'las']
    for coord_frame in coordinate_frames:
        # update coordinate units
        _update_sidecar(coordsystem_fname, 'iEEGCoordinateSystem', coord_frame)
        # read in raw file w/ updated coordinate frame
        # and make sure all digpoints are MRI coordinate frame
        with pytest.warns(RuntimeWarning, match="iEEG Coordinate frame is "
                                                "not accepted BIDS keyword"):
            raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
            assert raw_test.info['dig'] is None

    # ACPC should be read in as RAS for iEEG
    _update_sidecar(coordsystem_fname, 'iEEGCoordinateSystem', 'acpc')
    raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
    coord_frame_int = MNE_STR_TO_FRAME['ras']
    for digpoint in raw_test.info['dig']:
        assert digpoint['coord_frame'] == coord_frame_int

    # if we delete the coordsystem.json file, an error will be raised
    os.remove(coordsystem_fname)
    with pytest.raises(RuntimeError, match='BIDS mandates that '
                                           'the coordsystem.json'):
        raw = read_raw_bids(bids_path=bids_fname, verbose=False)

    # test error message if electrodes don't match
    bids_path.update(root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True)
    electrodes_dict = _from_tsv(electrodes_fname)
    # pop off 5 channels
    for key in electrodes_dict.keys():
        for i in range(5):
            electrodes_dict[key].pop()
    _to_tsv(electrodes_dict, electrodes_fname)
    with pytest.raises(RuntimeError, match='Channels do not correspond'):
        raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)

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
def test_get_head_mri_trans_ctf(fname):
    """Test getting a trans object from BIDS data in CTF."""
    import nibabel as nib

    ctf_data_path = op.join(testing.data_path(), 'CTF')
    raw_ctf_fname = op.join(ctf_data_path, fname)
    raw_ctf = _read_raw_ctf(raw_ctf_fname, clean_names=True)
    bids_root = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root)
    write_raw_bids(raw_ctf, bids_path, overwrite=False)

    # Take a fake trans
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    t1w_mgh = nib.load(t1w_mgh)

    t1w_bids_path = BIDSPath(subject=subject_id, session=session_id,
                             acquisition=acq, root=bids_root)
    write_anat(t1w_mgh, bids_path=t1w_bids_path,
               raw=raw_ctf, trans=trans)

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(bids_path=bids_path,
                                         extra_params=dict(clean_names=True))

    assert_almost_equal(trans['trans'], estimated_trans['trans'])


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_raw_bids_pathlike():
    """Test that read_raw_bids() can handle a Path-like bids_root."""
    bids_root = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    raw = read_raw_bids(bids_path=bids_path)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_raw_datatype():
    """Test that read_raw_bids() can infer the str_suffix if need be."""
    bids_root = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
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


def test_handle_channel_type_casing():
    """Test that non-uppercase entries in the `type` column are accepted."""
    bids_root = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root)
    raw = _read_raw_fif(raw_fname, verbose=False)

    write_raw_bids(raw, bids_path, overwrite=True,
                   verbose=False)

    ch_path = bids_path.copy().update(root=bids_root,
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
def test_bads_reading():
    bids_root = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
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
def test_write_read_fif_split_file():
    """Test split files are read correctly."""
    bids_root = _TempDir()
    tmp_dir = _TempDir()
    bids_path = _bids_path.copy().update(root=bids_root, datatype='meg')
    raw = _read_raw_fif(raw_fname, verbose=False)
    n_channels = len(raw.ch_names)
    n_times = int(2.2e9 / (n_channels * 4))  # enough to produce a split
    data = np.empty((n_channels, n_times), dtype=np.float32)
    raw = mne.io.RawArray(data, raw.info)
    big_fif_fname = pathlib.Path(tmp_dir) / 'test_raw.fif'
    raw.save(big_fif_fname)
    raw = _read_raw_fif(big_fif_fname, verbose=False)
    write_raw_bids(raw, bids_path, verbose=False)

    raw1 = read_raw_bids(bids_path=bids_path)
    assert 'split-01' in str(bids_path.fpath)

    bids_path.update(split='01')
    raw2 = read_raw_bids(bids_path=bids_path)
    bids_path.update(split='02')
    raw3 = read_raw_bids(bids_path=bids_path)
    assert len(raw) == len(raw1)
    assert len(raw) == len(raw2)
    assert len(raw) > len(raw3)
