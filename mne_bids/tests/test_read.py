"""Testing utilities for file io."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import json
import os
import os.path as op
from datetime import datetime, timezone
from pathlib import Path
from distutils.version import LooseVersion

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

from mne.io import anonymize_info
from mne.utils import _TempDir, requires_nibabel, check_version, object_diff
from mne.utils import assert_dig_allclose
from mne.datasets import testing, somato

from mne_bids import (get_matched_empty_room, BIDSPath,
                      make_bids_folders)
from mne_bids.config import MNE_STR_TO_FRAME
from mne_bids.read import (read_raw_bids,
                           _read_raw, get_head_mri_trans,
                           _handle_events_reading)
from mne_bids.tsv_handler import _to_tsv, _from_tsv
from mne_bids.utils import (_update_sidecar,
                            _write_json)
from mne_bids.path import _find_matching_sidecar
from mne_bids.write import write_anat, write_raw_bids

subject_id = '01'
session_id = '01'
run = '01'
acq = '01'
task = 'testing'

bids_basename = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)

bids_basename_minimal = BIDSPath(subject=subject_id, task=task)

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
        with open(raw_fname, 'w'):
            pass
        with pytest.raises(ValueError, match=('there is no IO support for '
                                              'this file format yet')):
            _read_raw(raw_fname)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_participants_data():
    """Test reading information from a BIDS sidecar.json file."""
    bids_root = _TempDir()
    raw = _read_raw_fif(raw_fname, verbose=False)

    # if subject info was set, we don't roundtrip birthday
    # due to possible anonymization in mne-bids
    subject_info = {
        'hand': 1,
        'sex': 2,
    }
    raw.info['subject_info'] = subject_info
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)
    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        kind='meg')
    print(raw.info['subject_info'])
    assert raw.info['subject_info']['hand'] == 1
    assert raw.info['subject_info']['sex'] == 2
    assert raw.info['subject_info'].get('birthday', None) is None

    # if modifying participants tsv, then read_raw_bids reflects that
    participants_tsv_fpath = op.join(bids_root, 'participants.tsv')
    participants_tsv = _from_tsv(participants_tsv_fpath)
    participants_tsv['hand'][0] = 'n/a'
    _to_tsv(participants_tsv, participants_tsv_fpath)
    raw = read_raw_bids(bids_basename=bids_basename, bids_root=Path(bids_root),
                        kind='meg')
    assert raw.info['subject_info']['hand'] == 0
    assert raw.info['subject_info']['sex'] == 2
    assert raw.info['subject_info'].get('birthday', None) is None

    # make sure things are read even if the entries don't make sense
    participants_tsv = _from_tsv(participants_tsv_fpath)
    participants_tsv['hand'][0] = 'righty'
    participants_tsv['sex'][0] = 'malesy'
    _to_tsv(participants_tsv, participants_tsv_fpath)
    with pytest.warns(RuntimeWarning, match='Unable to map'):
        raw = read_raw_bids(bids_basename=bids_basename,
                            bids_root=Path(bids_root), kind='meg')
        assert raw.info['subject_info']['hand'] is None
        assert raw.info['subject_info']['sex'] is None

    # make sure to read in if no participants file
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)
    os.remove(participants_tsv_fpath)
    with pytest.warns(RuntimeWarning, match='Participants file not found'):
        raw = read_raw_bids(bids_basename=bids_basename,
                            bids_root=Path(bids_root), kind='meg')
        assert raw.info['subject_info'] is None


@requires_nibabel()
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_get_head_mri_trans():
    """Test getting a trans object from BIDS data."""
    import nibabel as nib

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Write it to BIDS
    raw = _read_raw_fif(raw_fname)
    bids_root = _TempDir()
    write_raw_bids(raw, bids_basename, bids_root,
                   events_data=events_fname, event_id=event_id,
                   overwrite=False)

    # We cannot recover trans, if no MRI has yet been written
    with pytest.raises(RuntimeError):
        estimated_trans = get_head_mri_trans(bids_basename=bids_basename,
                                             bids_root=bids_root)

    # Write some MRI data and supply a `trans` so that a sidecar gets written
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    t1w_mgh = nib.load(t1w_mgh)

    anat_dir = write_anat(bids_root, subject_id, t1w_mgh, session_id, acq,
                          raw=raw, trans=trans, verbose=True)

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(bids_basename=bids_basename,
                                         bids_root=bids_root)

    assert trans['from'] == estimated_trans['from']
    assert trans['to'] == estimated_trans['to']
    assert_almost_equal(trans['trans'], estimated_trans['trans'])
    print(trans)
    print(estimated_trans)

    # provoke an error by introducing NaNs into MEG coords
    with pytest.raises(RuntimeError, match='AnatomicalLandmarkCoordinates'):
        raw.info['dig'][0]['r'] = np.ones(3) * np.nan
        sh.rmtree(anat_dir)
        write_anat(bids_root, subject_id, t1w_mgh, session_id, acq, raw=raw,
                   trans=trans, verbose=True)
        estimated_trans = get_head_mri_trans(bids_basename=bids_basename,
                                             bids_root=bids_root)


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
    bids_basename = BIDSPath(subject='01', session='01',
                             task='audiovisual', run='01')
    kind = "meg"
    bids_fname = bids_basename.copy().update(kind=kind,
                                             extension='.fif')
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)

    # find sidecar JSON fname
    sidecar_fname = _find_matching_sidecar(bids_fname, bids_root,
                                           kind=kind, extension='.json',
                                           allow_fail=True)

    # assert that we get the same line frequency set
    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        kind=kind)
    assert raw.info['line_freq'] == 60

    # 2. if line frequency is not set in raw file, then ValueError
    raw.info['line_freq'] = None
    with pytest.raises(ValueError, match="PowerLineFrequency .* required"):
        write_raw_bids(raw, bids_basename, bids_root, overwrite=True)

    # make a copy of the sidecar in "derivatives/"
    # to check that we make sure we always get the right sidecar
    # in addition, it should not break the sidecar reading
    # in `read_raw_bids`
    deriv_dir = op.join(bids_root, "derivatives")
    sidecar_copy = op.join(deriv_dir, op.basename(sidecar_fname))
    os.mkdir(deriv_dir)
    with open(sidecar_fname, "r") as fin:
        sidecar_json = json.load(fin)
        sidecar_json["PowerLineFrequency"] = 45
    _write_json(sidecar_copy, sidecar_json)
    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        kind=kind)
    assert raw.info['line_freq'] == 60

    # 3. assert that we get an error when sidecar json doesn't match
    _update_sidecar(sidecar_fname, "PowerLineFrequency", 55)
    with pytest.raises(ValueError, match="Line frequency in sidecar json"):
        raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                            kind=kind)
        assert raw.info['line_freq'] == 55


@pytest.mark.filterwarnings(warning_str['nasion_not_found'])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_eeg_coords_reading():
    """Test reading iEEG coordinates from BIDS files."""
    bids_root = _TempDir()

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
        write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
        coordsystem_fname = _find_matching_sidecar(bids_basename, bids_root,
                                                   kind='coordsystem',
                                                   extension='.json',
                                                   allow_fail=True)
        electrodes_fname = _find_matching_sidecar(bids_basename, bids_root,
                                                  kind='electrodes',
                                                  extension='.tsv',
                                                  allow_fail=True)
        assert coordsystem_fname is None
        assert electrodes_fname is None

    # create montage in head frame and set should result in
    # warning if landmarks not set
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                            coord_frame="head")
    raw.set_montage(montage)
    with pytest.warns(RuntimeWarning, match='Setting montage not possible '
                                            'if anatomical landmarks'):
        write_raw_bids(raw, bids_basename, bids_root, overwrite=True)

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                            coord_frame="head",
                                            nasion=[1, 0, 0],
                                            lpa=[0, 1, 0],
                                            rpa=[0, 0, 1])
    raw.set_montage(montage)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)

    # obtain the sensor positions and assert ch_coords are same
    raw_test = read_raw_bids(bids_basename, bids_root, verbose=True)
    assert not object_diff(raw.info['chs'], raw_test.info['chs'])

    # modify coordinate frame to not-captrak
    coordsystem_fname = _find_matching_sidecar(bids_basename, bids_root,
                                               kind='coordsystem',
                                               extension='.json',
                                               allow_fail=True)
    _update_sidecar(coordsystem_fname, 'EEGCoordinateSystem', 'besa')
    with pytest.warns(RuntimeWarning, match='EEG Coordinate frame is not '
                                            'accepted BIDS keyword'):
        raw_test = read_raw_bids(bids_basename, bids_root)
        assert raw_test.info['dig'] is None


@pytest.mark.parametrize('bids_basename', [bids_basename,
                                           bids_basename_minimal])
@pytest.mark.filterwarnings(warning_str['nasion_not_found'])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_ieeg_coords_reading(bids_basename):
    """Test reading iEEG coordinates from BIDS files."""
    bids_root = _TempDir()

    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')
    bids_fname = bids_basename.copy().update(kind='ieeg',
                                             extension='.edf')

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
        write_raw_bids(raw, bids_basename, bids_root,
                       overwrite=True, verbose=False)
        # read in raw file w/ updated coordinate frame
        # and make sure all digpoints are correct coordinate frames
        raw_test = read_raw_bids(bids_basename=bids_basename,
                                 bids_root=bids_root, verbose=False)
        coord_frame_int = MNE_STR_TO_FRAME[coord_frame]
        for digpoint in raw_test.info['dig']:
            assert digpoint['coord_frame'] == coord_frame_int

    # start w/ new bids root
    sh.rmtree(bids_root)
    write_raw_bids(raw, bids_basename, bids_root,
                   overwrite=True, verbose=False)

    # obtain the sensor positions and assert ch_coords are same
    raw_test = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                             verbose=False)
    orig_locs = raw.info['dig'][1]
    test_locs = raw_test.info['dig'][1]
    assert orig_locs == test_locs
    assert not object_diff(raw.info['chs'], raw_test.info['chs'])

    # read in the data and assert montage is the same
    # regardless of 'm', 'cm', 'mm', or 'pixel'
    scalings = {'m': 1, 'cm': 100, 'mm': 1000}
    coordsystem_fname = _find_matching_sidecar(bids_fname, bids_root,
                                               kind='coordsystem',
                                               extension='.json',
                                               allow_fail=True)
    electrodes_fname = _find_matching_sidecar(bids_fname, bids_root,
                                              kind='electrodes',
                                              extension='.tsv',
                                              allow_fail=True)
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
        raw_test = read_raw_bids(bids_basename=bids_basename,
                                 bids_root=bids_root, verbose=False)

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
        raw_test = read_raw_bids(bids_basename=bids_basename,
                                 bids_root=bids_root, verbose=False)

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
            raw_test = read_raw_bids(bids_basename=bids_basename,
                                     bids_root=bids_root, verbose=False)
            assert raw_test.info['dig'] is None

    # ACPC should be read in as RAS for iEEG
    _update_sidecar(coordsystem_fname, 'iEEGCoordinateSystem', 'acpc')
    raw_test = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                             verbose=False)
    coord_frame_int = MNE_STR_TO_FRAME['ras']
    for digpoint in raw_test.info['dig']:
        assert digpoint['coord_frame'] == coord_frame_int

    # if we delete the coordsystem.json file, an error will be raised
    os.remove(coordsystem_fname)
    with pytest.raises(RuntimeError, match='BIDS mandates that '
                                           'the coordsystem.json'):
        raw = read_raw_bids(bids_basename=bids_basename,
                            bids_root=bids_root, verbose=False)

    # test error message if electrodes don't match
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    electrodes_dict = _from_tsv(electrodes_fname)
    # pop off 5 channels
    for key in electrodes_dict.keys():
        for i in range(5):
            electrodes_dict[key].pop()
    _to_tsv(electrodes_dict, electrodes_fname)
    with pytest.raises(RuntimeError, match='Channels do not correspond'):
        raw_test = read_raw_bids(bids_basename=bids_basename,
                                 bids_root=bids_root, verbose=False)

    # make sure montage is set if there are coordinates w/ 'n/a'
    raw.info['bads'] = []
    write_raw_bids(raw, bids_basename, bids_root,
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
        raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                            verbose=False)
        for idx, ch in enumerate(raw.info['chs']):
            if ch['ch_name'] in nan_chs:
                assert all(np.isnan(ch['loc'][:3]))
            else:
                assert not any(np.isnan(ch['loc'][:3]))
            assert ch['ch_name'] not in raw.info['bads']


@requires_nibabel()
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_get_head_mri_trans_ctf():
    """Test getting a trans object from BIDS data in CTF."""
    import nibabel as nib

    ctf_data_path = op.join(testing.data_path(), 'CTF')
    raw_ctf_fname = op.join(ctf_data_path, 'testdata_ctf.ds')
    raw_ctf = _read_raw_ctf(raw_ctf_fname)
    bids_root = _TempDir()
    write_raw_bids(raw_ctf, bids_basename, bids_root,
                   overwrite=False)

    # Take a fake trans
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    t1w_mgh = nib.load(t1w_mgh)

    write_anat(bids_root, subject_id, t1w_mgh, session_id, acq,
               raw=raw_ctf, trans=trans)

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(bids_basename=bids_basename,
                                         bids_root=bids_root)

    assert_almost_equal(trans['trans'], estimated_trans['trans'])


@pytest.mark.filterwarnings(warning_str['meas_date_set_to_none'])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_get_matched_empty_room():
    """Test reading of empty room data."""
    bids_root = _TempDir()

    raw = _read_raw_fif(raw_fname)

    bids_basename = BIDSPath(subject='01', session='01',
                             task='audiovisual', run='01')
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)

    er_basename = get_matched_empty_room(bids_basename=bids_basename,
                                         bids_root=bids_root)
    assert er_basename is None

    # testing data has no noise recording, so save the actual data
    # as if it were noise
    er_raw_fname = op.join(data_path, 'MEG', 'sample', 'ernoise_raw.fif')
    raw.crop(0, 10).save(er_raw_fname, overwrite=True)

    er_raw = _read_raw_fif(er_raw_fname)
    er_date = er_raw.info['meas_date']
    if not isinstance(er_date, datetime):
        # mne < v0.20
        er_date = datetime.fromtimestamp(er_raw.info['meas_date'][0])
    er_date = er_date.strftime('%Y%m%d')
    er_bids_basename = BIDSPath(subject='emptyroom', task='noise',
                                session=er_date, kind='meg')
    write_raw_bids(er_raw, er_bids_basename, bids_root, overwrite=True)

    recovered_er_basename = get_matched_empty_room(bids_basename=bids_basename,
                                                   bids_root=bids_root)
    assert er_bids_basename.basename == recovered_er_basename

    # assert that we get best emptyroom if there are multiple available
    sh.rmtree(op.join(bids_root, 'sub-emptyroom'))
    dates = ['20021204', '20021201', '20021001']
    for date in dates:
        er_bids_basename.update(session=date)
        er_meas_date = datetime.strptime(date, '%Y%m%d')
        er_meas_date = er_meas_date.replace(tzinfo=timezone.utc)

        if check_version('mne', '0.20'):
            er_raw.set_meas_date(er_meas_date)
        else:
            er_raw.info['meas_date'] = (er_meas_date.timestamp(), 0)
        write_raw_bids(er_raw, er_bids_basename, bids_root)

    best_er_basename = get_matched_empty_room(bids_basename=bids_basename,
                                              bids_root=bids_root)
    assert '20021204' in best_er_basename

    # assert that we get error if meas_date is not available.
    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        kind='meg')
    if check_version('mne', '0.20'):
        raw.set_meas_date(None)
    else:
        raw.info['meas_date'] = None
        raw.annotations.orig_time = None
    anonymize_info(raw.info)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    with pytest.raises(ValueError, match='The provided recording does not '
                                         'have a measurement date set'):
        get_matched_empty_room(bids_basename=bids_basename,
                               bids_root=bids_root)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_get_matched_emptyroom_ties():
    """Test that we receive a warning on a date tie."""
    bids_root = _TempDir()
    session = '20010101'
    er_dir = make_bids_folders(subject='emptyroom', session=session,
                               kind='meg', bids_root=bids_root)

    meas_date = (datetime
                 .strptime(session, '%Y%m%d')
                 .replace(tzinfo=timezone.utc))

    raw = _read_raw_fif(raw_fname)

    er_raw_fname = op.join(data_path, 'MEG', 'sample', 'ernoise_raw.fif')
    raw.copy().crop(0, 10).save(er_raw_fname, overwrite=True)
    er_raw = _read_raw_fif(er_raw_fname)

    if check_version('mne', '0.20'):
        raw.set_meas_date(meas_date)
        er_raw.set_meas_date(meas_date)
    else:
        raw.info['meas_date'] = (meas_date.timestamp(), 0)
        er_raw.info['meas_date'] = (meas_date.timestamp(), 0)

    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    er_bids_path = BIDSPath(subject='emptyroom', session=session)
    er_basename_1 = str(er_bids_path)
    er_basename_2 = BIDSPath(subject='emptyroom', session=session,
                             task='noise')
    er_raw.save(op.join(er_dir, f'{er_basename_1}_meg.fif'))
    er_raw.save(op.join(er_dir, f'{er_basename_2}_meg.fif'))

    with pytest.warns(RuntimeWarning, match='Found more than one'):
        get_matched_empty_room(bids_basename=bids_basename,
                               bids_root=bids_root)


@pytest.mark.skipif(LooseVersion(mne.__version__) < LooseVersion('0.21'),
                    reason="requires mne 0.21.dev0 or higher")
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_get_matched_emptyroom_no_meas_date():
    """Test that we warn if measurement date can be read or inferred."""
    bids_root = _TempDir()
    er_session = 'mysession'
    er_meas_date = None

    er_dir = make_bids_folders(subject='emptyroom', session=er_session,
                               kind='meg', bids_root=bids_root)
    er_bids_path = BIDSPath(subject='emptyroom', session=er_session,
                            task='noise', check=False)
    er_basename = str(er_bids_path)
    raw = _read_raw_fif(raw_fname)

    er_raw_fname = op.join(data_path, 'MEG', 'sample', 'ernoise_raw.fif')
    raw.copy().crop(0, 10).save(er_raw_fname, overwrite=True)
    er_raw = _read_raw_fif(er_raw_fname)
    er_raw.set_meas_date(er_meas_date)
    er_raw.save(op.join(er_dir, f'{er_basename}_meg.fif'), overwrite=True)

    # Write raw file data using mne-bids, and remove participants.tsv
    # as it's incomplete (doesn't contain the emptyroom subject we wrote
    # manually using MNE's Raw.save() above)
    raw = _read_raw_fif(raw_fname)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    os.remove(op.join(bids_root, 'participants.tsv'))

    with pytest.warns(RuntimeWarning, match='Could not retrieve .* date'):
        get_matched_empty_room(bids_basename=bids_basename,
                               bids_root=bids_root)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_raw_bids_pathlike():
    """Test that read_raw_bids() can handle a Path-like bids_root."""
    bids_root = _TempDir()
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)
    raw = read_raw_bids(bids_basename=bids_basename, bids_root=Path(bids_root),
                        kind='meg')


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_raw_kind():
    """Test that read_raw_bids() can infer the kind if need be."""
    bids_root = _TempDir()
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)

    raw_1 = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                          kind='meg')
    raw_2 = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                          kind=None)
    raw_3 = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root)

    raw_1.crop(0, 2).load_data()
    raw_2.crop(0, 2).load_data()
    raw_3.crop(0, 2).load_data()

    assert raw_1 == raw_2
    assert raw_1 == raw_3


def test_handle_channel_type_casing():
    """Test that non-uppercase entries in the `type` column are accepted."""
    bids_root = _TempDir()
    raw = _read_raw_fif(raw_fname, verbose=False)

    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)

    subject_path = op.join(bids_root, f'sub-{subject_id}', f'ses-{session_id}',
                           'meg')
    bids_channels_fname = (bids_basename.copy()
                           .update(prefix=subject_path,
                                   kind='channels', extension='.tsv'))

    # Convert all channel type entries to lowercase.
    channels_data = _from_tsv(bids_channels_fname)
    channels_data['type'] = [t.lower() for t in channels_data['type']]
    _to_tsv(channels_data, bids_channels_fname)

    with pytest.warns(RuntimeWarning, match='lowercase spelling'):
        read_raw_bids(bids_basename, bids_root=bids_root)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_bads_reading():
    bids_root = _TempDir()
    channels_fname = (bids_basename.copy()
                      .update(prefix=op.join(bids_root, 'sub-01', 'ses-01',
                                             'meg'),
                              kind='channels', extension='.tsv'))
    raw_bids_fname = (bids_basename.copy()
                      .update(prefix=op.join(bids_root, 'sub-01', 'ses-01',
                                             'meg'),
                              kind='meg', extension='.fif'))
    raw = _read_raw_fif(raw_fname, verbose=False)

    ###########################################################################
    # bads in FIF only, no `status` column in channels.tsv
    bads = ['EEG 053', 'MEG 2443']
    raw.info['bads'] = bads
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)

    # Delete `status` column
    tsv_data = _from_tsv(channels_fname)
    del tsv_data['status'], tsv_data['status_description']
    _to_tsv(tsv_data, fname=channels_fname)

    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        verbose=False)
    assert raw.info['bads'] == bads

    ###########################################################################
    # bads in `status` column in channels.tsv, no bads in raw.info['bads']
    bads = ['EEG 053', 'MEG 2443']
    raw.info['bads'] = bads
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)

    # Remove info['bads'] from the raw file.
    raw = _read_raw_fif(raw_bids_fname, preload=True, verbose=False)
    raw.info['bads'] = []
    raw.save(raw_bids_fname, overwrite=True, verbose=False)

    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        verbose=False)
    assert type(raw.info['bads']) is list
    assert set(raw.info['bads']) == set(bads)

    ###########################################################################
    # Different bads in `status` column and raw.info['bads']
    bads_bids = ['EEG 053', 'MEG 2443']
    bads_raw = ['MEG 0112', 'MEG 0131']

    raw.info['bads'] = bads_bids
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)

    # Replace info['bads'] in the raw file.
    raw = _read_raw_fif(raw_bids_fname, preload=True, verbose=False)
    raw.info['bads'] = bads_raw
    raw.save(raw_bids_fname, overwrite=True, verbose=False)

    with pytest.warns(RuntimeWarning, match='conflicting information'):
        raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                            verbose=False)
    assert type(raw.info['bads']) is list
    assert set(raw.info['bads']) == set(bads_bids)
