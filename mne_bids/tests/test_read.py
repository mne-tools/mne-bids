"""Testing utilities for file io."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import json
import os
import os.path as op
from datetime import datetime, timezone
from pathlib import Path

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
from mne.datasets import testing, somato

import mne_bids
from mne_bids import get_matched_empty_room
from mne_bids.read import (read_raw_bids,
                           _read_raw, get_head_mri_trans,
                           _handle_events_reading, _handle_info_reading)
from mne_bids.tsv_handler import _to_tsv, _from_tsv
from mne_bids.utils import (_find_matching_sidecar, _update_sidecar,
                            _write_json)
from mne_bids.write import write_anat, write_raw_bids, make_bids_basename

subject_id = '01'
session_id = '01'
run = '01'
acq = '01'
task = 'testing'

bids_basename = make_bids_basename(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)

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
    raw = mne.io.read_raw_fif(raw_fname)
    bids_root = _TempDir()
    with pytest.warns(UserWarning, match='No line frequency'):
        write_raw_bids(raw, bids_basename, bids_root,
                       events_data=events_fname, event_id=event_id,
                       overwrite=False)

    # We cannot recover trans, if no MRI has yet been written
    with pytest.raises(RuntimeError):
        bids_fname = bids_basename + '_meg.fif'
        estimated_trans = get_head_mri_trans(bids_fname, bids_root)

    # Write some MRI data and supply a `trans` so that a sidecar gets written
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    t1w_mgh = nib.load(t1w_mgh)

    anat_dir = write_anat(bids_root, subject_id, t1w_mgh, session_id, acq,
                          raw=raw, trans=trans, verbose=True)

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(bids_fname, bids_root)

    assert trans['from'] == estimated_trans['from']
    assert trans['to'] == estimated_trans['to']
    assert_almost_equal(trans['trans'], estimated_trans['trans'])
    print(trans)
    print(estimated_trans)

    # Passing a path instead of a name works well
    bids_fpath = op.join(bids_root, 'sub-{}'.format(subject_id),
                         'ses-{}'.format(session_id), 'meg',
                         bids_basename + '_meg.fif')
    estimated_trans = get_head_mri_trans(bids_fpath, bids_root)

    # provoke an error by pointing introducing NaNs into MEG coords
    with pytest.raises(RuntimeError, match='AnatomicalLandmarkCoordinates'):
        raw.info['dig'][0]['r'] = np.ones(3) * np.nan
        sh.rmtree(anat_dir)
        write_anat(bids_root, subject_id, t1w_mgh, session_id, acq, raw=raw,
                   trans=trans, verbose=True)
        estimated_trans = get_head_mri_trans(bids_fname, bids_root)


def test_handle_events_reading():
    """Test reading events from a BIDS events.tsv file."""
    # We can use any `raw` for this
    raw = mne.io.read_raw_fif(raw_fname)

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
def test_line_freq_estimation():
    """Test estimating line frequency."""
    bids_root = _TempDir()

    # read in USA dataset, so it should find 50 Hz
    raw = mne.io.read_raw_fif(raw_fname)
    kind = "meg"

    # assert that we get the same line frequency set
    bids_fname = bids_basename + '_{}.fif'.format(kind)

    # find sidecar JSON fname
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    sidecar_fname = _find_matching_sidecar(bids_fname, bids_root,
                                           '{}.json'.format(kind),
                                           allow_fail=True)

    # 1. when nothing is set, default to use PSD estimation -> should be 60
    # for `sample` dataset
    raw.info['line_freq'] = None
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    _update_sidecar(sidecar_fname, "PowerLineFrequency", "n/a")
    with pytest.warns(UserWarning, match="No line frequency found"):
        raw = mne_bids.read_raw_bids(bids_fname, bids_root)
        assert raw.info['line_freq'] == 60

    # test that `somato` dataset finds 50 Hz (EU dataset)
    somato_raw = mne.io.read_raw_fif(somato_raw_fname)
    somato_raw.info['line_freq'] = None
    write_raw_bids(somato_raw, bids_basename, bids_root, overwrite=True)
    sidecar_fname = _find_matching_sidecar(bids_fname, bids_root,
                                           '{}.json'.format(kind),
                                           allow_fail=True)
    _update_sidecar(sidecar_fname, "PowerLineFrequency", "n/a")
    with pytest.warns(UserWarning, match="No line frequency found"):
        somato_raw = mne_bids.read_raw_bids(bids_fname, bids_root)
        assert somato_raw.info['line_freq'] == 50

    # assert that line_freq should be None when
    # all picks are not meg/eeg/ecog/seeg
    somato_raw.info['line_freq'] = None
    somato_raw.set_channel_types({somato_raw.ch_names[i]: 'bio'
                                  for i in range(len(somato_raw.ch_names))})
    somato_raw = _handle_info_reading(sidecar_fname, somato_raw, verbose=True)
    assert somato_raw.info['line_freq'] is None


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_info_reading():
    """Test reading information from a BIDS sidecar.json file."""
    bids_root = _TempDir()

    # read in USA dataset, so it should find 50 Hz
    raw = mne.io.read_raw_fif(raw_fname)
    raw.info['line_freq'] = 60

    # write copy of raw with line freq of 60
    # bids basename and fname
    bids_basename = make_bids_basename(subject='01', session='01',
                                       task='audiovisual', run='01')
    kind = "meg"
    bids_fname = bids_basename + '_{}.fif'.format(kind)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)

    # find sidecar JSON fname
    sidecar_fname = _find_matching_sidecar(bids_fname, bids_root,
                                           '{}.json'.format(kind),
                                           allow_fail=True)

    # assert that we get the same line frequency set
    raw = mne_bids.read_raw_bids(bids_fname, bids_root)
    assert raw.info['line_freq'] == 60

    # 2. if line frequency is not set in raw file, then default to sidecar
    raw.info['line_freq'] = None
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    _update_sidecar(sidecar_fname, "PowerLineFrequency", 55)
    raw = mne_bids.read_raw_bids(bids_fname, bids_root)
    assert raw.info['line_freq'] == 55

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
    raw = mne_bids.read_raw_bids(bids_fname, bids_root)
    assert raw.info['line_freq'] == 55

    # 3. if line frequency is set in raw file, but not sidecar
    raw.info['line_freq'] = 60
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    _update_sidecar(sidecar_fname, "PowerLineFrequency", "n/a")
    raw = mne_bids.read_raw_bids(bids_fname, bids_root)
    assert raw.info['line_freq'] == 60

    # 4. assert that we get an error when sidecar json doesn't match
    _update_sidecar(sidecar_fname, "PowerLineFrequency", 55)
    with pytest.raises(ValueError, match="Line frequency in sidecar json"):
        raw = mne_bids.read_raw_bids(bids_fname, bids_root)


@pytest.mark.filterwarnings(warning_str['nasion_not_found'])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_handle_coords_reading():
    """Test reading coordinates from BIDS files."""
    bids_root = _TempDir()

    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')

    raw = mne.io.read_raw_edf(raw_fname)

    # ensure we are writing 'ecog'/'ieeg' data
    raw.set_channel_types({ch: 'ecog'
                           for ch in raw.ch_names})

    # set a `random` montage
    ch_names = raw.ch_names
    elec_locs = np.random.random((len(ch_names), 3)).astype(float)
    ch_pos = dict(zip(ch_names, elec_locs))
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                            coord_frame="mri")
    raw.set_montage(montage)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)

    # read in the data and assert montage is the same
    bids_fname = bids_basename + "_ieeg.edf"
    raw_test = read_raw_bids(bids_fname, bids_root)

    # obtain the sensor positions
    orig_locs = raw.info['dig'][1]
    test_locs = raw_test.info['dig'][1]
    assert orig_locs == test_locs
    assert not object_diff(raw.info['chs'], raw_test.info['chs'])

    # test error message if electrodes don't match
    electrodes_fname = _find_matching_sidecar(bids_fname, bids_root,
                                              "electrodes.tsv",
                                              allow_fail=True)
    electrodes_dict = _from_tsv(electrodes_fname)
    # pop off 5 channels
    for key in electrodes_dict.keys():
        for i in range(5):
            electrodes_dict[key].pop()
    _to_tsv(electrodes_dict, electrodes_fname)
    with pytest.raises(RuntimeError, match='Channels do not correspond'):
        raw_test = read_raw_bids(bids_fname, bids_root)


@requires_nibabel()
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_get_head_mri_trans_ctf():
    """Test getting a trans object from BIDS data in CTF."""
    import nibabel as nib

    ctf_data_path = op.join(testing.data_path(), 'CTF')
    raw_ctf_fname = op.join(ctf_data_path, 'testdata_ctf.ds')
    raw_ctf = mne.io.read_raw_ctf(raw_ctf_fname)
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
    bids_fname = bids_basename + '_meg.ds'
    estimated_trans = get_head_mri_trans(bids_fname, bids_root)
    assert_almost_equal(trans['trans'], estimated_trans['trans'])

    # Test if Path objects work.
    estimated_trans = get_head_mri_trans(Path(bids_fname), Path(bids_root))
    assert_almost_equal(trans['trans'], estimated_trans['trans'])


@pytest.mark.filterwarnings(warning_str['meas_date_set_to_none'])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_get_matched_empty_room():
    """Test reading of empty room data."""
    bids_root = _TempDir()

    raw = mne.io.read_raw_fif(raw_fname)
    bids_basename = make_bids_basename(subject='01', session='01',
                                       task='audiovisual', run='01')
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)

    er_fname = get_matched_empty_room(bids_basename + '_meg.fif',
                                      bids_root)
    assert er_fname is None

    # testing data has no noise recording, so save the actual data
    # as if it were noise
    er_raw_fname = op.join(data_path, 'MEG', 'sample', 'ernoise_raw.fif')
    raw.crop(0, 10).save(er_raw_fname, overwrite=True)

    er_raw = mne.io.read_raw_fif(er_raw_fname)
    er_date = er_raw.info['meas_date']
    if not isinstance(er_date, datetime):
        # mne < v0.20
        er_date = datetime.fromtimestamp(er_raw.info['meas_date'][0])
    er_date = er_date.strftime('%Y%m%d')
    er_bids_basename = make_bids_basename(subject='emptyroom',
                                          task='noise', session=er_date)
    write_raw_bids(er_raw, er_bids_basename, bids_root, overwrite=True)

    er_fname = get_matched_empty_room(bids_basename + '_meg.fif',
                                      bids_root)
    assert er_bids_basename in er_fname

    # Test if Path objects work.
    er_fname = get_matched_empty_room(bids_basename + '_meg.fif',
                                      Path(bids_root))
    assert isinstance(er_fname, str)
    assert er_bids_basename in er_fname

    # assert that we get best emptyroom if there are multiple available
    sh.rmtree(op.join(bids_root, 'sub-emptyroom'))
    dates = ['20021204', '20021201', '20021001']
    for date in dates:
        er_bids_basename = make_bids_basename(subject='emptyroom',
                                              task='noise', session=date)
        er_meas_date = datetime.strptime(date, '%Y%m%d')
        er_meas_date = er_meas_date.replace(tzinfo=timezone.utc)

        if check_version('mne', '0.20'):
            er_raw.set_meas_date(er_meas_date)
        else:
            er_raw.info['meas_date'] = (er_meas_date.timestamp(), 0)
        write_raw_bids(er_raw, er_bids_basename, bids_root)

    best_er_fname = get_matched_empty_room(bids_basename + '_meg.fif',
                                           bids_root)
    assert '20021204' in best_er_fname

    # assert that we get error if meas_date is not available.
    raw = mne_bids.read_raw_bids(bids_basename + '_meg.fif', bids_root)
    if check_version('mne', '0.20'):
        raw.set_meas_date(None)
    else:
        raw.info['meas_date'] = None
        raw.annotations.orig_time = None
    anonymize_info(raw.info)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    with pytest.raises(ValueError, match='Measurement date not available'):
        get_matched_empty_room(bids_basename + '_meg.fif', bids_root)


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_read_raw_bids_pathlike():
    """Test that read_raw_bids() can handle a Path-like bids_root."""
    bids_root = _TempDir()
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True,
                   verbose=False)

    bids_fname = bids_basename + '_meg.fif'
    read_raw_bids(bids_fname, Path(bids_root))
