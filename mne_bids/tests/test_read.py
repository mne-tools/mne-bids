"""Testing utilities for file io."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op

import pytest
import shutil as sh

import numpy as np
from numpy.testing import assert_almost_equal

import mne
from mne.io import anonymize_info
from mne.utils import _TempDir, requires_nibabel
from mne.datasets import testing

import mne_bids
from mne_bids import get_matched_empty_room
from mne_bids.read import _read_raw, get_head_mri_trans, _handle_events_reading
from mne_bids.write import write_anat, write_raw_bids, make_bids_basename
from mne_bids.tsv_handler import _to_tsv

subject_id = '01'
session_id = '01'
run = '01'
acq = '01'
task = 'testing'

bids_basename = make_bids_basename(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)

# Get the MNE testing sample data
data_path = testing.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_raw.fif')


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
    events = {'onset': [11, 12, 13],
              'duration': ['n/a', 'n/a', 'n/a']
              }
    tmp_dir = _TempDir()
    events_fname = op.join(tmp_dir, 'sub-01_task-test_events.json')
    _to_tsv(events, events_fname)

    raw = _handle_events_reading(events_fname, raw)
    events, event_id = mne.events_from_annotations(raw)


@requires_nibabel()
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
    er_date = er_raw.info['meas_date'].strftime('%Y%m%d')
    er_bids_basename = make_bids_basename(subject='emptyroom',
                                          task='noise', session=er_date)
    write_raw_bids(er_raw, er_bids_basename, bids_root, overwrite=True)

    er_fname = get_matched_empty_room(bids_basename + '_meg.fif',
                                      bids_root)
    assert er_bids_basename in er_fname

    raw = mne_bids.read_raw_bids(bids_basename + '_meg.fif', bids_root)
    raw.set_meas_date(None)
    anonymize_info(raw.info)
    write_raw_bids(raw, bids_basename, bids_root, overwrite=True)
    with pytest.raises(ValueError, match='Measurement date not available'):
        get_matched_empty_room(bids_basename + '_meg.fif', bids_root)
