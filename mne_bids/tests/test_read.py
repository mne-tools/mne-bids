"""Testing utilities for file io."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import pytest
import shutil as sh

import numpy as np
import mne
from mne.utils import _TempDir, requires_nibabel
from mne.datasets import testing

from mne_bids.read import _read_raw, get_head_mri_trans
from mne_bids.write import write_anat, write_raw_bids, make_bids_basename

subject_id = '01'
session_id = '01'
run = '01'
acq = '01'
task = 'testing'

bids_basename = make_bids_basename(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)


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
    # Get the MNE testing sample data
    output_path = _TempDir()
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    # Write it to BIDS
    raw = mne.io.read_raw_fif(raw_fname)
    with pytest.warns(UserWarning, match='No line frequency'):
        write_raw_bids(raw, bids_basename, output_path,
                       events_data=events_fname, event_id=event_id,
                       overwrite=False)

    # We cannot recover trans, if no MRI has yet been written
    with pytest.raises(RuntimeError):
        bids_fname = bids_basename + '_meg.fif'
        estimated_trans = get_head_mri_trans(bids_fname, output_path)

    # Write some MRI data and supply a `trans` so that a sidecar gets written
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    t1w_mgh = nib.load(t1w_mgh)

    anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id, acq,
                          raw=raw, trans=trans, verbose=True)

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(bids_fname, output_path)

    assert trans['from'] == estimated_trans['from']
    assert trans['to'] == estimated_trans['to']
    np.testing.assert_almost_equal(trans['trans'],
                                   estimated_trans['trans'])
    print(trans)
    print(estimated_trans)

    # Passing a path instead of a name works well
    bids_fpath = op.join(output_path, 'sub-{}'.format(subject_id),
                         'ses-{}'.format(session_id), 'meg',
                         bids_basename + '_meg.fif')
    estimated_trans = get_head_mri_trans(bids_fpath, output_path)

    # provoke an error by pointing introducing NaNs into MEG coords
    with pytest.raises(RuntimeError, match='AnatomicalLandmarkCoordinates'):
        raw.info['dig'][0]['r'] = np.ones(3) * np.nan
        sh.rmtree(anat_dir)
        write_anat(output_path, subject_id, t1w_mgh, session_id, acq, raw=raw,
                   trans=trans, verbose=True)
        estimated_trans = get_head_mri_trans(bids_fname, output_path)
