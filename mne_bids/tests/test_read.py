"""Testing utilities for file io."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import pytest

import numpy as np
import mne
from mne.utils import _TempDir, requires_nibabel
from mne.datasets import testing

from mne_bids.read import _read_raw, fit_trans_from_points
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
def test_fit_trans_from_points():
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
    write_raw_bids(raw, bids_basename, output_path, events_data=events_fname,
                   event_id=event_id, overwrite=False)

    # Write some MRI data and supply a `trans` so that a sidecar gets written
    trans = mne.read_trans(raw_fname.replace('_raw.fif', '-trans.fif'))

    # Get the T1 weighted MRI data file
    # Needs to be converted to Nifti because we only have mgh in our test base
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
    tmpdir = _TempDir()
    t1w_nii = op.join(tmpdir, 't1_nii.nii.gz')
    nib.save(nib.load(t1w_mgh), t1w_nii)

    write_anat(output_path, subject_id, t1w_nii, session_id, acq, raw=raw,
               trans=trans, verbose=True)

    # Try to get trans back through fitting points
    bids_fname = bids_basename + '_meg.fif'
    reproduced_trans = fit_trans_from_points(bids_fname, output_path,
                                             verbose=True)

    assert trans['from'] == reproduced_trans['from']
    assert trans['to'] == reproduced_trans['to']
    np.testing.assert_almost_equal(trans['trans'],
                                   reproduced_trans['trans'])
    print(trans)
    print(reproduced_trans)
