# -*- coding: utf-8 -*-
"""Test the digitizations.

For each supported coordinate frame, implement a test.
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

import os
import os.path as op
import numpy as np
import pytest

import mne
from mne.datasets import testing
from mne_bids import BIDSPath, write_raw_bids
from mne_bids.dig import _write_dig_bids, _read_dig_bids
from mne_bids.config import ALLOWED_SPACES_WRITE, BIDS_TO_MNE_FRAMES

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
session_id = '01'
run = '01'
acq = '01'
run2 = '02'
task = 'testing'

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)

data_path = testing.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_raw.fif')
raw = mne.io.read_raw(raw_fname)
raw.info['line_freq'] = 60
montage = raw.get_montage()


def test_dig_io(tmp_path):
    """Test passing different coordinate frames give proper warnings."""
    bids_root = tmp_path / 'bids1'
    for datatype in ('eeg', 'ieeg'):
        os.makedirs(op.join(bids_root, 'sub-01', 'ses-01', datatype))

    # test no coordinate frame in dig or in bids_path.space
    mnt = montage.copy()
    mnt.apply_trans(mne.transforms.Transform('head', 'unknown'))
    for datatype in ('eeg', 'ieeg'):
        bids_path = _bids_path.copy().update(root=bids_root, datatype=datatype,
                                             space=None)
        with pytest.warns(RuntimeWarning,
                          match='Coordinate frame could not be inferred'):
            _write_dig_bids(bids_path, raw, mnt, acpc_aligned=True)

    # test coordinate frame-BIDSPath.space mismatch
    mnt = montage.copy()
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype='eeg', space='fsaverage')
    with pytest.raises(ValueError, match='Coordinates in the montage'):
        _write_dig_bids(bids_path, raw, mnt)

    # test weird raw coordinate frame with BIDSPath.space
    mnt = montage.copy()
    mnt.apply_trans(mne.transforms.Transform('head', 'meg'))
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype='eeg', space='CapTrak')
    with pytest.raises(ValueError, match='inconsistent'):
        _write_dig_bids(bids_path, raw, mnt)

    # test MEG space conflict fif (ElektaNeuromag) != CTF
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype='meg', space='CTF')
    with pytest.raises(ValueError, match='conflicts'):
        write_raw_bids(raw, bids_path)


def test_dig_eeg_ieeg(tmp_path):
    """Test that eeg and ieeg dig are stored properly."""
    bids_root = tmp_path / 'bids1'
    for datatype in ('eeg', 'ieeg'):
        os.makedirs(op.join(bids_root, 'sub-01', 'ses-01', datatype))

    for datatype in ('eeg', 'ieeg'):
        bids_path = _bids_path.copy().update(root=bids_root, datatype=datatype)
        for coord_frame in ALLOWED_SPACES_WRITE[datatype]:
            bids_path.update(space=coord_frame)
            mnt = montage.copy()
            pos = mnt.get_positions()
            mne_coord_frame = BIDS_TO_MNE_FRAMES.get(coord_frame, None)
            if mne_coord_frame is None:
                mnt.apply_trans(mne.transforms.Transform('head', 'unknown'))
            else:
                mnt.apply_trans(mne.transforms.Transform(
                    'head', mne_coord_frame))
            _write_dig_bids(bids_path, raw, mnt, acpc_aligned=True)
            electrodes_path = bids_path.copy().update(
                task=None, run=None, suffix='electrodes', extension='.tsv')
            coordsystem_path = bids_path.copy().update(
                task=None, run=None, suffix='coordsystem', extension='.json')
            if bids_path.space == 'Pixels':
                with pytest.warns(RuntimeWarning,
                                  match='not recognized by MNE'):
                    _read_dig_bids(electrodes_path, coordsystem_path,
                                   datatype, raw)
            elif mne_coord_frame is None:
                with pytest.warns(RuntimeWarning,
                                  match='not implemented in MNE'):
                    _read_dig_bids(electrodes_path, coordsystem_path,
                                   datatype, raw)
            else:
                _read_dig_bids(electrodes_path, coordsystem_path,
                               datatype, raw)
            mnt2 = raw.get_montage()
            pos2 = mnt2.get_positions()
            np.testing.assert_array_almost_equal(
                np.array(list(pos['ch_pos'].values())),
                np.array(list(pos2['ch_pos'].values())))
            if mne_coord_frame is None:
                assert pos2['coord_frame'] == 'unknown'
            else:
                assert pos2['coord_frame'] == mne_coord_frame
