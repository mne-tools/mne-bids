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
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids
from mne_bids.dig import _write_dig_bids, _read_dig_bids
from mne_bids.config import (BIDS_STANDARD_TEMPLATE_COORDINATE_FRAMES,
                             BIDS_TO_MNE_FRAMES)

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
raw.drop_channels(raw.info['bads'])
raw.info['line_freq'] = 60
montage = raw.get_montage()


def test_dig_io(tmp_path):
    """Test passing different coordinate frames give proper warnings."""
    bids_root = tmp_path / 'bids1'
    raw_test = raw.copy()
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
            _write_dig_bids(bids_path, raw_test, mnt, acpc_aligned=True)

    # test coordinate frame-BIDSPath.space mismatch
    mnt = montage.copy()
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype='eeg', space='fsaverage')
    with pytest.raises(ValueError, match='Coordinates in the raw object '
                                         'or montage are in the CapTrak '
                                         'coordinate frame but '
                                         'BIDSPath.space is fsaverage'):
        _write_dig_bids(bids_path, raw_test, mnt)

    # test MEG space conflict fif (ElektaNeuromag) != CTF
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype='meg', space='CTF')
    with pytest.raises(ValueError, match='conflicts'):
        write_raw_bids(raw_test, bids_path)


def test_dig_pixels(tmp_path):
    """Test dig stored correctly for the Pixels coordinate frame."""
    bids_root = tmp_path / 'bids1'

    # test coordinates in pixels
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype='ieeg', space='Pixels')
    os.makedirs(op.join(bids_root, 'sub-01', 'ses-01', bids_path.datatype),
                exist_ok=True)
    raw_test = raw.copy()
    raw_test.pick_types(eeg=True)
    raw_test.del_proj()
    raw_test.set_channel_types({ch: 'ecog' for ch in raw_test.ch_names})

    mnt = raw_test.get_montage()
    # fake transform to pixel coordinates
    mnt.apply_trans(mne.transforms.Transform('head', 'unknown'))
    _write_dig_bids(bids_path, raw_test, mnt)
    electrodes_path = bids_path.copy().update(
        task=None, run=None, suffix='electrodes', extension='.tsv')
    coordsystem_path = bids_path.copy().update(
        task=None, run=None, suffix='coordsystem', extension='.json')
    with pytest.warns(RuntimeWarning,
                      match='not an MNE-Python coordinate frame'):
        _read_dig_bids(electrodes_path, coordsystem_path,
                       bids_path.datatype, raw_test)
    mnt2 = raw_test.get_montage()
    assert mnt2.get_positions()['coord_frame'] == 'unknown'
    np.testing.assert_array_almost_equal(
        np.array(list(mnt.get_positions()['ch_pos'].values())),
        np.array(list(mnt2.get_positions()['ch_pos'].values()))
    )


@pytest.mark.filterwarnings('ignore:The unit for chann*.:RuntimeWarning:mne')
def test_dig_template(tmp_path):
    """Test that eeg and ieeg dig are stored properly."""
    bids_root = tmp_path / 'bids1'
    for datatype in ('eeg', 'ieeg'):
        (bids_root / 'sub-01' / 'ses-01' / datatype).mkdir(parents=True)

    raw_test = raw.copy().pick_types(eeg=True)

    for datatype in ('eeg', 'ieeg'):
        bids_path = _bids_path.copy().update(root=bids_root, datatype=datatype)
        for coord_frame in BIDS_STANDARD_TEMPLATE_COORDINATE_FRAMES:
            bids_path.update(space=coord_frame)
            mnt = montage.copy()
            pos = mnt.get_positions()
            mne_coord_frame = BIDS_TO_MNE_FRAMES.get(coord_frame, None)
            if mne_coord_frame is None:
                mnt.apply_trans(mne.transforms.Transform('head', 'unknown'))
            else:
                mnt.apply_trans(mne.transforms.Transform(
                    'head', mne_coord_frame))
            _write_dig_bids(bids_path, raw_test, mnt, acpc_aligned=True)
            electrodes_path = bids_path.copy().update(
                task=None, run=None, suffix='electrodes', extension='.tsv')
            coordsystem_path = bids_path.copy().update(
                task=None, run=None, suffix='coordsystem', extension='.json')
            if mne_coord_frame is None:
                with pytest.warns(RuntimeWarning,
                                  match='not an MNE-Python coordinate frame'):
                    _read_dig_bids(electrodes_path, coordsystem_path,
                                   datatype, raw_test)
            else:
                if coord_frame == 'MNI305':  # saved to fsaverage, same
                    electrodes_path.update(space='fsaverage')
                    coordsystem_path.update(space='fsaverage')
                _read_dig_bids(electrodes_path, coordsystem_path,
                               datatype, raw_test)
            mnt2 = raw_test.get_montage()
            pos2 = mnt2.get_positions()
            np.testing.assert_array_almost_equal(
                np.array(list(pos['ch_pos'].values())),
                np.array(list(pos2['ch_pos'].values())))
            if mne_coord_frame is None:
                assert pos2['coord_frame'] == 'unknown'
            else:
                assert pos2['coord_frame'] == mne_coord_frame

    # test MEG
    raw_test = raw.copy()
    for coord_frame in BIDS_STANDARD_TEMPLATE_COORDINATE_FRAMES:
        bids_path = _bids_path.copy().update(root=bids_root, datatype='meg',
                                             space=coord_frame)
        write_raw_bids(raw_test, bids_path)
        raw_test2 = read_raw_bids(bids_path)
        for ch, ch2 in zip(raw.info['chs'], raw_test2.info['chs']):
            np.testing.assert_array_equal(ch['loc'], ch2['loc'])
            assert ch['coord_frame'] == ch2['coord_frame']
