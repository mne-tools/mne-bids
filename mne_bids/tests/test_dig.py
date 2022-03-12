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
from numpy.testing import assert_almost_equal
import warnings

import mne
import mne_bids
from mne.datasets import testing
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids
from mne_bids.dig import _write_dig_bids, _read_dig_bids, template_to_head
from mne_bids.config import (BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS,
                             BIDS_TO_MNE_FRAMES, MNE_STR_TO_FRAME)

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
    assert_almost_equal(
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
        for coord_frame in BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS:
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
    for coord_frame in BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS:
        bids_path = _bids_path.copy().update(root=bids_root, datatype='meg',
                                             space=coord_frame)
        write_raw_bids(raw_test, bids_path)
        raw_test2 = read_raw_bids(bids_path)
        for ch, ch2 in zip(raw.info['chs'], raw_test2.info['chs']):
            np.testing.assert_array_equal(ch['loc'], ch2['loc'])
            assert ch['coord_frame'] == ch2['coord_frame']


def _set_montage_no_trans(raw, montage):
    """Set the montage without transforming to 'head'."""
    coord_frame = montage.get_positions()['coord_frame']
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                message='.*nasion not found', module='mne')
        raw.set_montage(montage, on_missing='ignore')
    for ch in raw.info['chs']:
        ch['coord_frame'] = MNE_STR_TO_FRAME[coord_frame]
    for d in raw.info['dig']:
        d['coord_frame'] = MNE_STR_TO_FRAME[coord_frame]


def _test_montage_trans(raw, montage, pos_test, space='fsaverage',
                        coord_frame='auto', unit='auto'):
    """Test if a montage is transformed correctly."""
    _set_montage_no_trans(raw, montage)
    trans = template_to_head(
        raw.info, space, coord_frame=coord_frame, unit=unit)[1]
    montage_test = raw.get_montage()
    montage_test.apply_trans(trans)
    assert_almost_equal(
        pos_test,
        np.array(list(montage_test.get_positions()['ch_pos'].values())))


def test_template_to_head():
    """Test transforming a template montage to head."""
    # test no montage
    raw_test = raw.copy()
    raw_test.set_montage(None)
    with pytest.raises(RuntimeError, match='No montage found'):
        template_to_head(raw_test.info, 'fsaverage', coord_frame='auto')

    # test no channels
    montage_empty = mne.channels.make_dig_montage(hsp=[[0, 0, 0]])
    _set_montage_no_trans(raw_test, montage_empty)
    with pytest.raises(RuntimeError, match='No channel locations '
                                           'found in the montage'):
        template_to_head(raw_test.info, 'fsaverage', coord_frame='auto')

    # test unexpected coordinate frame
    raw_test = raw.copy()
    with pytest.raises(RuntimeError, match='not expected for a template'):
        template_to_head(raw_test.info, 'fsaverage', coord_frame='auto')

    # test all coordinate frames
    raw_test = raw.copy()
    raw_test.set_montage(None)
    raw_test.pick_types(eeg=True)
    raw_test.drop_channels(raw_test.ch_names[3:])
    montage = mne.channels.make_dig_montage(
        ch_pos={raw_test.ch_names[0]: [0, 0, 0],
                raw_test.ch_names[1]: [0, 0, 0.1],
                raw_test.ch_names[2]: [0, 0, 0.2]},
        coord_frame='unknown')
    for space in BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS:
        for cf in ('mri', 'mri_voxel', 'ras'):
            _set_montage_no_trans(raw_test, montage)
            trans = template_to_head(raw_test.info, space, cf)[1]
            assert trans['from'] == MNE_STR_TO_FRAME['head']
            assert trans['to'] == MNE_STR_TO_FRAME['mri']
            montage_test = raw_test.get_montage()
            pos = montage_test.get_positions()
            assert pos['coord_frame'] == 'head'
            assert pos['nasion'] is not None
            assert pos['lpa'] is not None
            assert pos['rpa'] is not None

    # test that we get the right transform
    _set_montage_no_trans(raw_test, montage)
    trans = template_to_head(raw_test.info, 'fsaverage', 'mri')[1]
    trans2 = mne.read_trans(op.join(
        op.dirname(op.dirname(mne_bids.__file__)), 'mne_bids', 'data',
        'space-fsaverage_trans.fif'))
    assert_almost_equal(trans['trans'], trans2['trans'])

    # test auto coordinate frame

    # test auto voxels
    montage_vox = mne.channels.make_dig_montage(
        ch_pos={raw_test.ch_names[0]: [2, 0, 10],
                raw_test.ch_names[1]: [0, 0, 5.5],
                raw_test.ch_names[2]: [0, 1, 3]},
        coord_frame='unknown')
    pos_test = np.array([[0.126, -0.118, 0.128],
                         [0.128, -0.1225, 0.128],
                         [0.128, -0.125, 0.127]])
    _test_montage_trans(raw_test, montage_vox, pos_test,
                        coord_frame='auto', unit='mm')

    # now negative values => scanner RAS
    montage_ras = mne.channels.make_dig_montage(
        ch_pos={raw_test.ch_names[0]: [-30.2, 20, -40],
                raw_test.ch_names[1]: [10, 30, 53.5],
                raw_test.ch_names[2]: [30, -21, 33]},
        coord_frame='unknown')
    pos_test = np.array([[-0.0302, 0.02, -0.04],
                         [0.01, 0.03, 0.0535],
                         [0.03, -0.021, 0.033]])
    _set_montage_no_trans(raw_test, montage_ras)
    _test_montage_trans(raw_test, montage_ras, pos_test,
                        coord_frame='auto', unit='mm')

    # test auto unit
    montage_mm = montage_ras.copy()
    _set_montage_no_trans(raw_test, montage_mm)
    _test_montage_trans(raw_test, montage_mm, pos_test,
                        coord_frame='ras', unit='auto')

    montage_m = montage_ras.copy()
    for d in montage_m.dig:
        d['r'] = np.array(d['r']) / 1000
    _test_montage_trans(raw_test, montage_m, pos_test,
                        coord_frame='ras', unit='auto')
