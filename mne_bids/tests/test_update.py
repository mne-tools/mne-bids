"""Test for the MNE BIDS updating of BIDS datasets."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import json
import os.path as op

import mne
import pytest
from mne.datasets import testing

from mne_bids import (BIDSPath, write_raw_bids,
                      write_meg_calibration, write_meg_crosstalk)
from mne_bids.path import _mkdir_p
from mne_bids.sidecar_updates import update_sidecar_json
from mne_bids.utils import _write_json

subject_id = '01'
session_id = '01'
run = '01'
acq = None
task = 'testing'

bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)


@pytest.fixture(scope='session')
def _get_bids_test_dir(tmpdir_factory):
    """Return path to a written test BIDS dir."""
    bids_root = str(tmpdir_factory.mktemp('mnebids_utils_test_bids_ds'))
    data_path = testing.data_path()
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
        write_raw_bids(raw, name, events_data=events,
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


@pytest.mark.usefixtures('_get_bids_test_dir', '_bids_validate',
                         '_get_sidecar_json_update_file')
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
    sidecar_path = bids_path.copy().update(extension='.json')
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
