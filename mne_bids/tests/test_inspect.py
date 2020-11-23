"""Test the interactive data inspector."""

import os.path as op
import pytest

import mne
from mne.datasets import testing

from mne_bids import BIDSPath, write_raw_bids, inspect_bids

from test_read import _read_raw_fif, warning_str


subject_id = '01'
session_id = '01'
run = '01'
acq = None
task = 'testing'

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)


@pytest.fixture(scope='session')
def return_bids_test_dir(tmpdir_factory):
    """Return path to a written test BIDS dir."""
    bids_root = str(tmpdir_factory.mktemp('mnebids_utils_test_bids_ds'))
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw = mne.io.read_raw_fif(raw_fname)
    raw.info['line_freq'] = 60

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    bids_path = _bids_path.copy().update(root=bids_root)
    write_raw_bids(raw, bids_path=bids_path, events_data=events,
                   event_id=event_id, overwrite=True)

    return bids_root


@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect(return_bids_test_dir):
    bids_path = _bids_path.copy().update(root=return_bids_test_dir)
    inspect_bids(bids_path, block=False)
