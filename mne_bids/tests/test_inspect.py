"""Test the interactive data inspector."""

import os.path as op
import pytest
import matplotlib

import mne
from mne.datasets import testing
from mne.utils import requires_version
from mne.utils._testing import _click_ch_name

from mne_bids import BIDSPath, read_raw_bids, write_raw_bids, inspect_bids
import mne_bids.inspect

from test_read import warning_str

matplotlib.use('Agg')


subject_id = '01'
session_id = '01'
run = '01'
task = 'testing'
datatype = 'meg'

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, task=task,
    datatype=datatype)


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


@requires_version('mne', '0.22')
@pytest.mark.parametrize('save_changes', (True, False))
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect(return_bids_test_dir, save_changes):
    bids_path = _bids_path.copy().update(root=return_bids_test_dir)
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    old_bads = raw.info['bads'].copy()

    fig = inspect_bids(bids_path, block=False)

    # Mark some channels as bad by clicking on their name.
    _click_ch_name(fig, ch_index=0, button=1)
    _click_ch_name(fig, ch_index=1, button=1)
    _click_ch_name(fig, ch_index=4, button=1)

    # Closing the window should open a dialog box.
    fig.canvas.key_press_event(fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']

    if save_changes:
        key = 'return'
    else:
        key = 'escape'
    fig_dialog.canvas.key_press_event(key)

    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    new_bads = raw.info['bads'].copy()

    if save_changes:
        assert len(new_bads) > len(old_bads)
    else:
        assert old_bads == new_bads
