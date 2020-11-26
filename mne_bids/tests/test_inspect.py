"""Test the interactive data inspector."""

import os.path as op
import pytest
from functools import partial

import mne
from mne.datasets import testing
from mne.utils import requires_version
from mne.utils._testing import requires_module

from mne_bids import (BIDSPath, read_raw_bids, write_raw_bids, inspect_dataset,
                      write_meg_calibration, write_meg_crosstalk)
import mne_bids.inspect

from test_read import warning_str

requires_matplotlib = partial(requires_module, name='matplotlib',
                              call='import matplotlib')

_bids_path = BIDSPath(subject='01', session='01', run='01', task='testing',
                      datatype='meg')


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
    cal_fname = op.join(data_path, 'SSS', 'sss_cal_mgh.dat')
    crosstalk_fname = op.join(data_path, 'SSS', 'ct_sparse.fif')

    raw = mne.io.read_raw_fif(raw_fname)
    raw.info['line_freq'] = 60

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    bids_path = _bids_path.copy().update(root=bids_root)
    write_raw_bids(raw, bids_path=bids_path, events_data=events,
                   event_id=event_id, overwrite=True)
    write_meg_calibration(cal_fname, bids_path=bids_path)
    write_meg_crosstalk(crosstalk_fname, bids_path=bids_path)

    return bids_root


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.parametrize('save_changes', (True, False))
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_single_file(return_bids_test_dir, save_changes):
    from mne.utils._testing import _click_ch_name
    import matplotlib
    matplotlib.use('Agg')

    bids_path = _bids_path.copy().update(root=return_bids_test_dir)
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    old_bads = raw.info['bads'].copy()

    inspect_dataset(bids_path)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']

    # Mark some channels as bad by clicking on their name.
    _click_ch_name(raw_fig, ch_index=0, button=1)
    _click_ch_name(raw_fig, ch_index=1, button=1)
    _click_ch_name(raw_fig, ch_index=4, button=1)

    # Closing the window should open a dialog box.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
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


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_multiple_files(return_bids_test_dir):
    import matplotlib
    matplotlib.use('Agg')

    bids_path = _bids_path.copy().update(root=return_bids_test_dir)

    # Create a second subject
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    write_raw_bids(raw, bids_path.copy().update(subject='02'))
    del raw

    # Inspection should end with the second subject.
    inspect_dataset(bids_path.copy().update(subject=None))
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    assert raw_fig.mne.info['subject_info']['participant_id'] == 'sub-02'
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
