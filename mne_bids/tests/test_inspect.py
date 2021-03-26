"""Test the interactive data inspector."""

import os.path as op
import pytest
from functools import partial

import numpy as np

import mne
from mne.datasets import testing
from mne.utils import requires_version
from mne.utils._testing import requires_module
from mne.viz.utils import _fake_click

from mne_bids import (BIDSPath, read_raw_bids, write_raw_bids, inspect_dataset,
                      write_meg_calibration, write_meg_crosstalk)
import mne_bids.inspect
from mne_bids.read import _from_tsv

from test_read import warning_str

requires_matplotlib = partial(requires_module, name='matplotlib',
                              call='import matplotlib')

_bids_path = BIDSPath(subject='01', session='01', run='01', task='testing',
                      datatype='meg')


def setup_bids_test_dir(bids_root):
    """Return path to a written test BIDS dir."""
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
def test_inspect_single_file(tmp_path, save_changes):
    """Test inspecting a dataset consisting of only a single file."""
    from mne.utils._testing import _click_ch_name
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.close('all')

    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
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
def test_inspect_multiple_files(tmp_path):
    """Test inspecting a dataset consisting of more than one file."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.close('all')

    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)

    # Create a second subject
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    write_raw_bids(raw, bids_path.copy().update(subject='02'))
    del raw

    # Inspection should end with the second subject.
    inspect_dataset(bids_path.copy().update(subject=None))
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    assert raw_fig.mne.info['subject_info']['his_id'] == 'sub-02'
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_set_and_unset_bads(tmp_path):
    """Test marking channels as bad and later marking them as good again."""
    from mne.utils._testing import _click_ch_name
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.close('all')

    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    orig_bads = raw.info['bads'].copy()

    # Mark some channels as bad by clicking on their name.
    inspect_dataset(bids_path, find_flat=False)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    _click_ch_name(raw_fig, ch_index=0, button=1)
    _click_ch_name(raw_fig, ch_index=1, button=1)
    _click_ch_name(raw_fig, ch_index=4, button=1)

    # Close window and save changes.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']
    fig_dialog.canvas.key_press_event('return')

    # Inspect the data again, click on two of the bad channels to mark them as
    # good.
    inspect_dataset(bids_path, find_flat=False)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    _click_ch_name(raw_fig, ch_index=1, button=1)
    _click_ch_name(raw_fig, ch_index=4, button=1)

    # Close window and save changes.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']
    fig_dialog.canvas.key_press_event('return')

    # Check marking the channels as good has actually worked.
    expected_bads = orig_bads + ['MEG 0113']
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    new_bads = raw.info['bads']
    assert set(new_bads) == set(expected_bads)


def _add_annotation(raw_fig):
    """Add an Annotation to a Raw plot."""
    data_ax = raw_fig.mne.ax_main
    raw_fig.canvas.key_press_event('a')  # Toggle Annotation mode
    ann_fig = raw_fig.mne.fig_annotation
    for key in 'test':  # Annotation will be named: BAD_test
        ann_fig.canvas.key_press_event(key)
    ann_fig.canvas.key_press_event('enter')

    # Draw a 4 second long Annotation.
    _fake_click(raw_fig, data_ax, [1., 1.], xform='data', button=1,
                kind='press')
    _fake_click(raw_fig, data_ax, [5., 1.], xform='data', button=1,
                kind='motion')
    _fake_click(raw_fig, data_ax, [5., 1.], xform='data', button=1,
                kind='release')


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_annotations(tmp_path):
    """Test inspection of Annotations."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.close('all')

    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    orig_annotations = raw.annotations.copy()

    inspect_dataset(bids_path, find_flat=False)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    _add_annotation(raw_fig)

    # Close window and save changes.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']
    fig_dialog.canvas.key_press_event('return')

    # Ensure changes were saved.
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    assert 'BAD_test' in raw.annotations.description
    annot_idx = raw.annotations.description == 'BAD_test'
    assert raw.annotations.duration[annot_idx].squeeze() == 4

    # Remove the Annotation.
    inspect_dataset(bids_path, find_flat=False)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    data_ax = raw_fig.mne.ax_main
    raw_fig.canvas.key_press_event('a')  # Toggle Annotation mode
    _fake_click(raw_fig, data_ax, [1., 1.], xform='data', button=3,
                kind='press')

    # Close window and save changes.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']
    fig_dialog.canvas.key_press_event('return')

    # Ensure changes were saved.
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    assert 'BAD_test' not in raw.annotations.description
    assert raw.annotations == orig_annotations


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_annotations_remove_all(tmp_path):
    """Test behavior if all Annotations are removed by the user."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.close('all')

    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
    events_tsv_fpath = (bids_path.copy()
                        .update(suffix='events', extension='.tsv')
                        .fpath)

    # Remove all Annotations.
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    raw.set_annotations(None)
    raw.load_data()
    raw.save(raw.filenames[0], overwrite=True)
    # Delete events.tsv sidecar.
    (bids_path.copy()
     .update(suffix='events', extension='.tsv')
     .fpath
     .unlink())

    # Add custom Annotation.
    inspect_dataset(bids_path, find_flat=False)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    _add_annotation(raw_fig)

    # Close window and save changes.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']
    fig_dialog.canvas.key_press_event('return')

    # events.tsv sidecar should have been created.
    assert events_tsv_fpath.exists()

    # Remove the Annotation.
    inspect_dataset(bids_path, find_flat=False)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    data_ax = raw_fig.mne.ax_main
    raw_fig.canvas.key_press_event('a')  # Toggle Annotation mode
    _fake_click(raw_fig, data_ax, [1., 1.], xform='data', button=3,
                kind='press')

    # Close window and save changes.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']
    fig_dialog.canvas.key_press_event('return')

    # events.tsv sidecar should not exist anymore.
    assert not events_tsv_fpath.exists()


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_dont_show_annotations(tmp_path):
    """Test if show_annotations=False works."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.close('all')

    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
    inspect_dataset(bids_path, find_flat=False, show_annotations=False)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']
    assert not raw_fig.mne.annotations


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_bads_and_annotations(tmp_path):
    """Test adding bads and Annotations in one go."""
    from mne.utils._testing import _click_ch_name
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.close('all')

    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    orig_bads = raw.info['bads'].copy()

    inspect_dataset(bids_path, find_flat=False)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']

    # Mark some channels as bad by clicking on their name.
    _click_ch_name(raw_fig, ch_index=0, button=1)

    # Add custom Annotation.
    _add_annotation(raw_fig)

    # Close window and save changes.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']
    fig_dialog.canvas.key_press_event('return')

    # Check that the changes were saved.
    raw = read_raw_bids(bids_path=bids_path, verbose='error')
    new_bads = raw.info['bads']
    expected_bads = orig_bads + ['MEG 0113']
    assert set(new_bads) == set(expected_bads)
    assert 'BAD_test' in raw.annotations.description


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.parametrize('save_changes', (True, False))
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_auto_flats(tmp_path, save_changes):
    """Test flat channel & segment detection."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.close('all')

    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
    channels_tsv_fname = bids_path.copy().update(suffix='channels',
                                                 extension='.tsv')

    raw = read_raw_bids(bids_path=bids_path, verbose='error')

    # Inject an entirely flat channel.
    raw.load_data()
    raw._data[10] = np.zeros_like(raw._data[10], dtype=raw._data.dtype)
    # Add a a flat time segment (approx. 100 ms) to another channel
    raw._data[20, 500:500 + int(np.ceil(0.1 * raw.info['sfreq']))] = 0
    raw.save(raw.filenames[0], overwrite=True)
    old_bads = raw.info['bads'].copy()

    inspect_dataset(bids_path)
    raw_fig = mne_bids.inspect._global_vars['raw_fig']

    # Closing the window should open a dialog box.
    raw_fig.canvas.key_press_event(raw_fig.mne.close_key)
    fig_dialog = mne_bids.inspect._global_vars['dialog_fig']

    if save_changes:
        key = 'return'
    else:
        key = 'escape'
    fig_dialog.canvas.key_press_event(key)

    raw = read_raw_bids(bids_path=bids_path, verbose='error')

    if save_changes:
        assert old_bads != raw.info['bads']
        assert raw.ch_names[10] in raw.info['bads']
        channels_tsv_data = _from_tsv(channels_tsv_fname)
        assert (channels_tsv_data['status_description'][10] ==
                'Flat channel, auto-detected via MNE-BIDS')
        # This channel should not have been added to `bads`, but produced a
        # flat annotation.
        assert raw.ch_names[20] not in raw.info['bads']
        assert 'BAD_flat' in raw.annotations.description
    else:
        assert old_bads == raw.info['bads']
        assert 'BAD_flat' not in raw.annotations.description


@requires_matplotlib
@requires_version('mne', '0.22')
@pytest.mark.parametrize(('l_freq', 'h_freq'),
                         [(None, None),
                          (1, None),
                          (None, 30),
                          (1, 30)])
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_inspect_freq_filter(tmp_path, l_freq, h_freq):
    """Test frequency filter for Raw display."""
    bids_root = setup_bids_test_dir(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
    inspect_dataset(bids_path, l_freq=l_freq, h_freq=h_freq, find_flat=False)
