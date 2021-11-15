"""Inspect and annotate BIDS raw data."""
# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np

import mne
from mne.preprocessing import annotate_flat
from mne.utils import logger, verbose

from mne_bids import read_raw_bids, mark_channels
from mne_bids.read import _from_tsv, _read_events
from mne_bids.write import _events_tsv
from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS


@verbose
def inspect_dataset(bids_path, find_flat=True, l_freq=None, h_freq=None,
                    show_annotations=True, verbose=None):
    """Inspect and annotate BIDS raw data.

    This function allows you to browse MEG, EEG, and iEEG raw data stored in a
    BIDS dataset. You can toggle the status of a channel (bad or good) by
    clicking on the traces, and when closing the browse window, you will be
    asked whether you want to save the changes to the existing BIDS dataset or
    discard them.

    .. warning:: This functionality is still experimental and will be extended
                 in the future. Its API will likely change. Planned features
                 include automated bad channel detection and visualization of
                 MRI images.

    .. note:: Currently, only MEG, EEG, and iEEG data can be inspected.

    To add or modify annotations, press ``A`` to toggle annotation mode.

    Parameters
    ----------
    bids_path : BIDSPath
        A :class:`mne_bids.BIDSPath` containing at least a ``root``. All
        matching files will be inspected. To select only a subset of the data,
        set more :class:`mne_bids.BIDSPath` attributes. If ``datatype`` is not
        set and multiple datatypes are found, they will be inspected in the
        following order: MEG, EEG, iEEG.
        To read a specific file, set all the :class:`mne_bids.BIDSPath`
        attributes required to uniquely identify the file: If this ``BIDSPath``
        is accepted by :func:`mne_bids.read_raw_bids`, it will work here.
    find_flat : bool
        Whether to auto-detect channels producing "flat" signals, i.e., with
        unusually low variability. Flat **segments** will be added to
        ``*_events.tsv``, while channels with more than 5 percent of flat data
        will be marked as ``bad`` in ``*_channels.tsv``.

        .. note::
            This function calls :func:`mne.preprocessing.annotate_flat` and
            will only consider segments of at least **50 ms consecutive
            flatness** as "flat" (deviating from MNE-Python's default of 5 ms).
            If more than 5 percent of a channel's data has been marked as flat,
            the entire channel will be added to the list of bad channels. Only
            flat time segments applying to channels **not** marked as bad will
            be added to ``*_events.tsv``.

    l_freq : float | None
        The high-pass filter cutoff frequency to apply when displaying the
        data. This can be useful when inspecting data with slow drifts. If
        ``None``, no high-pass filter will be applied.
    h_freq : float | None
        The low-pass filter cutoff frequency to apply when displaying the
        data. This can be useful when inspecting data with high-frequency
        artifacts. If ``None``, no low-pass filter will be applied.
    show_annotations : bool
        Whether to show annotations (events, bad segments, …) or not. If
        ``False``, toggling annotations mode by pressing ``A`` will be disabled
        as well.
    %(verbose)s

    Examples
    --------
    Disable flat channel & segment detection, and apply a filter with a
    passband of 1–30 Hz.

    >>> from mne_bids import BIDSPath
    >>> root = Path('./mne_bids/tests/data/tiny_bids').absolute()
    >>> bids_path = BIDSPath(subject='01', task='rest', session='eeg',
    ...                      suffix='eeg', extension='.vhdr', root=root)
    >>> inspect_dataset(bids_path=bids_path, find_flat=False,  # doctest: +SKIP
    ...                 l_freq=1, h_freq=30)
    """
    allowed_extensions = set(ALLOWED_DATATYPE_EXTENSIONS['meg'] +
                             ALLOWED_DATATYPE_EXTENSIONS['eeg'] +
                             ALLOWED_DATATYPE_EXTENSIONS['ieeg'])

    bids_paths = [p for p in bids_path.match(check=True)
                  if (p.extension is None or
                      p.extension in allowed_extensions) and
                  p.acquisition != 'crosstalk']

    for bids_path_ in bids_paths:
        _inspect_raw(bids_path=bids_path_, l_freq=l_freq, h_freq=h_freq,
                     find_flat=find_flat, show_annotations=show_annotations)


# XXX This this should probably be refactored into a class attribute someday.
_global_vars = dict(raw_fig=None,
                    dialog_fig=None,
                    mne_close_key=None)


def _inspect_raw(*, bids_path, l_freq, h_freq, find_flat, show_annotations):
    """Raw data inspection."""
    # Delay the import
    import matplotlib
    import matplotlib.pyplot as plt

    extra_params = dict()
    if bids_path.extension == '.fif':
        extra_params['allow_maxshield'] = True
    raw = read_raw_bids(bids_path, extra_params=extra_params, verbose='error')
    old_bads = raw.info['bads'].copy()
    old_annotations = raw.annotations.copy()

    if find_flat:
        raw.load_data()  # Speeds up processing dramatically
        flat_annot, flat_chans = annotate_flat(raw=raw, min_duration=0.05)
        new_annot = raw.annotations + flat_annot
        raw.set_annotations(new_annot)
        raw.info['bads'] = list(set(raw.info['bads'] + flat_chans))
        del new_annot, flat_annot
    else:
        flat_chans = []

    show_options = bids_path.datatype == 'meg'
    fig = raw.plot(title=f'{bids_path.root.name}: {bids_path.basename}',
                   highpass=l_freq, lowpass=h_freq,
                   show_options=show_options,
                   block=False, show=False, verbose='warning')

    # Add our own event handlers so that when the MNE Raw Browser is being
    # closed, our dialog box will pop up, asking whether to save changes.
    def _handle_close(event):
        mne_raw_fig = event.canvas.figure
        # Bads alterations are only transferred to `inst` once the figure is
        # closed; Annotation changes are immediately reflected in `inst`
        new_bads = mne_raw_fig.mne.info['bads'].copy()
        new_annotations = mne_raw_fig.mne.inst.annotations.copy()

        if not new_annotations:
            # Ensure it's not an empty list, but an empty set of Annotations.
            new_annotations = mne.Annotations(
                onset=[], duration=[], description=[],
                orig_time=mne_raw_fig.mne.info['meas_date']
            )
        _save_raw_if_changed(old_bads=old_bads,
                             new_bads=new_bads,
                             flat_chans=flat_chans,
                             old_annotations=old_annotations,
                             new_annotations=new_annotations,
                             bids_path=bids_path)
        _global_vars['raw_fig'] = None

    def _keypress_callback(event):
        if event.key == _global_vars['mne_close_key']:
            _handle_close(event)

    fig.canvas.mpl_connect('close_event', _handle_close)
    fig.canvas.mpl_connect('key_press_event', _keypress_callback)

    if not show_annotations:
        # Remove annotations and kill `_toggle_annotation_fig` method, since
        # we cannot directly and easily remove the associated `a` keyboard
        # event callback.
        fig._clear_annotations()
        fig._toggle_annotation_fig = lambda: None
        # Ensure it's not an empty list, but an empty set of Annotations.
        old_annotations = mne.Annotations(
            onset=[], duration=[], description=[],
            orig_time=raw.info['meas_date']
        )

    if matplotlib.get_backend() != 'agg':
        plt.show(block=True)

    _global_vars['raw_fig'] = fig
    _global_vars['mne_close_key'] = fig.mne.close_key


def _annotations_almost_equal(old_annotations, new_annotations):
    """Allow for a tiny bit of floating point precision loss."""
    if (np.array_equal(old_annotations.description,
                       new_annotations.description) and
        np.array_equal(old_annotations.orig_time,
                       new_annotations.orig_time) and
        np.allclose(old_annotations.onset,
                    new_annotations.onset) and
        np.allclose(old_annotations.duration,
                    new_annotations.duration)):
        return True
    else:
        return False


def _save_annotations(*, annotations, bids_path):
    # Attach the new Annotations to our raw data so we can easily convert them
    # to events, which will be stored in the *_events.tsv sidecar.
    extra_params = dict()
    if bids_path.extension == '.fif':
        extra_params['allow_maxshield'] = True

    raw = read_raw_bids(bids_path=bids_path, extra_params=extra_params,
                        verbose='warning')
    raw.set_annotations(annotations)
    events, durs, descrs = _read_events(events_data=None, event_id=None,
                                        bids_path=bids_path, raw=raw)

    # Write sidecar – or remove it if no events are left.
    events_tsv_fname = (bids_path.copy()
                        .update(suffix='events',
                                extension='.tsv')
                        .fpath)

    if len(events) > 0:
        _events_tsv(events=events, durations=durs, raw=raw,
                    fname=events_tsv_fname, trial_type=descrs,
                    overwrite=True)
    elif events_tsv_fname.exists():
        logger.info(f'No events remaining after interactive inspection, '
                    f'removing {events_tsv_fname.name}')
        events_tsv_fname.unlink()


def _save_raw_if_changed(*, old_bads, new_bads, flat_chans,
                         old_annotations, new_annotations,
                         bids_path):
    """Save bad channel selection if it has been changed.

    Parameters
    ----------
    old_bads : list
        The original bad channels.
    new_bads : list
        The updated set of bad channels (i.e. **all** of them, not only the
        changed ones).
    flat_chans : list
        The auto-detected flat channels. This is either an empty list or a
        subset of ``new_bads``.
    old_annotations : mne.Annotations
        The original Annotations.
    new_annotations : mne.Annotations
        The new Annotations.
    """
    assert set(flat_chans).issubset(set(new_bads))

    if set(old_bads) == set(new_bads):
        bads = None
        bad_descriptions = []
    else:
        bads = new_bads
        bad_descriptions = []

        # Generate entries for the `status_description` column.
        channels_tsv_fname = (bids_path.copy()
                              .update(suffix='channels', extension='.tsv')
                              .fpath)
        channels_tsv_data = _from_tsv(channels_tsv_fname)

        for ch_name in bads:
            idx = channels_tsv_data['name'].index(ch_name)
            if channels_tsv_data['status'][idx] == 'bad':
                # Channel was already marked as bad in the data, so retain
                # existing description.
                description = channels_tsv_data['status_description'][idx]
            elif ch_name in flat_chans:
                description = 'Flat channel, auto-detected via MNE-BIDS'
            else:
                # Channel has been manually marked as bad during inspection
                description = 'Interactive inspection via MNE-BIDS'

            bad_descriptions.append(description)
            del ch_name, description

        del channels_tsv_data, channels_tsv_fname,

    if _annotations_almost_equal(old_annotations, new_annotations):
        annotations = None
    else:
        annotations = new_annotations

    if bads is None and annotations is None:
        # Nothing has changed, so we can just exit.
        return None

    return _save_raw_dialog_box(bads=bads,
                                bad_descriptions=bad_descriptions,
                                annotations=annotations,
                                bids_path=bids_path)


def _save_raw_dialog_box(*, bads, bad_descriptions, annotations, bids_path):
    """Display a dialog box asking whether to save the changes."""
    # Delay the imports
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from mne.viz.utils import figure_nobar

    title = 'Save changes?'
    message = 'You have modified '
    if bads is not None and annotations is None:
        message += 'the bad channel selection '
        figsize = (7.5, 2.5)
    elif bads is None and annotations is not None:
        message += 'the bad segments selection '
        figsize = (7.5, 2.5)
    else:
        message += 'the bad channel and\nannotations selection '
        figsize = (8.5, 3)

    message += (f'of\n'
                f'{bids_path.basename}.\n\n'
                f'Would you like to save these changes to the\n'
                f'BIDS dataset?')
    icon_fname = str(Path(__file__).parent / 'assets' / 'help-128px.png')
    icon = plt.imread(icon_fname)

    fig = figure_nobar(figsize=figsize)
    fig.canvas.manager.set_window_title('MNE-BIDS Inspector')
    fig.suptitle(title, y=0.95, fontsize='xx-large', fontweight='bold')

    gs = fig.add_gridspec(1, 2, width_ratios=(1.5, 5))

    # The dialog box tet.
    ax_msg = fig.add_subplot(gs[0, 1])
    ax_msg.text(x=0, y=0.8, s=message, fontsize='large',
                verticalalignment='top', horizontalalignment='left',
                multialignment='left')
    ax_msg.axis('off')

    # The help icon.
    ax_icon = fig.add_subplot(gs[0, 0])
    ax_icon.imshow(icon)
    ax_icon.axis('off')

    # Buttons.
    ax_save = fig.add_axes([0.6, 0.05, 0.3, 0.1])
    ax_dont_save = fig.add_axes([0.1, 0.05, 0.3, 0.1])

    save_button = Button(ax=ax_save, label='Save')
    save_button.label.set_fontsize('medium')
    save_button.label.set_fontweight('bold')

    dont_save_button = Button(ax=ax_dont_save, label="Don't save")
    dont_save_button.label.set_fontsize('medium')
    dont_save_button.label.set_fontweight('bold')

    # Store references to keep buttons alive.
    fig.save_button = save_button
    fig.dont_save_button = dont_save_button

    # Define callback functions.
    def _save_callback(event):
        plt.close(event.canvas.figure)  # Close dialog
        _global_vars['dialog_fig'] = None

        if bads is not None:
            _save_bads(bads=bads, descriptions=bad_descriptions,
                       bids_path=bids_path)
        if annotations is not None:
            _save_annotations(annotations=annotations, bids_path=bids_path)

    def _dont_save_callback(event):
        plt.close(event.canvas.figure)  # Close dialog
        _global_vars['dialog_fig'] = None

    def _keypress_callback(event):
        if event.key in ['enter', 'return']:
            _save_callback(event)
        elif event.key == _global_vars['mne_close_key']:
            _dont_save_callback(event)

    # Connect events to callback functions.
    save_button.on_clicked(_save_callback)
    dont_save_button.on_clicked(_dont_save_callback)
    fig.canvas.mpl_connect('close_event', _dont_save_callback)
    fig.canvas.mpl_connect('key_press_event', _keypress_callback)

    if matplotlib.get_backend() != 'agg':
        fig.show()

    _global_vars['dialog_fig'] = fig


def _save_bads(*, bads, descriptions, bids_path):
    """Update the set of channels marked as bad.

    Parameters
    ----------
    bads : list
        The complete list of bad channels.
    descriptions : list
        The values to be written to the `status_description` column.
    """
    # We first make all channels not passed as bad here to be marked as good.
    mark_channels(bids_path=bids_path, ch_names=[], status='good')
    mark_channels(bids_path=bids_path, ch_names=bads, status='bad',
                  descriptions=descriptions)
