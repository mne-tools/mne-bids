from pathlib import Path

from mne_bids import read_raw_bids, mark_bad_channels
from mne_bids.read import _from_tsv


def inspect_bids(bids_path, block=True, verbose=None):
    """Inspect and annotate BIDS raw data.

    This function allows you to browse MEG and EEG raw data stored in a BIDS
    dataset. You can toggle the status of a channel (bad or good) by clicking
    on the traces, and when closing the browse window, you will be asked
    whether you want to save the changes to the existing BIDS dataset or
    discard them.

    .. warning:: This functionality is still experimental and will be extended
                 in the future. Its API will likely change. Planned features
                 include modification of annotations and automated bad channel
                 detection.

    Parameters
    ----------
    bids_path : BIDSPath
        A :class:`mne_bids.BIDSPath` containing enough information to uniquely
        identify the raw data file. If this ``BIDSPath`` is accepted by
        :func:`mne_bids.read_raw_bids`, it will work here.
    block : bool
        Whether to halt further code execution until the inspection window has
        been closed.
    verbose : bool | None
        If a boolean, whether or not to produce verbose output. If ``None``,
        use the default log level.
    """

    fig = _inspect_raw(bids_path=bids_path, block=block, verbose=verbose)
    return fig


def _inspect_raw(*, bids_path, block, verbose=None):
    """Raw data inspection."""

    # Delay the import
    import matplotlib.pyplot as plt

    raw = read_raw_bids(bids_path, extra_params=dict(allow_maxshield=True),
                        verbose='error')
    fig = raw.plot(title=f'{bids_path.root.name}: {bids_path.basename}',
                   show_options=True, block=False, show=False,
                   verbose='warning')

    def _handle_close(event):
        mne_raw_fig = event.canvas.figure
        old_bads = mne_raw_fig.mne.inst.info['bads'].copy()
        new_bads = mne_raw_fig.mne.info['bads'].copy()
        mne_raw_fig._close(event)

        _save_bads_if_changed(old_bads=old_bads,
                              new_bads=new_bads,
                              bids_path=bids_path,
                              verbose=verbose)

    # Deactivate Raw Browser's default "close event" handler – we will take
    # care of running the _close() method in our own event handler.
    fig.canvas.mpl_disconnect(list(fig.canvas.
                                   callbacks.
                                   callbacks['close_event']
                                   .keys())[0])
    fig.canvas.mpl_connect('close_event', _handle_close)

    plt.show(block=block)
    return fig


def _save_bads_if_changed(*, old_bads, new_bads, bids_path, verbose):
    if set(old_bads) == set(new_bads):
        # Nothing has changed, so we can just exit.
        return

    _save_bads_dialog_box(bads=new_bads, bids_path=bids_path, verbose=verbose)


def _save_bads_dialog_box(*, bads, bids_path, verbose):
    """Display a dialog box asking whether to save the changes."""

    # # Delay the imports
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from mne.viz.utils import figure_nobar

    title = 'Save changes?'
    message = (f'You have modified the bad channel selection of\n'
               f'{bids_path.basename}.\n\nWould you like to save these '
               f'changes to the\nBIDS dataset?')
    icon_fname = Path(__file__).parent / 'assets' / 'help-128px.png'
    icon = plt.imread(icon_fname)

    fig = figure_nobar(figsize=(7.5, 2))
    fig.canvas.set_window_title('MNE-BIDS Inspector')
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

    # Define callback functions.
    def _save_callback(event):
        plt.close(event.canvas.figure)  # Close dialog
        _save_bads(bads=bads, bids_path=bids_path, verbose=verbose)

    def _dont_save_callback(event):
        plt.close(event.canvas.figure)  # Close dialog

    def _keypress_callback(event):
        if event.key == 'escape':
            _dont_save_callback(event)

    # Connect events to callback functions.
    save_button.on_clicked(_save_callback)
    dont_save_button.on_clicked(_dont_save_callback)
    fig.canvas.mpl_connect('close_event', _dont_save_callback)
    fig.canvas.mpl_connect('key_press_event', _keypress_callback)

    # plt.show(block=True)
    fig.show()


def _save_bads(*, bads, bids_path, verbose):
    """Update the set of channels marked as bad."""
    channels_tsv_fname = bids_path.copy().update(suffix='channels',
                                                 extension='.tsv')
    channels_tsv_data = _from_tsv(channels_tsv_fname)

    descriptions = []
    for ch_name in bads:
        idx = channels_tsv_data['name'].index(ch_name)
        if channels_tsv_data['status'][idx] == 'bad':
            # Channel was already marked as bad, retain existing description.
            description = channels_tsv_data['status_description'][idx]
        else:
            # Channel has been manually marked as bad during inspection, assign
            # default description.
            description = 'Manual inspection via MNE-BIDS'

        descriptions.append(description)

    # We pass overwrite=True, causing all channels not passed as bad here to
    # be marked as good.
    mark_bad_channels(ch_names=bads, descriptions=descriptions,
                      bids_path=bids_path, overwrite=True, verbose=verbose)
