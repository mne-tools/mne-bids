"""
=========================================================
03. Interactive data inspection and bad channel selection
=========================================================

You can use MNE-BIDS interactively inspect your  MEG or (i)EEG data.
Problematic channels can be marked as "bad", for example if the connected
sensor produced mostly noise – or no signal at all. Similarly, you can declare
channels as "good", should you discover they were incorrectly marked as bad.
Bad channel selection can also be performed non-interactively.

Furthermore, you can view and edit the experimental events and mark time
segments as "bad".

.. _MNE-Python Annotations tutorial: https://mne.tools/stable/auto_tutorials/raw/30_annotate_raw.html#annotating-raw-objects-interactively
"""  # noqa:E501

# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD-3-Clause

# %%
# We will demonstrate how to mark individual channels as bad on the MNE
# "sample" dataset. After that, we will mark channels as good again.
#
# Let's start by importing the required modules and functions, reading the
# "sample" data, and writing it in the BIDS format.

import os.path as op
import shutil

import mne
from mne_bids import (BIDSPath, write_raw_bids, read_raw_bids,
                      inspect_dataset, mark_channels)

data_path = mne.datasets.sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
bids_root = op.join(data_path, '..', 'MNE-sample-data-bids')
bids_path = BIDSPath(subject='01', session='01', task='audiovisual', run='01',
                     root=bids_root)

# %%
# To ensure the output path doesn't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if op.exists(bids_root):
    shutil.rmtree(bids_root)

# %%
# Now write the raw data to BIDS.

raw = mne.io.read_raw_fif(raw_fname, verbose=False)
raw.info['line_freq'] = 60  # Specify power line frequency as required by BIDS.
write_raw_bids(raw, bids_path=bids_path, events_data=events_fname,
               event_id=event_id, overwrite=True, verbose=False)

# %%
# Interactive use
# ---------------
#
# Using :func:`mne_bids.inspect_dataset`, we can interactively explore the raw
# data and toggle the channel status – ``bad`` or ``good`` – by clicking on the
# respective traces or channel names. If there are any SSP projectors stored
# with the data, a small popup window will allow you to toggle the projectors
# on and off. If you changed the selection of bad channels, you will be
# prompted whether you would like to save the changes when closing the main
# window. Your raw data and the `*_channels.tsv` sidecar file will be updated
# upon saving.

inspect_dataset(bids_path)

# %%
# You can even apply frequency filters when viewing the data: A high-pass
# filter can remove slow drifts, while a low-pass filter will get rid of
# high-frequency artifacts. This can make visual inspection easier. Let's
# apply filters with a 1-Hz high-pass cutoff, and a 30-Hz low-pass cutoff:

inspect_dataset(bids_path, l_freq=1., h_freq=30.)

# %%
# By pressing the ``A`` key, you can toggle annotation mode to add, edit, or
# remove experimental events, or to mark entire time periods as bad. Please see
# the `MNE-Python Annotations tutorial`_ for an introduction to the interactive
# interface. If you're closing the main window after changing the annotations,
# you will be prompted whether you wish to save the changes. Your raw data and
# the `*_events.tsv` sidecar file will be updated upon saving.
#
# Non-interactive (programmatic) bad channel selection
# ----------------------------------------------------
#
# Read the (now BIDS-formatted) data and print a list of channels currently
# marked as bad.

raw = read_raw_bids(bids_path=bids_path, verbose=False)
print(f'The following channels are currently marked as bad:\n'
      f'    {", ".join(raw.info["bads"])}\n')

# %%
# So currently, two channels are marked as bad: ``EEG 053`` and ``MEG 2443``.
# Let's assume that through visual data inspection, we found that two more
# MEG channels are problematic, and we would like to mark them as bad as well.
# To do that, we simply add them to a list, which we then pass to
# :func:`mne_bids.mark_channels`:

bads = ['MEG 0112', 'MEG 0131']
mark_channels(bids_path=bids_path, ch_names=bads, status='bad',
              verbose=False)

# %%
# That's it! Let's verify the result.

raw = read_raw_bids(bids_path=bids_path, verbose=False)
print(f'After marking MEG 0112 and MEG 0131 as bad, the following channels '
      f'are now marked as bad:\n    {", ".join(raw.info["bads"])}\n')

# %%
# As you can see, now a total of **four** channels is marked as bad: the ones
# that were already bad when we started – ``EEG 053`` and ``MEG 2443`` – and
# the two channels we passed to :func:`mne_bids.mark_channels` –
# ``MEG 0112`` and ``MEG 0131``. This shows that marking bad channels via
# :func:`mne_bids.mark_channels`, by default, is an **additive** procedure,
# which allows you to mark additional channels as bad while retaining the
# information about all channels that had *previously* been marked as bad.
#
# If you instead would like to **replace** the collection of bad channels
# entirely, pass the argument ``overwrite=True``:

bads = ['MEG 0112', 'MEG 0131']
mark_channels(bids_path=bids_path, ch_names=bads, status='bad', verbose=False)

raw = read_raw_bids(bids_path=bids_path, verbose=False)
print(f'After marking MEG 0112 and MEG 0131 as bad and passing '
      f'`overwrite=True`, the following channels '
      f'are now marked as bad:\n    {", ".join(raw.info["bads"])}\n')

# %%
# Lastly, if you're looking for a way to mark all channels as good, simply
# pass an empty list as ``ch_names``, combined with ``overwrite=True``:

bads = []
mark_channels(bids_path=bids_path, ch_names=bads, status='bad', verbose=False)

raw = read_raw_bids(bids_path=bids_path, verbose=False)
print(f'After passing `ch_names=[]` and `overwrite=True`, the following '
      f'channels are now marked as bad:\n    {", ".join(raw.info["bads"])}\n')
