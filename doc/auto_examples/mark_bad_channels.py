"""
===============================================
03. Changing which channels are marked as "bad"
===============================================

You can use MNE-BIDS to mark MEG or (i)EEG recording channels as "bad", for
example if the connected sensor produced mostly noise – or no signal at
all.

Similarly, you can declare channels as "good", should you discover they were
incorrectly marked as bad.
"""

# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
# License: BSD (3-clause)

###############################################################################
# We will demonstrate how to mark individual channels as bad on the MNE
# "sample" dataset. After that, we will mark channels as good again.
#
# Let's start by importing the required modules and functions, reading the
# "sample" data, and writing it in the BIDS format.

import os.path as op
import mne
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids, mark_bad_channels

data_path = mne.datasets.sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
bids_root = op.join(data_path, '..', 'MNE-sample-data-bids')
bids_path = BIDSPath(subject='01', session='01', task='audiovisual', run='01',
                     root=bids_root)

raw = mne.io.read_raw_fif(raw_fname, verbose=False)
raw.info['line_freq'] = 60  # Specify power line frequency as required by BIDS.
write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

###############################################################################
# Read the (now BIDS-formatted) data and print a list of channels currently
# marked as bad.

raw = read_raw_bids(bids_path=bids_path, verbose=False)
print(f'The following channels are currently marked as bad:\n'
      f'    {", ".join(raw.info["bads"])}\n')

###############################################################################
# So currently, two channels are maked as bad: ``EEG 053`` and ``MEG 2443``.
# Let's assume that through visual data inspection, we found that two more
# MEG channels are problematic, and we would like to mark them as bad as well.
# To do that, we simply add them to a list, which we then pass to
# :func:`mne_bids.mark_bad_channels`:

bads = ['MEG 0112', 'MEG 0131']
mark_bad_channels(ch_names=bads, bids_path=bids_path, verbose=False)

###############################################################################
# That's it! Let's verify the result.

raw = read_raw_bids(bids_path=bids_path, verbose=False)
print(f'After marking MEG 0112 and MEG 0131 as bad, the following channels '
      f'are now marked as bad:\n    {", ".join(raw.info["bads"])}\n')

###############################################################################
# As you can see, now a total of **four** channels is marked as bad: the ones
# that were already bad when we started – ``EEG 053`` and ``MEG 2443`` – and
# the two channels we passed to :func:`mne_bids.mark_bad_channels` –
# ``MEG 0112`` and ``MEG 0131``. This shows that marking bad channels via
# :func:`mne_bids.mark_bad_channels`, by default, is an **additive** procedure,
# which allows you to mark additional channels as bad while retaining the
# information about all channels that had *previously* been marked as bad.
#
# If you instead would like to **replace** the collection of bad channels
# entirely, pass the argument ``overwrite=True``:

bads = ['MEG 0112', 'MEG 0131']
mark_bad_channels(ch_names=bads, bids_path=bids_path, overwrite=True,
                  verbose=False)

raw = read_raw_bids(bids_path=bids_path, verbose=False)
print(f'After marking MEG 0112 and MEG 0131 as bad and passing '
      f'`overwrite=True`, the following channels '
      f'are now marked as bad:\n    {", ".join(raw.info["bads"])}\n')

###############################################################################
# Lastly, if you're looking for a way to mark all channels as good, simply
# pass an empty list as ``ch_names``, combined with ``overwrite=True``:

bads = []
mark_bad_channels(ch_names=bads, bids_path=bids_path, overwrite=True,
                  verbose=False)

raw = read_raw_bids(bids_path=bids_path, verbose=False)
print(f'After passing `ch_names=[]` and `overwrite=True`, the following '
      f'channels are now marked as bad:\n    {", ".join(raw.info["bads"])}\n')
