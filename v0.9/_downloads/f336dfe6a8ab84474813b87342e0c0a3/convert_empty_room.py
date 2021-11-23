"""
.. _ex-convert-empty-room:

==========================================
09. Storing empty room data in BIDS format
==========================================

This example demonstrates how to store empty room data in BIDS format
and how to retrieve them.
"""

# Authors: Mainak Jas <mainakjas@gmail.com>
#
# License: BSD-3-Clause

# %%
# We are dealing with MEG data, which is often accompanied by so-called
# "empty room" recordings for noise modeling. Below we show that we can use
# MNE-BIDS to also save such a recording with the just converted data.
#
# Let us first import mne_bids.

import os.path as op
import shutil

from datetime import datetime, timezone

import mne
from mne.datasets import sample

from mne_bids import (write_raw_bids, read_raw_bids,
                      BIDSPath, print_dir_tree)

# %%
# And define the paths and event_id dictionary.

data_path = sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')

bids_root = op.join(data_path, '..', 'MNE-sample-data-bids')

# %%
# To ensure the output path doesn't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if op.exists(bids_root):
    shutil.rmtree(bids_root)

# %%
# Specify the raw_file and events_data and run the BIDS conversion, and write
# the BIDS data.

raw = mne.io.read_raw_fif(raw_fname)
raw.info['line_freq'] = 60  # specify power line frequency as required by BIDS

bids_path = BIDSPath(subject='01', session='01',
                     task='audiovisual', run='01', root=bids_root)
write_raw_bids(raw, bids_path, overwrite=True)

# %%
# Specify some empty room data and run BIDS conversion on it.
er_raw_fname = op.join(data_path, 'MEG', 'sample', 'ernoise_raw.fif')
er_raw = mne.io.read_raw_fif(er_raw_fname)
er_raw.info['line_freq'] = 60  # specify power line frequency as req. by BIDS

# For empty room data we need to specify the recording date in the format
# YYYYMMDD for the session id.
er_date = er_raw.info['meas_date'].strftime('%Y%m%d')
print(er_date)

# %%
# The measurement date is
raw_date = raw.info['meas_date'].strftime('%Y%m%d')
print(raw_date)

# %%
# We also need to specify that the subject ID is 'emptyroom', and that the
# task is 'noise' (these are BIDS rules).
er_bids_path = BIDSPath(subject='emptyroom', session=er_date,
                        task='noise', root=bids_root)
write_raw_bids(er_raw, er_bids_path, overwrite=True)

# %%
# Just to illustrate, we can save more than one empty room file for different
# dates. Here, they will all contain the same data but in your study, they
# will be different on different days.
dates = ['20021204', '20021201', '20021001']

for date in dates:
    er_bids_path.update(session=date)
    er_meas_date = datetime.strptime(date, '%Y%m%d')
    er_raw.set_meas_date(er_meas_date.replace(tzinfo=timezone.utc))
    write_raw_bids(er_raw, er_bids_path, overwrite=True)

# %%
# Let us look at the directory structure

print_dir_tree(bids_root)

# %%
# To get an accurate estimate of the noise, it is important that the empty
# room recording be as close in date as the raw data.
# We can retrieve the basename corresponding to the empty room
# recording that is closest in time to the experimental measurement.

er_bids_path = bids_path.find_empty_room()
print(er_bids_path)

# %%
# Finally, we can read the empty room file using
raw = read_raw_bids(bids_path=er_bids_path)
print(raw)
