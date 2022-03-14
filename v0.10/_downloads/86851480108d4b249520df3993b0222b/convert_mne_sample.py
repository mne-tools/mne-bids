"""
.. currentmodule:: mne_bids

.. _ex-convert-mne-sample:

==========================================
02. Convert MNE sample data to BIDS format
==========================================

In this example we will use MNE-BIDS to organize the MNE sample data according
to the BIDS standard.
In a second step we will read the organized dataset using MNE-BIDS.

"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD-3-Clause

# %%
# First we import some basic Python libraries, followed by MNE-Python and its
# sample data, and then finally the MNE-BIDS functions we need for this example

import os.path as op
import shutil

import mne
from mne.datasets import sample

from mne_bids import (write_raw_bids, read_raw_bids, write_meg_calibration,
                      write_meg_crosstalk, BIDSPath, print_dir_tree)
from mne_bids.stats import count_events

# %%
# Now we can read the MNE sample data. We define an `event_id` based on our
# knowledge of the data, to give meaning to events in the data.
#
# With `raw_fname` and `events_data`, we determine where to get the sample data
# from. `output_path` determines where we will write the BIDS conversion to.

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
output_path = op.join(data_path, '..', 'MNE-sample-data-bids')

# %%
# To ensure the output path doesn't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if op.exists(output_path):
    shutil.rmtree(output_path)

# %%
#
# .. note::
#
#   ``mne-bids`` will try to infer as much information from the data as
#   possible to then save this data in BIDS-specific "sidecar" files. For
#   example the manufacturer information, which is inferred from the data file
#   extension. However, sometimes inferring is ambiguous (e.g., if your file
#   format is non-standard for the manufacturer). In these cases, MNE-BIDS does
#   *not* guess and you will have to update your BIDS fields manually.
#
# Based on our path definitions above, we read the raw data file, define
# a new BIDS name for it, and then run the automatic BIDS conversion.

raw = mne.io.read_raw_fif(raw_fname)
raw.info['line_freq'] = 60  # specify power line frequency as required by BIDS

bids_path = BIDSPath(subject='01', session='01',
                     task='audiovisual', run='01', root=output_path)
write_raw_bids(raw, bids_path, events_data=events_data,
               event_id=event_id, overwrite=True)

# %%
# Let's pause and check that the information that we've written out to the
# sidecar files that describe our data is correct.

# Get the sidecar ``.json`` file
print(bids_path.copy().update(extension='.json').fpath.read_text(
    encoding='utf-8-sig'))

# %%
# The sample MEG dataset comes with fine-calibration and crosstalk files that
# are required when processing Elekta/Neuromag/MEGIN data using MaxFilter®.
# Let's store these data in appropriate places, too.

cal_fname = op.join(data_path, 'SSS', 'sss_cal_mgh.dat')
ct_fname = op.join(data_path, 'SSS', 'ct_sparse_mgh.fif')

write_meg_calibration(cal_fname, bids_path)
write_meg_crosstalk(ct_fname, bids_path)

# %%
# Now let's see the structure of the BIDS folder we created.

print_dir_tree(output_path)

# %%
# Now let's get an overview of the events on the whole dataset

counts = count_events(output_path)
counts

# %%
# A big advantage of having data organized according to BIDS is that software
# packages can automate your workflow. For example, reading the data back
# into MNE-Python can easily be done using :func:`read_raw_bids`.

raw = read_raw_bids(bids_path=bids_path)

# %%
# The resulting data is already in a convenient form to create epochs and
# evoked data.

events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id)
epochs['Auditory'].average().plot()

# %%
# It is trivial to retrieve the path of the fine-calibration and crosstalk
# files, too.

print(bids_path.meg_calibration_fpath)
print(bids_path.meg_crosstalk_fpath)

# %%
# The README created by :func:`write_raw_bids` also takes care of the citation
# for mne-bids. If you are preparing a manuscript, please make sure to also
# cite MNE-BIDS there.
readme = op.join(output_path, 'README')
with open(readme, 'r', encoding='utf-8-sig') as fid:
    text = fid.read()
print(text)
