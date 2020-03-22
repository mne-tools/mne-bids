"""
======================================
Convert MNE sample data to BIDS format
======================================

This example demonstrates how to convert your existing files into a
BIDS-compatible folder.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>

# License: BSD (3-clause)

###############################################################################
# Let us import mne_bids

import os.path as op

import mne
from mne.datasets import sample

from mne_bids import write_raw_bids
from mne_bids.utils import print_dir_tree

###############################################################################
# And define the paths and event_id dictionary.

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
output_path = op.join(data_path, '..', 'MNE-sample-data-bids')

###############################################################################
# Finally, we specify the raw_file and events_data

raw = mne.io.read_raw_fif(raw_fname)
bids_basename = 'sub-01_ses-01_task-audiovisual_run-01'
write_raw_bids(raw, bids_basename, output_path, events_data=events_data,
               event_id=event_id, overwrite=True)

###############################################################################
# Now let's see the structure of the BIDS folder we created.
print_dir_tree(output_path)
