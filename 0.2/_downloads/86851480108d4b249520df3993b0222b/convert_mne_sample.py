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

from datetime import datetime

import mne
from mne.datasets import sample

from mne_bids import write_raw_bids, read_raw_bids, make_bids_basename
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
# Specify the raw_file and events_data and run the BIDS conversion.

raw = mne.io.read_raw_fif(raw_fname)
bids_basename = make_bids_basename(subject='01', session='01',
                                   task='audiovisual', run='01')
write_raw_bids(raw, bids_basename, output_path, events_data=events_data,
               event_id=event_id, overwrite=True)

###############################################################################
# Specify some empty room data and run BIDS conversion on it.
er_raw_fname = op.join(data_path, 'MEG', 'sample', 'ernoise_raw.fif')
er_raw = mne.io.read_raw_fif(er_raw_fname)
# For empty room data we need to specify that the subject ID is
# 'emptyroom', and that the task is 'noise'.
# We also need to specify the recording date in the format YYYYMMDD for the
# session id.
er_date = datetime.fromtimestamp(
    er_raw.info['meas_date'][0]).strftime('%Y%m%d')
er_bids_basename = 'sub-emptyroom_ses-{0}_task-noise'.format(er_date)
write_raw_bids(er_raw, er_bids_basename, output_path, overwrite=True)

###############################################################################
# Now let's see the structure of the BIDS folder we created.
print_dir_tree(output_path)

###############################################################################
# Finally, we can read the BIDS data we created as well.
raw, events, event_id = read_raw_bids(bids_basename + '_meg.fif', output_path)

###############################################################################
# The data is already in a convenient form to create epochs and evokeds.
epochs = mne.Epochs(raw, events, event_id)
epochs['Auditory'].average().plot()
