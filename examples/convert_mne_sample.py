"""
============================================
Convert Raw data to a BIDS-compatible folder
============================================

Brain Imaging Data Structure (BIDS) MEG is a new standard for
storing MEG files. This example demonstrates how to convert
your existing files into a BIDS-compatible folder.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>

# License: BSD (3-clause)

###############################################################################
# Let us import mne_bids

import os.path as op
from mne.datasets import sample
from mne_bids import raw_to_bids
from mne_bids.utils import print_dir_tree

###############################################################################
# And define the paths and event_id dictionary.

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

raw_file = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
output_path = op.join(data_path, '..', 'MNE-sample-data-bids')

###############################################################################
# Finally, we specify the raw_file and events_data

raw_to_bids(subject_id='01', task='audiovisual', raw_file=raw_file,
            kind='meg', output_path=output_path, run='01', session_id='01',
            events_data=events_data, event_id=event_id, overwrite=True)

###############################################################################
# Now let's see the structure of the BIDS folder we created.
print_dir_tree(output_path)
