"""
===================================
Create a BIDS-compatible MEG folder
===================================

Brain Imaging Data Structure (BIDS) MEG is a new standard for
storing MEG files. This example demonstrates how to convert
your existing files into a BIDS-compatible folder.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

# License: BSD (3-clause)

import os.path as op
from mne.datasets import sample
from mne_bids import folder_to_bids

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

fnames = dict(events='sample_audvis_raw-eve.fif',
              raw='sample_audvis_raw.fif')

input_path = op.join(data_path, 'MEG', 'sample')
output_path = op.join(data_path, '..', 'MNE-sample-data-bids')
folder_to_bids(input_path=input_path, output_path=output_path,
               fnames=fnames, subject_id='01', run='01', task='audiovisual',
               event_id=event_id, overwrite=True)
