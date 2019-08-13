"""
=================================
BIDS conversion for group studies
=================================

Here, we show how to do BIDS conversion for group studies.
The data from Wakeman et al. [1]_ is available here:
https://openneuro.org/datasets/ds000117

References
----------
.. [1] Wakeman, Daniel G., and Richard N. Henson. "A multi-subject, multi-modal
   human neuroimaging dataset." Scientific data, 2 (2015): 150001.

"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>

# License: BSD (3-clause)

###############################################################################
# Let us import ``mne_bids``

import os.path as op
import subprocess

import mne
from mne_bids import write_raw_bids, make_bids_basename
from mne_bids.utils import print_dir_tree

###############################################################################
# And fetch the data.
# .. warning :: This will download 11.8 GB of data!

repo = 'ds000117'
data_address = 's3://openneuro.org/{}/'.format(repo)

data_path = op.join(op.expanduser('~'), 'mne_data', 'mne_bids_examples', repo)

# Prepare the aws command
cmd = ['aws', 's3', 'sync', '--no-sign-request', data_address, data_path,
       '--exclude', '*']

subject_ids = [1, 2]
for subject_id in subject_ids:
    cmd += ['--include', 'sub-{:02}*'.format(subject_id)]

# Download
subprocess.run(cmd)


output_path = op.join(data_path, 'ds000117-bids')

###############################################################################
# Define event_ids.

event_id = {
    'face/famous/first': 5,
    'face/famous/immediate': 6,
    'face/famous/long': 7,
    'face/unfamiliar/first': 13,
    'face/unfamiliar/immediate': 14,
    'face/unfamiliar/long': 15,
    'scrambled/first': 17,
    'scrambled/immediate': 18,
    'scrambled/long': 19,
}

###############################################################################
# Let us loop over the subjects and create a BIDS-compatible folder

# The data contains 6 runs
runs = range(1, 7)

for subject_id in subject_ids:
    subject = 'sub-{:02}'.format(subject_id)
    for run in runs:
        raw_fname = op.join(data_path, subject, 'ses-meg', 'meg',
                            ('{}_ses-meg_task-facerecognition_{}_meg.fif'
                             .format(subject, 'run-{:02}'.format(run)))
                            )

        raw = mne.io.read_raw_fif(raw_fname)
        raw.info['meas_date'] = None
        bids_basename = make_bids_basename(subject=str(subject_id),
                                           session='01', task='VisualFaces',
                                           run=str(run))
        write_raw_bids(raw, bids_basename, output_path, event_id=event_id,
                       overwrite=True)

###############################################################################
# Now let's see the structure of the BIDS folder we created.

print_dir_tree(output_path)
