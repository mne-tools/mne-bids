"""
=====================================
05. BIDS conversion for group studies
=====================================

Here, we show how to do BIDS conversion for group studies.
The data from Wakeman et al. [1]_ is available here:
https://openneuro.org/datasets/ds000117

We recommend that you go through the more basic BIDS conversion example before
checking out this group conversion example: :ref:`ex-convert-mne-sample`

References
----------
.. [1] Wakeman, Daniel G., and Richard N. Henson. "A multi-subject, multi-modal
   human neuroimaging dataset." Scientific data, 2 (2015): 150001.

"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>

# License: BSD (3-clause)

###############################################################################
# Let us import ``mne_bids``

import os.path as op

import mne
from mne_bids import (write_raw_bids, make_bids_basename,
                      get_anonymization_daysback)
from mne_bids.datasets import fetch_faces_data
from mne_bids.utils import print_dir_tree

###############################################################################
# And fetch the data.
#
# .. warning:: This will download 7.9 GB of data for one subject!

subject_ids = [1]
runs = range(1, 7)

home = op.expanduser('~')
data_path = op.join(home, 'mne_data', 'mne_bids_examples')
repo = 'ds000117'
fetch_faces_data(data_path, repo, subject_ids)

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
# Let us loop over the subjects and create BIDS-compatible folder

# Get a list of the raw objects for this dataset to use their dates
# to determine the number of daysback to use to anonymize.
# While we're looping through the files, also generate the
# BIDS-compatible names that will be used to save the files in BIDS.
raw_list = list()
bids_list = list()
for subject_id in subject_ids:
    subject = 'sub%03d' % subject_id
    for run in runs:
        raw_fname = op.join(data_path, repo, subject, 'MEG',
                            'run_%02d_raw.fif' % run)
        raw = mne.io.read_raw_fif(raw_fname)
        raw_list.append(raw)
        bids_basename = make_bids_basename(subject=str(subject_id),
                                           session='01', task='VisualFaces',
                                           run=str(run))
        bids_list.append(bids_basename)

daysback_min, daysback_max = get_anonymization_daysback(raw_list)

for raw, bids_basename in zip(raw_list, bids_list):
    # By using the same anonymization `daysback` number we can
    # preserve the longitudinal structure of multiple sessions for a
    # single subject and the relation between subjects. Be sure to
    # change or delete this number before putting code online, you
    # wouldn't want to inadvertently de-anonymize your data.
    write_raw_bids(raw, bids_basename, output_path, event_id=event_id,
                   anonymize=dict(daysback=daysback_min + 2117),
                   overwrite=True)

###############################################################################
# Now let's see the structure of the BIDS folder we created.

print_dir_tree(output_path)
