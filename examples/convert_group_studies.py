"""
=====================================
05. BIDS conversion for group studies
=====================================
Here, we show how to do BIDS conversion for group studies.
We will use the
`EEG Motor Movement/Imagery Dataset <https://doi.org/10.13026/C28G6P>`_
available on the PhysioBank database.
We recommend that you go through the more basic BIDS conversion example before
checking out this group conversion example: :ref:`ex-convert-mne-sample`
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

###############################################################################
# Let us import ``mne_bids``

import os.path as op

import mne
from mne.datasets import eegbci

from mne_bids import (write_raw_bids, BIDSPath,
                      get_anonymization_daysback, make_report,
                      print_dir_tree)

###############################################################################
# And fetch the data for several subjects and runs of a single task.

subject_ids = [1, 2]

# The run numbers in the eegbci are not consecutive ... we follow the online
# documentation to get the 1st, 2nd, and 3rd run of one of the the motor
# imagery task
runs = [
    4,   # This is run #1 of imagining to open/close left or right fist
    8,   # ... run #2
    12,  # ... run #3
]

# map the eegbci run numbers to the number of the run in the motor imagery task
run_map = dict(zip(runs, range(1, 4)))

for subject_id in subject_ids:
    eegbci.load_data(subject=subject_id, runs=runs, update_path=True)

# get path to MNE directory with the downloaded example data
mne_data_dir = mne.get_config('MNE_DATASETS_EEGBCI_PATH')
data_dir = op.join(mne_data_dir, 'MNE-eegbci-data')

###############################################################################
# Define event_ids, this is knowledge we get from the online documentation of
# the data as well.

event_id = {
    'imagine_motion_fist/left': 1,
    'imagine_motion_fist/right': 2,
}

###############################################################################
# Let us loop over the subjects and create BIDS-compatible folder

# Make a path where we can save the data to
bids_root = op.join(mne_data_dir, 'eegmmidb_bids_group_conversion')

# Get a list of the raw objects for this dataset to use their dates
# to determine the number of daysback to use to anonymize.
# While we're looping through the files, also generate the
# BIDS-compatible names that will be used to save the files in BIDS.
raw_list = list()
bids_list = list()
for subject_id in subject_ids:
    for run in runs:
        raw_fname = eegbci.load_data(subject=subject_id, runs=run)[0]
        raw = mne.io.read_raw_edf(raw_fname)
        raw.info['line_freq'] = 50  # specify power line frequency
        raw_list.append(raw)
        bids_path = BIDSPath(subject=f'{subject_id:03}',
                             session='01', task='MotorImagery',
                             run=f'{run_map[run]:02}')
        bids_list.append(bids_path)

daysback_min, daysback_max = get_anonymization_daysback(raw_list)

for raw, bids_path in zip(raw_list, bids_list):
    # By using the same anonymization `daysback` number we can
    # preserve the longitudinal structure of multiple sessions for a
    # single subject and the relation between subjects. Be sure to
    # change or delete this number before putting code online, you
    # wouldn't want to inadvertently de-anonymize your data.
    write_raw_bids(raw, bids_path, bids_root, event_id=event_id,
                   anonymize=dict(daysback=daysback_min + 2117),
                   overwrite=True)

###############################################################################
# Now let's see the structure of the BIDS folder we created.

print_dir_tree(bids_root)

###############################################################################
# Now let's generate a report on the dataset.
dataset_report = make_report(bids_root=bids_root)
print(dataset_report)
