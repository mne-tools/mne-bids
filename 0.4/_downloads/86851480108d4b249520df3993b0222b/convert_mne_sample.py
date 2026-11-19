"""
.. _ex-convert-mne-sample:

==========================================
03. Convert MNE sample data to BIDS format
==========================================

In this example we will use MNE-BIDS to organize the MNE sample data according
to the BIDS standard.
In a second step we will read the organized dataset using MNE-BIDS.

"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

###############################################################################
# First we import some basic Python libraries, followed by MNE-Python and its
# sample data, and then finally the MNE-BIDS functions we need for this example

import os.path as op

import mne
from mne.datasets import sample

from mne_bids import write_raw_bids, read_raw_bids, make_bids_basename
from mne_bids.utils import print_dir_tree

###############################################################################
# Now we can read the MNE sample data. We define an `event_id` based on our
# knowledge of the data, to give meaning to events in the data.
#
# With `raw_fname` and `events_data` we determine where to get the sample data
# from. `output_path` determines where we will write the BIDS conversion to.

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
output_path = op.join(data_path, '..', 'MNE-sample-data-bids')

###############################################################################
#
# .. note::
#
#   ``mne-bids`` will try to infer as much information from the data as
#   possible to then save this data in BIDS specific "sidecar" files. For
#   example the manufacturer information, which is inferred from the data file
#   extension. However, sometimes inferring is ambiguous (e.g., if your file
#   format is non-standard for the manufacturer). In these cases, MNE-BIDS does
#   *not* guess and you will have to update your BIDS fields manually.
#
# Based on our path definitions above, we read the raw data file, define
# a new BIDS name for it, and then run the automatic BIDS conversion.

raw = mne.io.read_raw_fif(raw_fname)
bids_basename = make_bids_basename(subject='01', session='01',
                                   task='audiovisual', run='01')
write_raw_bids(raw, bids_basename, output_path, events_data=events_data,
               event_id=event_id, overwrite=True)

###############################################################################
# Now let's see the structure of the BIDS folder we created.

print_dir_tree(output_path)

###############################################################################
# A big advantage of having data organized according to BIDS is that software
# packages can automate your workflow. For example, reading the data back
# into MNE-Python can easily be done using :func:`read_raw_bids`.

bids_fname = bids_basename + '_meg.fif'
raw = read_raw_bids(bids_fname, output_path)

###############################################################################
# The resulting data is already in a convenient form to create epochs and
# evoked data.

events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id)
epochs['Auditory'].average().plot()
