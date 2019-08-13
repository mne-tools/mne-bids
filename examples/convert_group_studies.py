"""
=================================
BIDS conversion for group studies
=================================

Here, we show how to do BIDS conversion for group studies. We use the SPM faces
data for the purpose of the example:
https://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/

"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>

# License: BSD (3-clause)

###############################################################################
# Let us import ``mne_bids``

import os.path as op

import mne
from mne_bids import write_raw_bids, make_bids_basename
from mne_bids.utils import print_dir_tree

###############################################################################
# And fetch the data.
# .. warning:: This will download 1.6 GB of data!

data_path = mne.datasets.spm_face.data_path()

# Prepare a path to save the BIDS converted data
output_path = op.join(op.dirname(data_path), 'MNE-spm-face-bids')

###############################################################################
# Define event_ids.

event_id = {
    'faces': 1,
    'scrambled': 2,
}

###############################################################################
# Let us loop over the subjects and create a BIDS-compatible folder
#
# Note that the SPM faces data actually contains two runs of a single subject.
# But for the sake of the example, we will pretend that these are two subjects.

subject_ids = [1, 2]

for subject_id in subject_ids:
    raw_fname = op.join(data_path, 'MEG', 'spm',
                        'SPM_CTF_MEG_example_faces{}_3D.ds'.format(subject_id))

    raw = mne.io.read_raw_ctf(raw_fname)
    bids_basename = make_bids_basename(subject='{:02}'.format(subject_id),
                                       session='01', task='VisualFaces')
    write_raw_bids(raw, bids_basename, output_path, event_id=event_id,
                   overwrite=True)

###############################################################################
# Now let's see the structure of the BIDS folder we created.

print_dir_tree(output_path)
