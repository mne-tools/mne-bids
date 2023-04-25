"""
.. currentmodule:: mne_bids

.. _ex-convert-mne-sample:

==========================================
02. Convert MNE sample data to BIDS format
==========================================

In this example we will use MNE-BIDS to organize the MNE sample data according
to the BIDS standard.
In a second step we will read the organized dataset using MNE-BIDS.

.. _BIDS dataset_description.json definition: https://bids-specification.readthedocs.io/en/latest/modality-agnostic-files.html#dataset-description
.. _ds000248 dataset_description.json: https://github.com/sappelhoff/bids-examples/blob/master/ds000248/dataset_description.json
"""  # noqa: D400 D205 E501

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

import json
import os.path as op
from pprint import pprint
import shutil

import mne
from mne.datasets import sample

from mne_bids import (write_raw_bids, read_raw_bids, write_meg_calibration,
                      write_meg_crosstalk, BIDSPath, print_dir_tree,
                      make_dataset_description)
from mne_bids.stats import count_events

# %%
# Now we can read the MNE sample data. We define an `event_id` based on our
# knowledge of the data, to give meaning to events in the data.
#
# With `raw_fname` and `events`, we determine where to get the sample data
# from. `output_path` determines where we will write the BIDS conversion to.

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
er_fname = op.join(data_path, 'MEG', 'sample', 'ernoise_raw.fif')  # empty room
events_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
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
# a new BIDS name for it, and then run the automatic BIDS conversion for both
# the experimental data and its associated empty-room recording.

raw = mne.io.read_raw(raw_fname)
raw_er = mne.io.read_raw(er_fname)

# specify power line frequency as required by BIDS
raw.info['line_freq'] = 60
raw_er.info['line_freq'] = 60

task = 'audiovisual'
bids_path = BIDSPath(
    subject='01',
    session='01',
    task=task,
    run='1',
    datatype='meg',
    root=output_path
)
write_raw_bids(
    raw=raw,
    bids_path=bids_path,
    events=events_fname,
    event_id=event_id,
    empty_room=raw_er,
    overwrite=True
)

# %%
# Let's pause and check that the information that we've written out to the
# sidecar files that describe our data is correct.

# Get the sidecar ``.json`` file
sidecar_json_bids_path = bids_path.copy().update(
    suffix='meg', extension='.json'
)
sidecar_json_content = sidecar_json_bids_path.fpath.read_text(
    encoding='utf-8-sig'
)
print(sidecar_json_content)

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
# We can easily get the :class:`mne_bids.BIDSPath` of the empty-room recording
# that was associated with the experimental data while writing. The empty-room
# data can then be loaded with :func:`read_raw_bids`.

er_bids_path = bids_path.find_empty_room(use_sidecar_only=True)
er_data = read_raw_bids(er_bids_path)
er_data

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

# %%
# It is also generally a good idea to add a description of your dataset,
# see the `BIDS dataset_description.json definition`_ for more information.

how_to_acknowledge = """\
If you reference this dataset in a publication, please acknowledge its \
authors and cite MNE papers: A. Gramfort, M. Luessi, E. Larson, D. Engemann, \
D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen, \
MNE software for processing MEG and EEG data, NeuroImage, Volume 86, \
1 February 2014, Pages 446-460, ISSN 1053-8119 \
and \
A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, \
R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data \
analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013, \
ISSN 1662-453X"""

make_dataset_description(
    path=bids_path.root,
    name=task,
    authors=["Alexandre Gramfort", "Matti Hämäläinen"],
    how_to_acknowledge=how_to_acknowledge,
    acknowledgements="""\
Alexandre Gramfort, Mainak Jas, and Stefan Appelhoff prepared and updated the \
data in BIDS format.""",
    data_license='CC0',
    ethics_approvals=['Human Subjects Division at the University of Washington'],  # noqa: E501
    funding=[
        "NIH 5R01EB009048",
        "NIH 1R01EB009048",
        "NIH R01EB006385",
        "NIH 1R01HD40712",
        "NIH 1R01NS44319",
        "NIH 2R01NS37462",
        "NIH P41EB015896",
        "ANR-11-IDEX-0003-02",
        "ERC-StG-263584",
        "ERC-StG-676943",
        "ANR-14-NEUC-0002-01"
    ],
    references_and_links=[
        "https://doi.org/10.1016/j.neuroimage.2014.02.017",
        "https://doi.org/10.3389/fnins.2013.00267",
        "https://mne.tools/stable/overview/datasets_index.html#sample"
    ],
    doi="doi:10.18112/openneuro.ds000248.v1.2.4",
    overwrite=True
)
desc_json_path = bids_path.root / 'dataset_description.json'
with open(desc_json_path, 'r', encoding='utf-8-sig') as fid:
    pprint(json.loads(fid.read()))

# %%
# This should be very similar to the `ds000248 dataset_description.json`_!
