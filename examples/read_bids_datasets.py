"""
.. _read_bids_datasets-example:

======================
01. Read BIDS datasets
======================

When working with electrophysiological data in the BIDS format, an important
resource is the `OpenNeuro <https://openneuro.org/>`_ database. OpenNeuro
works great with MNE-BIDS because every dataset must pass a validator
that tests to ensure its format meets BIDS specifications before the dataset
can be uploaded, so you know the data will work with a script like in this
example without modification.

We have various data types that can be loaded via the ``read_raw_bids``
function:

- MEG
- EEG (scalp electrodes)
- iEEG (ECoG and SEEG)
- the anatomical MRI scan of a study participant

In this tutorial, we show how ``read_raw_bids`` can be used to load and
inspect BIDS-formatted data.

"""
# Authors: Adam Li <adam2392@gmail.com>
#          Richard Höchenberger <richard.hoechenberger@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

# %%
# Imports
# -------
# We are importing everything we need for this example:
import os
import os.path as op
import openneuro

from mne.datasets import sample
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report

# %%
# Download a subject's data from an OpenNeuro BIDS dataset
# --------------------------------------------------------
#
# Download the data, storing each in a ``target_dir`` target directory, which,
# in ``mne-bids`` terminology, is the `root` of each BIDS dataset. This example
# uses this `EEG dataset <https://openneuro.org/datasets/ds002778>`_ of
# resting-state recordings of patients with Parkinson's disease.
#

# .. note: If the keyword argument include is left out of
#          ``openneuro.download``, the whole dataset will be downloaded.
#          We're just using data from one subject to reduce the time
#          it takes to run the example.

dataset = 'ds002778'
subject = 'pd6'

# Download one subject's data from each dataset
bids_root = op.join(op.dirname(sample.data_path()), dataset)
if not op.isdir(bids_root):
    os.makedirs(bids_root)

openneuro.download(dataset=dataset, target_dir=bids_root,
                   include=[f'sub-{subject}'])

# %%
# Explore the dataset contents
# ----------------------------
#
# We can use MNE-BIDS to print a tree of all
# included files and folders. We pass the ``max_depth`` parameter to
# `mne_bids.print_dir_tree` to the output to four levels of folders, for
# better readability in this example.

print_dir_tree(bids_root, max_depth=4)

# %%
# We can even ask MNE-BIDS to produce a human-readbale summary report
# on the dataset contents.

print(make_report(bids_root))

# %%
# Now it's time to get ready for reading some of the data! First, we need to
# create an :class:`mne_bids.BIDSPath`, which is the workhorse object of
# MNE-BIDS when it comes to file and folder operations.
#
# For now, we're interested only in the EEG data in the BIDS root directory
# of the Parkinson's disease patient dataset. There were two sessions, one
# where the patients took their regular anti-Parkinsonian medications and
# one where they abstained for more than twelve hours. Let's start with the
# off-medication session.

datatype = 'eeg'
session = 'off'
bids_path = BIDSPath(root=bids_root, session=session, datatype=datatype)

# %%
# We can now retrieve a list of all MEG-related files in the dataset:

print(bids_path.match())

# %%
# The returned list contains ``BIDSpaths`` of 3 files:
# ``sub-pd6_ses-off_task-rest_channels.tsv``,
# ``sub-pd6_ses-off_task-rest_events.tsv``, and
# ``sub-pd6_ses-off_task-rest_eeg.bdf``.
# The first two are so-called sidecar files that contain information on the
# recording channels and experimental events, and the third one is the actual
# data file.
#
# Prepare reading the data
# ------------------------
#
# There is only one subject and one experimental task (``rest``).
# Let's use this knowledge to create a new ``BIDSPath`` with
# all the information required to actually read the EEG data. We also need to
# pass a ``suffix``, which is the last part of the filename just before the
# extension -- ``'channels'`` and ``'events'`` for the two TSV files in
# our example, and ``'eeg'`` for EEG raw data. For MEG and EEG raw data, the
# suffix is identical to the datatype, so don't let yourself be confused here!

task = 'rest'
suffix = 'eeg'

bids_path = BIDSPath(subject=subject, session=session, task=task,
                     suffix=suffix, datatype=datatype, root=bids_root)

# %%
# Now let's print the contents of ``bids_path``.

print(bids_path)

# %%
# You probably noticed two things: Firstly, this looks like an ordinary string
# now, not like the more-or-less neatly formatted output we saw before. And
# secondly, that there's suddenly a filename extension which we never specified
# anywhere!
#
# The reason is that when you call ``print(bids_path)``, ``BIDSPath`` returns
# a string representation of ``BIDSPath.fpath``, which looks different. If,
# instead, you simply typed ``bids_path`` (or ``print(repr(bids_path))``, which
# is the same) into your Python console, you would get the nicely formatted
# output:

bids_path

# %%
# The ``root`` here is – you guessed it – the directory we passed via the
# ``root`` parameter: the "home" of our BIDS dataset. The ``datatype``, again,
# is self-explanatory. The ``basename``, on the other hand, is created
# automatically based on the suffix and **BIDS entities**  we passed to
# ``BIDSPath``: in our case, ``subject``, ``session`` and ``task``.
#
# .. note::
#   There are many more supported entities, the most-commonly used among them
#   probably being ``acquisition``. Please see
#   :ref:`our introduction to BIDSPath <bidspath-example>` to learn more
#   about entities, ``basename``, and ``BIDSPath`` in general.
#
# But what about that filename extension, now? ``BIDSPath.fpath``, which –
# as you hopefully remember – is invoked when you run ``print(bids_path)`` –
# employs some heuristics to auto-detect some missing filename components.
# Omitting the filename extension in your script can make your code
# more portable. Note that, however, you **can** explicitly specify an
# extension too, by passing e.g. ``extension='.bdf'`` to ``BIDSPath``.

# %%
# Read the data
# -------------
#
# Let's read the data! It's just a single line of code.

raw = read_raw_bids(bids_path=bids_path, verbose=False)

# %%
# Now we can inspect the ``raw`` object to check that it contains to correct
# metadata.
#
# Basic subject metadata is here.

print(raw.info['subject_info'])

# %%
# Power line frequency is here.

print(raw.info['line_freq'])

# %%
# Sampling frequency is here.

print(raw.info['sfreq'])

# %%
# Events are now Annotations
print(raw.annotations)

# %%
# Plot the raw data.

raw.plot()

# %%
# .. LINKS
#
# .. _parkinsons_eeg_dataset:
#    https://openneuro.org/datasets/ds002778
#
