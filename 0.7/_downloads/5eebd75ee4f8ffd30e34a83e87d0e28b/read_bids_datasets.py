"""
.. _read_bids_datasets-example:

======================
01. Read BIDS datasets
======================

When working with electrophysiological data in the BIDS format, we usually have
varying data types, which can be loaded via the ``read_raw_bids`` function.

- MEG
- EEG (scalp electrodes)
- iEEG (ECoG and SEEG)
- the anatomical MRI scan of a study participant

In this tutorial, we show how ``read_raw_bids`` can be used to load and
inspect BIDS-formatted data.

"""
# Authors: Adam Li <adam2392@gmail.com>
#          Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD (3-clause)

###############################################################################
# Imports
# -------
# We are importing everything we need for this example:
from mne.datasets import somato

from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report

###############################################################################
# We will be using the `MNE somato data <mne_somato_data_>`_, which
# is already stored in BIDS format.
# For more information, you can check out the
# respective :ref:`example <ex-convert-mne-sample>`.

###############################################################################
# Download the ``somato`` BIDS dataset
# ------------------------------------
#
# Download the data if it hasn't been downloaded already, and return the path
# to the download directory. This directory is the so-called `root` of this
# BIDS dataset.
bids_root = somato.data_path()

###############################################################################
# Explore the dataset contents
# ----------------------------
#
# We can use MNE-BIDS to print a tree of all
# included files and folders. We pass the ``max_depth`` parameter to
# `mne_bids.print_dir_tree` to the output to three levels of folders, for
# better readability in this example.

print_dir_tree(bids_root, max_depth=3)

###############################################################################
# We can even ask MNE-BIDS to produce a human-readbale summary report
# on the dataset contents.

print(make_report(bids_root))

###############################################################################
# Now it's time to get ready for reading some of the data! First, we need to
# create a `mne_bids.BIDSPath`, which is the working horse object of MNE-BIDS
# when it comes to file and folder operations.
#
# For now, we're interested only in the MEG data in the BIDS root directory
# of the ``somato`` dataset.

datatype = 'meg'
bids_path = BIDSPath(root=bids_root, datatype=datatype)

###############################################################################
# We can now retrieve a list of all MEG-related files in the dataset:

print(bids_path.match())

###############################################################################
# The returned list contains ``BIDSpaths`` of 3 files:
# ``sub-01_task-somato_channels.tsv``, ``sub-01_task-somato_events.tsv``, and
# ``sub-01_task-somato_meg.fif``.
# The first two are so-called sidecar files that contain information on the
# recording channels and experimental events, and the third one is the actual
# MEG data file.
#
# Prepare reading the data
# ------------------------
#
# There is obviously only one subject (``01``) and one experimental task
# (``somato``). Let's use this knowledge to create a new ``BIDSPath`` with
# all the information required to actually read the MEG data. We also need to
# pass a ``suffix``, which is the last part of the filename just before the
# extension -- ``'channels'`` and ``'events'`` for the two TSV files in
# our example, and ``'meg'`` for MEG raw data. For MEG and EEG raw data, the
# suffix is identical to the datatype, so don't let yourselve be confused here!

bids_root = somato.data_path()
datatype = 'meg'
subject = '01'
task = 'somato'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, task=task, suffix=suffix,
                     datatype=datatype, root=bids_root)

###############################################################################
# Now let's print the contents of ``bids_path``.

print(bids_path)

###############################################################################
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

###############################################################################
# The ``root`` here is – you guessed it – the directory we passed via the
# ``root`` parameter: the "home" of our BIDS dataset. The ``datatype``, again,
# is self-explanatory. The ``basename``, on the other hand, is created
# automatically based on the suffix and **BIDS entities**  we passed to
# ``BIDSPath``: in our case, ``subject`` and ``task``.
#
# .. note::
#   There are many more supported entities, the most-commonly used among them
#   probably being ``session``. Please see
#   :ref:`our introduction to BIDSPath <bidspath-example>` to learn more
#   about entities, ``basename``, and ``BIDSPath`` in general.
#
# But what about that filename extension, now? ``BIDSPath.fpath``, which –
# as you hopefully remember – is invoked when you run ``print(bids_path)`` –
# employs some heuristics to auto-detect some missing filename components.
# Omitting the filename extension in your script can make your code
# more portable. Note that, however, you **can** explicitly specify an
# extension too, by passing e.g. ``extension='.fif'`` to ``BIDSPath``.

###############################################################################
# Read the data
# -------------
#
# Let's read the data! It's just a single line of code.

raw = read_raw_bids(bids_path=bids_path, verbose=False)

###############################################################################
# Now we can inspect the ``raw`` object to check that it contains to correct
# metadata.
#
# Basic subject metadata is here.

print(raw.info['subject_info'])

###############################################################################
# Power line frequency is here.

print(raw.info['line_freq'])

###############################################################################
# Sampling frequency is here.

print(raw.info['sfreq'])

###############################################################################
# Events are now Annotations
print(raw.annotations)

###############################################################################
# Plot the raw data.

raw.plot()

###############################################################################
# .. LINKS
#
# .. _mne_somato_data:
#    https://mne.tools/dev/generated/mne.datasets.somato.data_path.html
#
