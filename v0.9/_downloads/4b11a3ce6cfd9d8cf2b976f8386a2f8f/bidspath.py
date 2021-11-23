"""
.. _bidspath-example:

===============================
10. An introduction to BIDSPath
===============================

BIDSPath is MNE-BIDS's working horse when it comes to file and folder
operations. Learn here how to use it.
"""
# Author: Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD-3-Clause

# %%
# Obviously, to start exploring BIDSPath, we first need to import it.

from pathlib import Path

import mne_bids
from mne_bids import BIDSPath

# %%
# Now let's discuss a little bit of background on the BIDS file and folder
# naming scheme. The first term we are going to introduce is the **BIDS root**.
# The BIDS root is simply the root folder of your BIDS dataset. For
# example, if the BIDS data of one of your studies is stored in
# `/Users/me/Studies/Study_01`, then this will be the BIDS root.
#
# Similarly, if you have **no** BIDS dataset to begin with, you need to
# consider where to store your data upon BIDS conversion. Again, the intended
# target folder will be the BIDS root of your data.
#
# For the purpose of this demonstration, let's pick the ``tiny_bids`` example
# dataset that ships with the MNE-BIDS test suite.

# We are using a pathlib.Path object for convenience, but you could just use
# a string to specify ``bids_root`` here.
bids_root = Path(mne_bids.__file__).parent / 'tests' / 'data' / 'tiny_bids'

# %%
# This refers to a folder named ``my_bids_root`` in the current working
# directory. Finally, let is create a ``BIDSPath``, and tell it about our
# BIDS root. We can then also query the ``BIDSPath`` for its root.

bids_path = BIDSPath(root=bids_root)
print(bids_path.root)

# %%
# Great! But not really useful so far. BIDS also asks us to specify **subject
# identifiers**. We can either create a new ``BIDSPath``, or update our
# existing one. The value can be retrieved via the ``.subject`` attribute.

subject = '01'

# Option 1: Create an entirely new BIDSPath.
bids_path_new = BIDSPath(subject=subject, root=bids_root)
print(bids_path_new.subject)

# Option 2: Update the existing BIDSPath in-place.
bids_path.update(subject=subject)
print(bids_path.subject)

# %%
# In this example, we are going to update the existing ``BIDSPath`` using its
# ``update()`` method. But note that all parameters we pass to this method can
# also be used when creating a ``BIDSPath``.
#
# Many studies consist of multiple **sessions**. As you may have guessed,
# BIDS specifies how to store data for each session, and consequently,
# ``BIDSPath`` handles this for you too! Let's update our ``BIDSPath`` with
# information on our experimental session, and try to retrieve it again via
# ``.session``.

session = 'eeg'
bids_path.update(session=session)
print(bids_path.session)

# %%
# Now that was easy! We're almost there! We also need to specify a
# **data type**, i.e., ``meg`` for MEG data, ``eeg`` and ``ieeg`` for EEG and
# iEEG data, or ``anat`` for anatomical MRI scans. Typically, MNE-BIDS will
# infer the data type of your data automatically, for example when writing data
# using `mne_bids.write_raw_bids`. For the sake of this example, however, we
# are going to specify the data type explicitly.

datatype = 'eeg'
bids_path.update(datatype=datatype)
print(bids_path.datatype)

# %%
# Excellent! Let's have a look at the path we have constructed!
print(bids_path)

# %%
# As you can see, ``BIDSPath`` automatically arranged all the information we
# provided such that it creates a valid BIDS folder structure. You can also
# retrieve a `pathlib.Path` object of this path:

pathlib_path = bids_path.fpath
pathlib_path

# %%
# Let's have a closer look at the components of our ``BIDSPath`` again.

bids_path

# %%
# The most interesting thing here is probably the **basename**. It's what
# MNE-BIDS uses to name individual files. The basename consists of a set of
# so-called **entities**, which are concatenated using underscores. You can
# access it directly:

bids_path.basename

# %%
# The two entities you can see here are the ``subject`` entity (``sub``) and
# the ``session`` entity (``ses``). Each entity name also has a value; for
# ``sub``, this is ``01``, and for ``ses``, it is ``eeg`` in our example.
# Entity names (or "keys") and values are separated via hyphens.
# BIDS knows a much larger number of entities, and MNE-BIDS allows you to make
# use of them. To get a list of all supported entities, use:

bids_path.entities

# %%
# As you can see, most entity keys are set to ``None``, which is the default
# and implies that no value has been set. Let us add a ``run`` entity, and
# remove the ``session``:

run = '01'
session = None
bids_path.update(run=run, session=session)
bids_path

# %%
# As you can see, the ``basename`` has been updated. In fact, the entire
# **path** has been updated, and the ``ses-eeg`` folder has been dropped from
# the path:

print(bids_path.fpath)

# %%
# Oups! The cell above produced a ``RuntimeWarning`` that our data file could
# not be found. That's because we changed the ``run`` and ``session`` entities
# above, and the ``tiny_bids`` dataset does not contain corresponding data.
#
# That shows us that ``BIDSPath`` is doing a lot of guess-work and checking
# in the background, but note that this may change in the future.
#
# For now, let's revert to the last working iteration of our ``bids_path``
# instance.

bids_path.update(run=None, session='eeg')
print(bids_path.fpath)

# %%
# Awesome! We're almost done! Two important things are still missing, though:
# the so-called **suffix** and the filename **extension**. Sometimes these
# terms are used interchangeably, but in BIDS, they have a very specific
# and different meaning!
#
# The **suffix** is the last part of a BIDS filename before the extension. It
# is the same as the datatype for MEG, EEG, and iEEG recordings (i.e.
# ``meg``, ``eeg``, and ``ieeg``, respectively) and ``T1w`` for T1-weighted
# MRI scans. But the suffix is also used to create the names of sidecar files
# like ``*_events.tsv``.
#
# Which brings us directly to the **extension**: the very last part of a
# filename. In MNE-BIDS, the extension contains a leading period, e.g.
# ``.tsv``.
# Let's put our new knowledge to use!

bids_path.update(suffix='eeg', extension='.vhdr')
print(bids_path.fpath)
bids_path

# %%
# By default, most MNE-BIDS functions will try to infer to correct
# suffix and extension for your data, and you don't need to specify them
# manually.
