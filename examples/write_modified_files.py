"""
========================================
13. Writing modified files with MNE-BIDS
========================================

MNE-BIDS is designed such that it enforces good practices when working with
BIDS data. One of the principles of creating BIDS datasets from raw data is
that the raw data should ideally be written unmodified, as-is. To enforce
this, :func:`mne_bids.write_raw_bids` performs some basic checks and will
throw an exception if it believes you're doing something that you really
shouldn't be doing (i.e., trying to store modified "raw" data as a BIDS
raw data set.)

There might be some – rare! – situations, however, when working around this
intentional limitation in MNE-BIDS can be warranted. For example, you might
encounter data that has manually been split across multiple files during
recording, even though it belongs to a single experimental run. In this case,
you might want to concatenate the data before storing them in BIDS. This
tutorial will give you an example on how to use :func:`mne_bids.write_raw_bids`
to store such data, despite it being modified before writing.

.. warning:: Please be aware that the situations in which you would need
             to apply the following solution are **extremely** rare. If you
             ever find yourself wanting to apply this solution, please take a
             step back, take a deep breath and re-consider whether this is
             **absolutely** necessary. If even a slight doubt remains,
             reach out to the MNE-BIDS developers.

"""

# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
# License: BSD (3-clause)

###############################################################################
# Load the ``sample`` dataset, and create a concatenated raw data object.

from pathlib import Path
from tempfile import NamedTemporaryFile

import mne
from mne.datasets import sample

from mne_bids import write_raw_bids, BIDSPath


data_path = Path(sample.data_path())
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
output_path = data_path / '..' / 'MNE-sample-data-bids'
bids_path = BIDSPath(subject='01', task='audiovisual', root=output_path)

raw = mne.io.read_raw_fif(raw_fname)
raw.info['line_freq'] = 60
raw_concat = mne.concatenate_raws([raw.copy(), raw])

###############################################################################
# Trying to write these data will fail.

try:
    write_raw_bids(raw=raw_concat, bids_path=bids_path, overwrite=True)
except ValueError as e:
    print(f'Data cannot be written. Exception message was: {e}')

###############################################################################
# We can work around this limitation by first writing the modified data to
# a temporary file, reading it back in, and then writing it via MNE-BIDS.

with NamedTemporaryFile(suffix='_raw.fif') as f:
    fname = f.name
    raw_concat.save(fname, overwrite=True)
    raw_concat = mne.io.read_raw_fif(fname, preload=False)
    write_raw_bids(raw=raw_concat, bids_path=bids_path, overwrite=True)

###############################################################################
# That's it!
#
# .. warning:: **Remember, this should only ever be a last resort!**
#
