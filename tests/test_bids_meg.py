# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import os.path as op

import mne
from mne.datasets import sample
from mne.utils import _TempDir

from mne_bids import raw_to_bids

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')


def test_bids_meg():
    """Smoke test BIDS converter."""
    output_path = _TempDir()
    raw_to_bids(subject_id='01', run='01', task='audiovisual',
                raw_fname=raw_fname, events_fname=events_fname,
                output_path=output_path, event_id=event_id,
                overwrite=True)

    # let's do some modifications to meas_date
    raw = mne.io.read_raw_fif(raw_fname)
    raw.info['meas_date'] = None
    data_path2 = _TempDir()
    raw_fname2 = op.join(data_path2, 'sample_audvis_raw.fif')
    raw.save(raw_fname2)

    raw_to_bids(subject_id='01', run='01', task='audiovisual',
                raw_fname=raw_fname2, events_fname=events_fname,
                output_path=output_path, event_id=event_id,
                overwrite=True)
