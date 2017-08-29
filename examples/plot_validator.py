import os.path as op

from mne_bids.validator import validate_meg
from mne.datasets import sample

data_path = sample.data_path()
json_fname = op.join('.', 'sub-01_task-audiovisual_meg.json')

validate_meg(json_fname)
