import os.path as op

from mne_bids.validator import validate_meg

# test an _meg.json file:
json_fname = op.join('.', 'sub-01_task-audiovisual_meg.json')

validate_meg(json_fname)

# test an _fid.json file:
json_fname = op.join('.', 'sub-testme01_meg_fid.json')

validate_meg(json_fname)
