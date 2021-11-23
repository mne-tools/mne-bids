"""MNE software for easily interacting with BIDS compatible datasets."""

__version__ = '0.9'
from mne_bids import commands
from mne_bids.report import make_report
from mne_bids.path import (BIDSPath, get_datatypes, get_entity_vals,
                           print_dir_tree, get_entities_from_fname,
                           search_folder_for_text, get_bids_path_from_fname)
from mne_bids.read import get_head_mri_trans, read_raw_bids
from mne_bids.utils import get_anonymization_daysback
from mne_bids.write import (make_dataset_description, write_anat,
                            write_raw_bids, mark_channels,
                            write_meg_calibration, write_meg_crosstalk,
                            get_anat_landmarks, anonymize_dataset)
from mne_bids.sidecar_updates import update_sidecar_json, update_anat_landmarks
from mne_bids.inspect import inspect_dataset
