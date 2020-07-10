"""MNE software for easily interacting with BIDS compatible datasets."""

__version__ = '0.5.dev0'
from mne_bids import commands  # noqa: F401
from mne_bids.read import get_head_mri_trans, read_raw_bids
from mne_bids.utils import (get_anonymization_daysback,  # noqa: F401
                            make_bids_basename, make_bids_folders)
from mne_bids.write import make_dataset_description, write_anat

from mne_bids.read import get_matched_empty_room  # noqa: F401; noqa: F401
from mne_bids.write import write_raw_bids  # noqa: F401; noqa: E501 F401
