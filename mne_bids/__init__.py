"""MNE software for easily interacting with BIDS compatible datasets."""

__version__ = '0.3'


from mne_bids.write import (write_raw_bids, make_bids_folders, make_bids_basename,  # noqa: E501 F401
                            make_dataset_description, write_anat)  # noqa: F401
from mne_bids.read import read_raw_bids, get_head_mri_trans  # noqa: F401
