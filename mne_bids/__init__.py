"""MNE software for easily interacting with BIDS compatible datasets."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

try:
    from importlib.metadata import version

    __version__ = version("mne_bids")
except Exception:
    __version__ = "0.0.0"

from mne_bids import commands
from mne_bids.report import make_report
from mne_bids.path import (
    BIDSPath,
    get_datatypes,
    get_entity_vals,
    print_dir_tree,
    get_entities_from_fname,
    search_folder_for_text,
    get_bids_path_from_fname,
    find_matching_paths,
)
from mne_bids.read import (
    get_head_mri_trans,
    read_raw_bids,
    events_file_to_annotation_kwargs,
)
from mne_bids.utils import get_anonymization_daysback
from mne_bids.write import (
    make_dataset_description,
    write_anat,
    write_raw_bids,
    mark_channels,
    write_meg_calibration,
    write_meg_crosstalk,
    get_anat_landmarks,
    anonymize_dataset,
)
from mne_bids.sidecar_updates import update_sidecar_json, update_anat_landmarks
from mne_bids.inspect import inspect_dataset
from mne_bids.dig import (
    template_to_head,
    convert_montage_to_ras,
    convert_montage_to_mri,
)
