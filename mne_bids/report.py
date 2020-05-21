"""Make BIDS report from dataset and sidecar files."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import os
import os.path as op
from datetime import datetime, date, timedelta, timezone
import shutil as sh
from collections import defaultdict, OrderedDict

import numpy as np
from scipy import linalg
from numpy.testing import assert_array_equal
from mne.transforms import (_get_trans, apply_trans, get_ras_to_neuromag_trans,
                            rotation, translation)
from mne import Epochs
from mne.io.constants import FIFF
from mne.io.pick import channel_type
from mne.io import BaseRaw, anonymize_info, read_fiducials
try:
    from mne.io._digitization import _get_fid_coords
except ImportError:
    from mne._digitization._utils import _get_fid_coords
from mne.channels.channels import _unit2human
from mne.utils import check_version, has_nibabel, _check_ch_locs, logger, warn

from mne_bids.pick import coil_type
from mne_bids.utils import (_write_json, _write_tsv, _read_events, _mkdir_p,
                            _age_on_date, _infer_eeg_placement_scheme,
                            _check_key_val,
                            _parse_bids_filename, _handle_kind, _check_types,
                            _path_to_str,
                            _extract_landmarks, _parse_ext,
                            _get_ch_type_mapping, make_bids_folders,
                            _estimate_line_freq)
from mne_bids.copyfiles import (copyfile_brainvision, copyfile_eeglab,
                                copyfile_ctf, copyfile_bti, copyfile_kit)
from mne_bids.read import reader
from mne_bids.tsv_handler import _from_tsv, _combine, _drop, _contains_row

from mne_bids.config import (ORIENTATION, UNITS, MANUFACTURERS,
                             IGNORED_CHANNELS, ALLOWED_EXTENSIONS,
                             BIDS_VERSION, MNE_VERBOSE_IEEG_COORD_FRAME,
                             _convert_hand_options, _convert_sex_options)

def create_methods_paragraph(bids_root, template, verbose=True):
    params = _parse_bids_filename(bids_basename, verbose='warning')
    sub = params['sub']
    ses = params['ses']

    if kind is None:
        kind = _infer_kind(bids_basename=bids_basename, bids_root=bids_root,
                           sub=sub, ses=ses)

    data_dir = make_bids_folders(subject=sub, session=ses, kind=kind,
                                 make_dir=False)

    # read in associated subject info from participants.tsv
    participants_tsv_fpath = op.join(bids_root, 'participants.tsv')
    if op.exists(participants_tsv_fpath):
        participants_dict = _parse_participants_file(participants_tsv_fpath, verbose=verbose)


def _generate_session_report():
    pass

def _generate_subject_report():
    pass

def _parse_sidecars(session_path, verbose=True):
    pass

def _parse_sidecar_run(sidecar_fpath, verbose=True):
    pass

def _parse_channels_run(channels_fpath, verbose=True):
    pass


def _parse_scans_file(scans_tsv_fpath, verbose=True):
    pass

def _parse_participants_file(participants_tsv_fpath, verbose=True):
    pass