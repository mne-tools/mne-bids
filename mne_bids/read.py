"""Check whether a file format is supported by BIDS and then load it."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import glob
import json

import numpy as np
import mne
from mne import io
from mne.coreg import fit_matched_points
from mne.transforms import apply_trans

from .tsv_handler import _from_tsv, _drop
from .config import ALLOWED_EXTENSIONS
from .utils import (_parse_bids_filename, _extract_landmarks,
                    _find_matching_sidecar, _parse_ext)

reader = {'.con': io.read_raw_kit, '.sqd': io.read_raw_kit,
          '.fif': io.read_raw_fif, '.pdf': io.read_raw_bti,
          '.ds': io.read_raw_ctf, '.vhdr': io.read_raw_brainvision,
          '.edf': io.read_raw_edf, '.bdf': io.read_raw_edf,
          '.set': io.read_raw_eeglab}


def _read_raw(raw_fname, electrode=None, hsp=None, hpi=None, config=None,
              montage=None, verbose=None, allow_maxshield=False):
    """Read a raw file into MNE, making inferences based on extension."""
    fname, ext = _parse_ext(raw_fname)

    # KIT systems
    if ext in ['.con', '.sqd']:
        raw = io.read_raw_kit(raw_fname, elp=electrode, hsp=hsp,
                              mrk=hpi, preload=False)

    # BTi systems
    elif ext == '.pdf':
        raw = io.read_raw_bti(raw_fname, config_fname=config,
                              head_shape_fname=hsp,
                              preload=False, verbose=verbose)

    elif ext == '.fif':
        raw = reader[ext](raw_fname, allow_maxshield=allow_maxshield)

    elif ext in ['.ds', '.vhdr', '.set']:
        raw = reader[ext](raw_fname)

    # EDF (european data format) or BDF (biosemi) format
    # TODO: integrate with lines above once MNE can read
    # annotations with preload=False
    elif ext in ['.edf', '.bdf']:
        raw = reader[ext](raw_fname, preload=True)

    # MEF and NWB are allowed, but not yet implemented
    elif ext in ['.mef', '.nwb']:
        raise ValueError('Got "{}" as extension. This is an allowed extension '
                         'but there is no IO support for this file format yet.'
                         .format(ext))

    # No supported data found ...
    # ---------------------------
    else:
        raise ValueError('Raw file name extension must be one of {}\n'
                         'Got {}'.format(ALLOWED_EXTENSIONS, ext))
    return raw


def read_raw_bids(bids_fname, bids_root, verbose=True):
    """Read BIDS compatible data.

    Will attempt to read associated events.tsv and channels.tsv files to
    populate the returned raw object with raw.annotations and raw.info['bads'].

    Parameters
    ----------
    bids_fname : str
        Full name of the data file
    bids_root : str
        Path to root of the BIDS folder
    verbose : bool
        The verbosity level

    Returns
    -------
    raw : instance of Raw
        The data as MNE-Python Raw object.

    """
    # Full path to data file is needed so that mne-bids knows
    # what is the modality -- meg, eeg, ieeg to read
    bids_basename = '_'.join(bids_fname.split('_')[:-1])
    kind = bids_fname.split('_')[-1].split('.')[0]
    _, ext = _parse_ext(bids_fname)

    params = _parse_bids_filename(bids_basename, verbose)
    kind_dir = op.join(bids_root, 'sub-%s' % params['sub'],
                       'ses-%s' % params['ses'], kind)

    config = None
    if ext in ('.fif', '.ds', '.vhdr', '.edf', '.bdf', '.set', '.sqd', '.con'):
        bids_fname = op.join(kind_dir,
                             bids_basename + '_%s%s' % (kind, ext))

    elif ext == '.pdf':
        bids_raw_folder = op.join(kind_dir, bids_basename + '_%s' % kind)
        bids_fname = glob.glob(op.join(bids_raw_folder, 'c,rf*'))[0]
        config = op.join(bids_raw_folder, 'config')

    raw = _read_raw(bids_fname, electrode=None, hsp=None, hpi=None,
                    config=config, montage=None, verbose=None)

    # Try to find an associated events.tsv to get information about the
    # events in the recorded data
    events_fname = op.join(kind_dir, bids_basename + '_events.tsv')
    if op.exists(events_fname):
        events_dict = _from_tsv(events_fname)
    else:
        events_dict = dict()

    if 'trial_type' in events_dict:
        # Drop events unrelated to a trial type
        events_dict = _drop(events_dict, 'n/a', 'trial_type')

        # Add Events to raw as annotations
        onsets = np.asarray(events_dict['onset'], dtype=float)
        durations = np.asarray(events_dict['duration'], dtype=float)
        descriptions = np.asarray(events_dict['trial_type'], dtype=str)
        annot_from_events = mne.Annotations(onset=onsets,
                                            duration=durations,
                                            description=descriptions,
                                            orig_time=raw.info['meas_date'])
        raw.set_annotations(annot_from_events)

    # Try to find an associated channels.tsv to get information about the
    # status of present channels
    channels_fname = op.join(kind_dir, bids_basename + '_channels.tsv')
    if op.exists(channels_fname):
        channels_dict = _from_tsv(channels_fname)
    else:
        channels_dict = dict()

    # If we have a channels.tsv file, make sure there is the optional "status"
    # column from which to infer good and bad channels
    if 'status' in channels_dict:
        # find bads from channels.tsv
        bad_bool = [True if chn == 'bad' else False
                    for chn in channels_dict['status']]
        bads = np.asarray(channels_dict['name'])[bad_bool]

        # merge with bads already present in raw data file (if there are any)
        unique_bads = set(raw.info['bads']).union(set(bads))
        raw.info['bads'] = list(unique_bads)

    return raw


def get_head_mri_trans(bids_fname, bids_root):
    """Produce transformation matrix from MEG and MRI landmark points.

    Will attempt to read the landmarks of Nasion, LPA, and RPA from the sidecar
    files of (i) the MEG and (ii) the T1 weighted MRI data. The two sets of
    points will then be used to calculate a transformation matrix from HEAD
    coordinates to MRI coordinates.

    Parameters
    ----------
    bids_fname : str
        Full name of the MEG data file (not a path)
    bids_root : str
        Path to root of the BIDS folder

    Returns
    -------
    trans : instance of mne.transforms.Transform
        The data transformation matrix from HEAD to MRI coordinates

    """
    try:  # pragma: no cover
        import nibabel as nib
    except ImportError:  # pragma: no cover
        raise ImportError('This function requires nibabel.')

    # Get the sidecar file for MRI landmarks
    t1w_json_path = _find_matching_sidecar(bids_fname, bids_root, 'T1w.json')

    # Get MRI landmarks from the JSON sidecar
    with open(t1w_json_path, 'r') as f:
        t1w_json = json.load(f)
    mri_coords_dict = t1w_json['AnatomicalLandmarkCoordinates']
    mri_landmarks = np.asarray((mri_coords_dict['LPA'],
                                mri_coords_dict['NAS'],
                                mri_coords_dict['RPA']))

    # The MRI landmarks are in "voxels". We need to convert the to the
    # neuromag RAS coordinate system in order to compare the with MEG landmarks
    # see also: `mne_bids.write.write_anat`
    t1w_path = t1w_json_path.replace('.json', '.nii')
    if not op.exists(t1w_path):
        t1w_path += '.gz'  # perhaps it is .nii.gz? ... else raise an error
    if not op.exists(t1w_path):
        raise RuntimeError('Could not find the T1 weighted MRI associated '
                           'with "{}". Tried: "{}" but it does not exist.'
                           .format(t1w_json_path, t1w_path))
    t1_nifti = nib.load(t1w_path)
    # Convert to MGH format to access vox2ras method
    t1_mgh = nib.MGHImage(t1_nifti.dataobj, t1_nifti.affine)

    # now extract transformation matrix and put back to RAS coordinates of MRI
    vox2ras_tkr = t1_mgh.header.get_vox2ras_tkr()
    mri_landmarks = apply_trans(vox2ras_tkr, mri_landmarks)
    mri_landmarks = mri_landmarks * 1e-3

    # Get MEG landmarks from the raw file
    raw = read_raw_bids(bids_fname, bids_root)
    meg_coords_dict = _extract_landmarks(raw.info['dig'])
    meg_landmarks = np.asarray((meg_coords_dict['LPA'],
                                meg_coords_dict['NAS'],
                                meg_coords_dict['RPA']))

    # Given the two sets of points, fit the transform
    trans_fitted = fit_matched_points(src_pts=meg_landmarks,
                                      tgt_pts=mri_landmarks)
    trans = mne.transforms.Transform(fro='head', to='mri', trans=trans_fitted)
    return trans
