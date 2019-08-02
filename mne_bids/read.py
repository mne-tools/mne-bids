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
from mne.utils import has_nibabel, logger
from mne.coreg import fit_matched_points
from mne.transforms import apply_trans

from mne_bids.tsv_handler import _from_tsv, _drop
from mne_bids.config import ALLOWED_EXTENSIONS
from mne_bids.utils import (_parse_bids_filename, _extract_landmarks,
                            _find_matching_sidecar, _parse_ext,
                            _get_ch_type_mapping)

reader = {'.con': io.read_raw_kit, '.sqd': io.read_raw_kit,
          '.fif': io.read_raw_fif, '.pdf': io.read_raw_bti,
          '.ds': io.read_raw_ctf, '.vhdr': io.read_raw_brainvision,
          '.edf': io.read_raw_edf, '.bdf': io.read_raw_bdf,
          '.set': io.read_raw_eeglab}


def _read_raw(raw_fpath, electrode=None, hsp=None, hpi=None, config=None,
              montage=None, verbose=None, allow_maxshield=False):
    """Read a raw file into MNE, making inferences based on extension."""
    _, ext = _parse_ext(raw_fpath)

    # KIT systems
    if ext in ['.con', '.sqd']:
        raw = io.read_raw_kit(raw_fpath, elp=electrode, hsp=hsp,
                              mrk=hpi, preload=False)

    # BTi systems
    elif ext == '.pdf':
        raw = io.read_raw_bti(raw_fpath, config_fname=config,
                              head_shape_fname=hsp,
                              preload=False, verbose=verbose)

    elif ext == '.fif':
        raw = reader[ext](raw_fpath, allow_maxshield=allow_maxshield)

    elif ext in ['.ds', '.vhdr', '.set']:
        raw = reader[ext](raw_fpath)

    # EDF (european data format) or BDF (biosemi) format
    # TODO: integrate with lines above once MNE can read
    # annotations with preload=False
    elif ext in ['.edf', '.bdf']:
        raw = reader[ext](raw_fpath, preload=True)

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


def _handle_events_reading(events_fname, raw):
    """Read associated events.tsv and populate raw.

    Handle onset, duration, and description of each event.
    """
    logger.info('Reading events from {}.'.format(events_fname))
    events_dict = _from_tsv(events_fname)

    # Get the descriptions of the events
    if 'trial_type' in events_dict:
        # Drop events unrelated to a trial type
        events_dict = _drop(events_dict, 'n/a', 'trial_type')
        descriptions = np.asarray(events_dict['trial_type'], dtype=str)

    # If we don't have a proper description of the events, perhaps we have
    # at least an event value?
    elif 'value' in events_dict:
        # Drop events unrelated to value
        events_dict = _drop(events_dict, 'n/a', 'value')
        descriptions = np.asarray(events_dict['value'], dtype=str)

    # Worst case, we go with 'n/a' for all events
    else:
        descriptions = 'n/a'

    # Deal with "n/a" strings before converting to float
    ons = [np.nan if on == 'n/a' else on for on in events_dict['onset']]
    dus = [np.nan if du == 'n/a' else du for du in events_dict['duration']]

    # Add Events to raw as annotations
    onsets = np.asarray(ons, dtype=float)
    durations = np.asarray(dus, dtype=float)
    annot_from_events = mne.Annotations(onset=onsets,
                                        duration=durations,
                                        description=descriptions,
                                        orig_time=None)
    raw.set_annotations(annot_from_events)

    return raw


def _handle_channels_reading(channels_fname, bids_fname, raw):
    """Read associated channels.tsv and populate raw.

    Updates status (bad) and types of channels.
    """
    logger.info('Reading channel info from {}.'.format(channels_fname))
    channels_dict = _from_tsv(channels_fname)

    # First, make sure that ordering of names in channels.tsv matches the
    # ordering of names in the raw data. The "name" column is mandatory in BIDS
    ch_names_raw = list(raw.ch_names)
    ch_names_tsv = channels_dict['name']
    if ch_names_raw != ch_names_tsv:

        msg = ('Channels do not correspond between raw data and the '
               'channels.tsv file. For MNE-BIDS, the channel names in the '
               'tsv MUST be equal and in the same order as the channels in '
               'the raw data.\n\n'
               '{} channels in tsv file: "{}"\n\n --> {}\n\n'
               '{} channels in raw file: "{}"\n\n --> {}\n\n'
               .format(len(ch_names_tsv), channels_fname, ch_names_tsv,
                       len(ch_names_raw), bids_fname, ch_names_raw)
               )

        # XXX: this could be due to MNE inserting a 'STI 014' channel as the
        # last channel: In that case, we can work. --> Can be removed soon,
        # because MNE will stop the synthesis of stim channels in the near
        # future
        if not (ch_names_raw[-1] == 'STI 014' and
                ch_names_raw[:-1] == ch_names_tsv):
            raise RuntimeError(msg)

    # Now we can do some work.
    # The "type" column is mandatory in BIDS. We can use it to set channel
    # types in the raw data using a mapping between channel types
    channel_type_dict = dict()

    # Get the best mapping we currently have from BIDS to MNE nomenclature
    bids_to_mne_ch_types = _get_ch_type_mapping(fro='bids', to='mne')
    ch_types_json = channels_dict['type']
    for ch_name, ch_type in zip(ch_names_tsv, ch_types_json):

        # Try to map from BIDS nomenclature to MNE, leave channel type
        # untouched if we are uncertain
        updated_ch_type = bids_to_mne_ch_types.get(ch_type, None)
        if updated_ch_type is not None:
            channel_type_dict[ch_name] = updated_ch_type

    # Set the channel types in the raw data according to channels.tsv
    raw.set_channel_types(channel_type_dict)

    # Check whether there is the optional "status" column from which to infer
    # good and bad channels
    if 'status' in channels_dict:
        # find bads from channels.tsv
        bad_bool = [True if chn.lower() == 'bad' else False
                    for chn in channels_dict['status']]
        bads = np.asarray(channels_dict['name'])[bad_bool]

        # merge with bads already present in raw data file (if there are any)
        unique_bads = set(raw.info['bads']).union(set(bads))
        raw.info['bads'] = list(unique_bads)

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
    bids_fname = op.basename(bids_fname)
    bids_basename = '_'.join(bids_fname.split('_')[:-1])
    kind = bids_fname.split('_')[-1].split('.')[0]
    _, ext = _parse_ext(bids_fname)

    # Get the BIDS parameters (=entities)
    params = _parse_bids_filename(bids_basename, verbose)

    # Construct the path to the "kind" where the data is stored
    # Subject is mandatory ...
    kind_dir = op.join(bids_root, 'sub-{}'.format(params['sub']))
    # Session is optional ...
    if params['ses'] is not None:
        kind_dir = op.join(kind_dir, 'ses-{}'.format(params['ses']))
    # Kind is mandatory
    kind_dir = op.join(kind_dir, kind)

    config = None
    if ext in ('.fif', '.ds', '.vhdr', '.edf', '.bdf', '.set', '.sqd', '.con'):
        bids_fpath = op.join(kind_dir,
                             bids_basename + '_{}{}'.format(kind, ext))

    elif ext == '.pdf':
        bids_raw_folder = op.join(kind_dir, bids_basename + '_{}'.format(kind))
        bids_fpath = glob.glob(op.join(bids_raw_folder, 'c,rf*'))[0]
        config = op.join(bids_raw_folder, 'config')

    raw = _read_raw(bids_fpath, electrode=None, hsp=None, hpi=None,
                    config=config, montage=None, verbose=None)

    # Try to find an associated events.tsv to get information about the
    # events in the recorded data
    events_fname = _find_matching_sidecar(bids_fname, bids_root, 'events.tsv',
                                          allow_fail=True)
    if events_fname is not None:
        raw = _handle_events_reading(events_fname, raw)

    # Try to find an associated channels.tsv to get information about the
    # status and type of present channels
    channels_fname = _find_matching_sidecar(bids_fname, bids_root,
                                            'channels.tsv', allow_fail=True)
    if channels_fname is not None:
        raw = _handle_channels_reading(channels_fname, bids_fname, raw)

    return raw


def get_head_mri_trans(bids_fname, bids_root):
    """Produce transformation matrix from MEG and MRI landmark points.

    Will attempt to read the landmarks of Nasion, LPA, and RPA from the sidecar
    files of (i) the MEG and (ii) the T1 weighted MRI data. The two sets of
    points will then be used to calculate a transformation matrix from head
    coordinates to MRI coordinates.

    Parameters
    ----------
    bids_fname : str
        Full name of the MEG data file
    bids_root : str
        Path to root of the BIDS folder

    Returns
    -------
    trans : instance of mne.transforms.Transform
        The data transformation matrix from head to MRI coordinates

    """
    if not has_nibabel():  # pragma: no cover
        raise ImportError('This function requires nibabel.')
    import nibabel as nib

    # Get the sidecar file for MRI landmarks
    bids_fname = op.basename(bids_fname)
    t1w_json_path = _find_matching_sidecar(bids_fname, bids_root, 'T1w.json')

    # Get MRI landmarks from the JSON sidecar
    with open(t1w_json_path, 'r') as f:
        t1w_json = json.load(f)
    mri_coords_dict = t1w_json.get('AnatomicalLandmarkCoordinates', dict())
    mri_landmarks = np.asarray((mri_coords_dict.get('LPA', np.nan),
                                mri_coords_dict.get('NAS', np.nan),
                                mri_coords_dict.get('RPA', np.nan)))
    if np.isnan(mri_landmarks).any():
        raise RuntimeError('Could not parse T1w sidecar file: "{}"\n\n'
                           'The sidecar file MUST contain a key '
                           '"AnatomicalLandmarkCoordinates" pointing to a '
                           'dict with keys "LPA", "NAS", "RPA". '
                           'Yet, the following structure was found:\n\n"{}"'
                           .format(t1w_json_path, t1w_json))

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
