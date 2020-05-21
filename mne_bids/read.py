"""Check whether a file format is supported by BIDS and then load it."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
from datetime import datetime
import glob
import json

import numpy as np
import mne
from mne import io
from mne.utils import has_nibabel, logger, warn
from mne.coreg import fit_matched_points
from mne.transforms import apply_trans

from mne_bids.dig import _read_dig_bids
from mne_bids.tsv_handler import _from_tsv, _drop
from mne_bids.config import (ALLOWED_EXTENSIONS, _convert_hand_options,
                             _convert_sex_options)
from mne_bids.utils import (_parse_bids_filename, _extract_landmarks,
                            _find_matching_sidecar, _parse_ext,
                            _get_ch_type_mapping, make_bids_folders,
                            make_bids_basename, _estimate_line_freq,
                            _get_kinds_for_sub)

reader = {'.con': io.read_raw_kit, '.sqd': io.read_raw_kit,
          '.fif': io.read_raw_fif, '.pdf': io.read_raw_bti,
          '.ds': io.read_raw_ctf, '.vhdr': io.read_raw_brainvision,
          '.edf': io.read_raw_edf, '.bdf': io.read_raw_bdf,
          '.set': io.read_raw_eeglab}


def _read_raw(raw_fpath, electrode=None, hsp=None, hpi=None, config=None,
              verbose=None, **kwargs):
    """Read a raw file into MNE, making inferences based on extension."""
    _, ext = _parse_ext(raw_fpath)

    # KIT systems
    if ext in ['.con', '.sqd']:
        raw = io.read_raw_kit(raw_fpath, elp=electrode, hsp=hsp,
                              mrk=hpi, preload=False, **kwargs)

    # BTi systems
    elif ext == '.pdf':
        raw = io.read_raw_bti(raw_fpath, config_fname=config,
                              head_shape_fname=hsp,
                              preload=False, verbose=verbose,
                              **kwargs)

    elif ext == '.fif':
        raw = reader[ext](raw_fpath, **kwargs)

    elif ext in ['.ds', '.vhdr', '.set']:
        raw = reader[ext](raw_fpath, **kwargs)

    # EDF (european data format) or BDF (biosemi) format
    # TODO: integrate with lines above once MNE can read
    # annotations with preload=False
    elif ext in ['.edf', '.bdf']:
        raw = reader[ext](raw_fpath, preload=True, **kwargs)

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


def _handle_participants_reading(participants_fname, raw,
                                 subject, verbose=None):
    participants_tsv = _from_tsv(participants_fname)
    subjects = participants_tsv['participant_id']
    row_ind = subjects.index(subject)

    # set data from participants tsv into subject_info
    for infokey, infovalue in participants_tsv.items():
        if infokey == 'sex':
            value = _convert_sex_options(infovalue[row_ind],
                                         fro='bids', to='mne')
            # We don't know how to translate to MNE, so skip.
            if value is None:
                warn('Unable to map `sex` value to MNE. '
                     'Not setting subject sex.')
        elif infokey == 'hand':
            value = _convert_hand_options(infovalue[row_ind],
                                          fro='bids', to='mne')
            # We don't know how to translate to MNE, so skip.
            if value is None:
                warn('Unable to map `hand` value to MNE. '
                     'Not setting subject handedness.')
        else:
            value = infovalue[row_ind]
        # add data into raw.Info
        if raw.info['subject_info'] is None:
            raw.info['subject_info'] = dict()
        raw.info['subject_info'][infokey] = value

    return raw


def _handle_info_reading(sidecar_fname, raw, verbose=None):
    """Read associated sidecar.json and populate raw.

    Handle PowerLineFrequency of recording.
    """
    with open(sidecar_fname, "r") as fin:
        sidecar_json = json.load(fin)

    # read in the sidecar JSON's line frequency
    line_freq = sidecar_json.get("PowerLineFrequency")
    if line_freq == "n/a":
        line_freq = None

    if line_freq is None and raw.info["line_freq"] is None:
        # estimate line noise using PSD from multitaper FFT
        powerlinefrequency = _estimate_line_freq(raw, verbose=verbose)
        raw.info["line_freq"] = powerlinefrequency
        warn('No line frequency found, defaulting to {} Hz '
             'estimated from multi-taper FFT '
             'on 10 seconds of data.'.format(powerlinefrequency))

    elif raw.info["line_freq"] is None and line_freq is not None:
        # if the read in frequency is not set inside Raw
        # -> set it to what the sidecar JSON specifies
        raw.info["line_freq"] = line_freq
    elif raw.info["line_freq"] is not None \
            and line_freq is not None:
        # if both have a set Power Line Frequency, then
        # check that they are the same, else there is a
        # discrepency in the metadata of the dataset.
        if raw.info["line_freq"] != line_freq:
            raise ValueError("Line frequency in sidecar json does "
                             "not match the info datastructure of "
                             "the mne.Raw. "
                             "Raw is -> {} ".format(raw.info["line_freq"]),
                             "Sidecar JSON is -> {} ".format(line_freq))

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
    dus = [0 if du == 'n/a' else du for du in events_dict['duration']]
    onsets = np.asarray(ons, dtype=float)
    durations = np.asarray(dus, dtype=float)

    # Keep only events where onset is known
    good_events_idx = ~np.isnan(onsets)
    onsets = onsets[good_events_idx]
    durations = durations[good_events_idx]
    descriptions = descriptions[good_events_idx]
    del good_events_idx

    # Add Events to raw as annotations
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


def _infer_kind(*, bids_basename, bids_root, sub, ses):
    # Check which kind is available for this particular
    # subject & session. If we get no or multiple hits, throw an error.

    kinds = _get_kinds_for_sub(bids_basename=bids_basename,
                               bids_root=bids_root, sub=sub, ses=ses)

    # We only want to handle electrophysiological data here.
    allowed_kinds = ['meg', 'eeg', 'ieeg']
    kinds = list(set(kinds) & set(allowed_kinds))
    if not kinds:
        raise ValueError('No electrophysiological data found.')
    elif len(kinds) >= 2:
        msg = (f'Found data of more than one recording modality. Please '
               f'pass the `kind` parameter to specify which data to load. '
               f'Found the following kinds: {kinds}')
        raise RuntimeError(msg)

    assert len(kinds) == 1
    return kinds[0]


def _get_bids_fname_from_filesystem(*, bids_basename, bids_root, sub, ses,
                                    kind):
    if kind is None:
        kind = _infer_kind(bids_basename=bids_basename, bids_root=bids_root,
                           sub=sub, ses=ses)

    data_dir = make_bids_folders(subject=sub, session=ses, kind=kind,
                                 make_dir=False)

    bti_dir = op.join(bids_root, data_dir, f'{bids_basename}_{kind}')
    if op.isdir(bti_dir):
        logger.info(f'Assuming BTi data in {bti_dir}')
        bids_fname = f'{bti_dir}.pdf'
    else:
        # Find all matching files in all supported formats.
        valid_exts = list(reader.keys())
        matching_paths = glob.glob(op.join(bids_root, data_dir,
                                           f'{bids_basename}_{kind}.*'))
        matching_paths = [p for p in matching_paths
                          if _parse_ext(p)[1] in valid_exts]

        if not matching_paths:
            msg = ('Could not locate a data file of a supported format. This '
                   'is likely a problem with your BIDS dataset. Please run '
                   'the BIDS validator on your data.')
            raise RuntimeError(msg)

        # FIXME This will break e.g. with FIFF data split across multiple
        # FIXME files.
        if len(matching_paths) > 1:
            msg = ('Found more than one matching data file for the requested '
                   'recording. Cannot proceed due to the ambiguity. This is '
                   'likely a problem with your BIDS dataset. Please run the '
                   'BIDS validator on your data.')
            raise RuntimeError(msg)

        matching_path = matching_paths[0]
        bids_fname = op.basename(matching_path)

    return bids_fname


def _make_bids_fname(bids_basename, bids_root=None, kind=None, extension=None,
                     verbose=False):
    """Construct the filename of a BIDS data file.

    Parameters
    ----------
    bids_basename : str
        The base filename of the BIDS-compatible files. Typically, this can be
        generated using :func:`mne_bids.make_bids_basename`.
    bids_root : str | pathlib.Path | None
        Path to root of the BIDS folder
    kind : str | None
        The kind of recording to read. If ``None`` and only one kind (e.g.,
        only EEG or only MEG data) is present in the dataset, it will be
        selected automatically.
    extra_params : None | dict
        Extra parameters to be passed to MNE read_raw_* functions.
        If a dict, for example: ``extra_params=dict(allow_maxshield=True)``.
    extension : None | str
        If ``None``, try to infer the filename extension by searching for the
        file on disk. If the file cannot be found, an error will be raised. To
        disable this automatic inference attempt, pass a string (like
        ``'.fif'`` or ``'.vhdr'``). If an empty string is passed, no extension
        will be added to the filename.
    verbose : bool
        The verbosity level.

    """
    # Get the BIDS parameters (=entities)
    params = _parse_bids_filename(bids_basename, verbose)
    sub = params['sub']
    ses = params['ses']

    if extension is None and bids_root is None:
        msg = ('No filename extension was provided, and it cannot be '
               'automatically inferred because no bids_root was passed.')
        raise ValueError(msg)

    if extension is None:
        bids_fname = _get_bids_fname_from_filesystem(
            bids_basename=bids_basename, bids_root=bids_root, sub=sub, ses=ses,
            kind=kind)
    else:
        bids_fname = f'{bids_basename}_{kind}.{extension}'

    return bids_fname


def read_raw_bids(bids_basename, bids_root, kind=None, extra_params=None,
                  verbose=True):
    """Read BIDS compatible data.

    Will attempt to read associated events.tsv and channels.tsv files to
    populate the returned raw object with raw.annotations and raw.info['bads'].

    Parameters
    ----------
    bids_basename : str
        The base filename of the BIDS compatible files. Typically, this can be
        generated using :func:`mne_bids.make_bids_basename`.
    bids_root : str | pathlib.Path
        Path to root of the BIDS folder
    kind : str | None
        The kind of recording to read. If ``None`` and only one kind (e.g.,
        only EEG or only MEG data) is present in the dataset, it will be
        selected automatically.
    extra_params : None | dict
        Extra parameters to be passed to MNE read_raw_* functions.
        If a dict, for example: ``extra_params=dict(allow_maxshield=True)``.
    verbose : bool
        The verbosity level.

    Returns
    -------
    raw : instance of Raw
        The data as MNE-Python Raw object.

    Raises
    ------
    RuntimeError
        If multiple recording kinds are present in the dataset, but
        ``kind=None``.

    RuntimeError
        If more than one data files exist for the specified recording.

    RuntimeError
        If no data file in a supported format can be located.

    ValueError
        If the specified ``kind`` cannot be found in the dataset.

    """
    params = _parse_bids_filename(bids_basename, verbose='warning')
    sub = params['sub']
    ses = params['ses']

    if kind is None:
        kind = _infer_kind(bids_basename=bids_basename, bids_root=bids_root,
                           sub=sub, ses=ses)

    data_dir = make_bids_folders(subject=sub, session=ses, kind=kind,
                                 make_dir=False)
    bids_fname = _make_bids_fname(bids_basename=bids_basename,
                                  bids_root=bids_root, kind=kind)

    if op.splitext(bids_fname)[1] == '.pdf':
        bids_raw_folder = op.join(bids_root, data_dir,
                                  f'{bids_basename}_{kind}')
        bids_fpath = glob.glob(op.join(bids_raw_folder, 'c,rf*'))[0]
        config = op.join(bids_raw_folder, 'config')
    else:
        bids_fpath = op.join(bids_root, data_dir, bids_fname)
        config = None

    if extra_params is None:
        extra_params = dict()
    raw = _read_raw(bids_fpath, electrode=None, hsp=None, hpi=None,
                    config=config, verbose=None, **extra_params)

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

    # Try to find an associated electrodes.tsv and coordsystem.json
    # to get information about the status and type of present channels
    acq = params['acq']
    elec_suffix = 'acq-{}*_electrodes.tsv'.format(acq)
    coord_suffix = 'acq-{}*_coordsystem.json'.format(acq)
    electrodes_fname = _find_matching_sidecar(bids_fname, bids_root,
                                              suffix=elec_suffix,
                                              allow_fail=True)
    coordsystem_fname = _find_matching_sidecar(bids_fname, bids_root,
                                               suffix=coord_suffix,
                                               allow_fail=True)

    if electrodes_fname is not None:
        if coordsystem_fname is None:
            raise RuntimeError("BIDS mandates that the coordsystem.json "
                               "should exist if electrodes.tsv does. "
                               "Please create coordsystem.json for"
                               "{}".format(bids_basename))
        if kind in ['meg', 'eeg', 'ieeg']:
            raw = _read_dig_bids(electrodes_fname, coordsystem_fname,
                                 raw, kind, verbose)

    # Try to find an associated sidecar.json to get information about the
    # recording snapshot
    sidecar_fname = _find_matching_sidecar(bids_fname, bids_root,
                                           '{}.json'.format(kind),
                                           allow_fail=True)
    if sidecar_fname is not None:
        raw = _handle_info_reading(sidecar_fname, raw, verbose=verbose)

    # read in associated subject info from participants.tsv
    participants_tsv_fpath = op.join(bids_root, 'participants.tsv')
    params = _parse_bids_filename(bids_basename, verbose='warning')
    subject = f"sub-{params['sub']}"
    if op.exists(participants_tsv_fpath):
        raw = _handle_participants_reading(participants_tsv_fpath, raw,
                                           subject, verbose=verbose)
    else:
        warn("Participants file not found for {}... Not reading "
             "in any particpants.tsv data.".format(bids_fname))

    return raw


def get_matched_empty_room(bids_basename, bids_root):
    """Get matching empty-room file for an MEG recording.

    Parameters
    ----------
    bids_basename : str
        The base filename of the BIDS-compatible file. Typically, this can be
        generated using :func:`mne_bids.make_bids_basename`.
    bids_root : str | pathlib.Path
        Path to the BIDS root folder.

    Returns
    -------
    er_basename : str | None.
        The basename corresponding to the best-matching empty-room measurement.
        Returns None if none was found.
    """
    kind = 'meg'
    bids_fname = _make_bids_fname(bids_basename=bids_basename,
                                  bids_root=bids_root, kind=kind)
    _, ext = _parse_ext(bids_fname)

    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        kind='meg')
    if raw.info['meas_date'] is None:
        raise ValueError('Measurement date not available. Cannot get matching'
                         ' empty room file')

    ref_date = raw.info['meas_date']
    if not isinstance(ref_date, datetime):
        # for MNE < v0.20
        ref_date = datetime.fromtimestamp(raw.info['meas_date'][0])
    search_path = make_bids_folders(bids_root=bids_root, subject='emptyroom',
                                    session='**', make_dir=False)
    search_path = op.join(search_path, '**', '**%s' % ext)
    er_fnames = glob.glob(search_path)

    best_er_fname = None
    min_seconds = np.inf
    for er_fname in er_fnames:
        params = _parse_bids_filename(er_fname, verbose=False)
        dt = datetime.strptime(params['ses'], '%Y%m%d')
        dt = dt.replace(tzinfo=ref_date.tzinfo)
        delta_t = dt - ref_date
        if abs(delta_t.total_seconds()) < min_seconds:
            min_seconds = abs(delta_t.total_seconds())
            best_er_fname = er_fname

    if best_er_fname is None:
        er_basename = None
    else:
        params = _parse_bids_filename(best_er_fname, verbose='warning')
        er_basename = make_bids_basename(
            subject=params.get('sub', None),
            session=params.get('ses', None),
            task=params.get('task', None),
            acquisition=params.get('acq', None),
            run=params.get('run', None),
            processing=params.get('proc', None),
            recording=params.get('recording', None),
            space=params.get('space', None)
        )

    return er_basename


def get_head_mri_trans(bids_basename, bids_root):
    """Produce transformation matrix from MEG and MRI landmark points.

    Will attempt to read the landmarks of Nasion, LPA, and RPA from the sidecar
    files of (i) the MEG and (ii) the T1 weighted MRI data. The two sets of
    points will then be used to calculate a transformation matrix from head
    coordinates to MRI coordinates.

    Parameters
    ----------
    bids_basename : str
        The base filename of the BIDS-compatible file. Typically, this can be
        generated using :func:`mne_bids.make_bids_basename`.
    bids_root : str | pathlib.Path
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
    bids_fname = _make_bids_fname(bids_basename=bids_basename,
                                  bids_root=bids_root, kind='meg')
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
    _, ext = _parse_ext(bids_fname)
    extra_params = None
    if ext == '.fif':
        extra_params = dict(allow_maxshield=True)

    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        extra_params=extra_params, kind='meg')
    meg_coords_dict = _extract_landmarks(raw.info['dig'])
    meg_landmarks = np.asarray((meg_coords_dict['LPA'],
                                meg_coords_dict['NAS'],
                                meg_coords_dict['RPA']))

    # Given the two sets of points, fit the transform
    trans_fitted = fit_matched_points(src_pts=meg_landmarks,
                                      tgt_pts=mri_landmarks)
    trans = mne.transforms.Transform(fro='head', to='mri', trans=trans_fitted)
    return trans
