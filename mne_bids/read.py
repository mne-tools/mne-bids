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
import pathlib

import numpy as np
import mne
from mne import io
from mne.utils import has_nibabel, logger, warn
from mne.coreg import fit_matched_points
from mne.transforms import apply_trans

from mne_bids.dig import _read_dig_bids
from mne_bids.tsv_handler import _from_tsv, _drop
from mne_bids.config import (ALLOWED_EXTENSIONS, _convert_hand_options,
                             _convert_sex_options, reader)
from mne_bids.utils import (_parse_bids_filename, _extract_landmarks,
                            _find_matching_sidecar, _parse_ext,
                            _get_ch_type_mapping, make_bids_folders,
                            BIDSPath,
                            _estimate_line_freq, _infer_kind)


def _read_raw(raw_fpath, electrode=None, hsp=None, hpi=None, config=None,
              verbose=None, allow_maxshield=False, **kwargs):
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
        raw = reader[ext](raw_fpath, allow_maxshield, **kwargs)

    elif ext in ['.ds', '.vhdr', '.set', '.edf', '.bdf']:
        raw = reader[ext](raw_fpath, **kwargs)

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


def read_raw_bids(bids_basename, bids_root, kind=None, extra_params=None,
                  verbose=True):
    """Read BIDS compatible data.

    Will attempt to read associated events.tsv and channels.tsv files to
    populate the returned raw object with raw.annotations and raw.info['bads'].

    Parameters
    ----------
    bids_basename : str | BIDSPath
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
    # convert to BIDS Path
    if isinstance(bids_basename, str):
        params = _parse_bids_filename(bids_basename, verbose)
        bids_basename = BIDSPath(subject=params.get('sub'),
                                 session=params.get('ses'),
                                 recording=params.get('rec'),
                                 acquisition=params.get('acq'),
                                 processing=params.get('proc'),
                                 space=params.get('space'),
                                 run=params.get('run'),
                                 task=params.get('task'))
    sub = bids_basename.subject
    ses = bids_basename.session
    acq = bids_basename.acquisition

    if kind is None:
        kind = _infer_kind(bids_basename=bids_basename, bids_root=bids_root,
                           sub=sub, ses=ses)

    data_dir = make_bids_folders(subject=sub, session=ses, kind=kind,
                                 make_dir=False)
    bids_fname = bids_basename.get_bids_fname(kind=kind, bids_root=bids_root)

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
    search_modifier = f'acq-{acq}' if acq else ''
    elec_suffix = f'{search_modifier}*_electrodes.tsv'
    coord_suffix = f'{search_modifier}*_coordsystem.json'
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
    subject = f"sub-{bids_basename.subject}"
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
    bids_basename : str | BIDSPath
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
    # convert to BIDS Path
    if isinstance(bids_basename, str):
        params = _parse_bids_filename(bids_basename, False)
        bids_basename = BIDSPath(subject=params.get('sub'),
                                 session=params.get('ses'),
                                 recording=params.get('rec'),
                                 acquisition=params.get('acq'),
                                 processing=params.get('proc'),
                                 space=params.get('space'),
                                 run=params.get('run'),
                                 task=params.get('task'))

    kind = 'meg'  # We're only concerned about MEG data here
    bids_fname = bids_basename.get_bids_fname(kind=kind, bids_root=bids_root)
    _, ext = _parse_ext(bids_fname)
    if ext == '.fif':
        extra_params = dict(allow_maxshield=True)
    else:
        extra_params = None

    raw = read_raw_bids(bids_basename=bids_basename, bids_root=bids_root,
                        kind=kind, extra_params=extra_params)
    if raw.info['meas_date'] is None:
        raise ValueError('The provided recording does not have a measurement '
                         'date set. Cannot get matching empty-room file.')

    ref_date = raw.info['meas_date']
    if not isinstance(ref_date, datetime):
        # for MNE < v0.20
        ref_date = datetime.fromtimestamp(raw.info['meas_date'][0])

    emptyroom_dir = pathlib.Path(make_bids_folders(bids_root=bids_root,
                                                   subject='emptyroom',
                                                   make_dir=False))

    if not emptyroom_dir.exists():
        return None

    # Find the empty-room recording sessions.
    emptyroom_session_dirs = [x for x in emptyroom_dir.iterdir()
                              if x.is_dir() and str(x.name).startswith('ses-')]
    if not emptyroom_session_dirs:  # No session sub-directories found
        emptyroom_session_dirs = [emptyroom_dir]

    # Now try to discover all recordings inside the session directories.

    allowed_extensions = list(reader.keys())
    # `.pdf` is just a "virtual" extension for BTi data (which is stored inside
    # a dedicated directory that doesn't have an extension)
    del allowed_extensions[allowed_extensions.index('.pdf')]

    candidate_er_fnames = []
    for session_dir in emptyroom_session_dirs:
        dir_contents = glob.glob(op.join(session_dir, kind,
                                         f'sub-emptyroom_*_{kind}*'))
        for item in dir_contents:
            item = pathlib.Path(item)
            if ((item.suffix in allowed_extensions) or
                    (not item.suffix and item.is_dir())):  # Hopefully BTi?
                candidate_er_fnames.append(item.name)

    # Walk through recordings, trying to extract the recording date:
    # First, from the filename; and if that fails, from `info['meas_date']`.
    best_er_basename = None
    min_delta_t = np.inf
    date_tie = False

    failed_to_get_er_date_count = 0
    for er_fname in candidate_er_fnames:
        params = _parse_bids_filename(er_fname, verbose=False)
        er_meas_date = None

        er_bids_path = BIDSPath(subject='emptyroom',
                                session=params.get('ses', None),
                                task=params.get('task', None),
                                acquisition=params.get('acq', None),
                                run=params.get('run', None),
                                processing=params.get('proc', None),
                                recording=params.get('rec', None),
                                space=params.get('space', None))
        er_basename = str(er_bids_path)

        # Try to extract date from filename.
        if params['ses'] is not None:
            try:
                er_meas_date = datetime.strptime(params['ses'], '%Y%m%d')
            except (ValueError, TypeError):
                # There is a session in the filename, but it doesn't encode a
                # valid date.
                pass

        if er_meas_date is None:  # No luck so far! Check info['meas_date']
            _, ext = _parse_ext(er_fname)
            if ext == '.fif':
                extra_params = dict(allow_maxshield=True)
            else:
                extra_params = None

            er_raw = read_raw_bids(bids_basename=er_basename,
                                   bids_root=bids_root,
                                   kind=kind,
                                   extra_params=extra_params)

            er_meas_date = er_raw.info['meas_date']
            if er_meas_date is None:  # There's nothing we can do.
                failed_to_get_er_date_count += 1
                continue

        er_meas_date = er_meas_date.replace(tzinfo=ref_date.tzinfo)
        delta_t = er_meas_date - ref_date

        if abs(delta_t.total_seconds()) == min_delta_t:
            date_tie = True
        elif abs(delta_t.total_seconds()) < min_delta_t:
            min_delta_t = abs(delta_t.total_seconds())
            best_er_basename = er_basename
            date_tie = False

    if failed_to_get_er_date_count > 0:
        msg = (f'Could not retrieve the empty-room measurement date from '
               f'a total of {failed_to_get_er_date_count} recording(s).')
        warn(msg)

    if date_tie:
        msg = ('Found more than one matching empty-room measurement with the '
               'same recording date. Selecting the first match.')
        warn(msg)

    return best_er_basename


def get_head_mri_trans(bids_basename, bids_root):
    """Produce transformation matrix from MEG and MRI landmark points.

    Will attempt to read the landmarks of Nasion, LPA, and RPA from the sidecar
    files of (i) the MEG and (ii) the T1 weighted MRI data. The two sets of
    points will then be used to calculate a transformation matrix from head
    coordinates to MRI coordinates.

    Parameters
    ----------
    bids_basename : str | BIDSPath
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

    # convert to BIDS Path
    if isinstance(bids_basename, str):
        params = _parse_bids_filename(bids_basename, False)
        bids_basename = BIDSPath(subject=params.get('sub'),
                                 session=params.get('ses'),
                                 recording=params.get('rec'),
                                 acquisition=params.get('acq'),
                                 processing=params.get('proc'),
                                 space=params.get('space'),
                                 run=params.get('run'),
                                 task=params.get('task'))

    # Get the sidecar file for MRI landmarks
    bids_fname = bids_basename.get_bids_fname(kind='meg', bids_root=bids_root)
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
    _, ext = _parse_ext(bids_basename)
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
