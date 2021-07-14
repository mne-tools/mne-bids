"""Make BIDS compatible directory structures and infer meta data from MNE."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)
import json
import sys
import os
import os.path as op
from pathlib import Path
from datetime import datetime, timezone, timedelta
import shutil
from collections import defaultdict, OrderedDict

from pkg_resources import parse_version

import numpy as np
from scipy import linalg
import mne
from mne.transforms import (_get_trans, apply_trans, rotation, translation)
from mne import Epochs
from mne.io.constants import FIFF
from mne.io.pick import channel_type
from mne.io import BaseRaw, read_fiducials
from mne.channels.channels import _unit2human
from mne.utils import (check_version, has_nibabel, logger, warn, Bunch,
                       _validate_type, get_subjects_dir)
import mne.preprocessing

from mne_bids.pick import coil_type
from mne_bids.dig import _write_dig_bids, _write_coordsystem_json
from mne_bids.utils import (_write_json, _write_tsv, _write_text,
                            _age_on_date, _infer_eeg_placement_scheme,
                            _get_ch_type_mapping, _check_anonymize,
                            _stamp_to_dt, _handle_datatype)
from mne_bids import BIDSPath
from mne_bids.path import _parse_ext, _mkdir_p, _path_to_str
from mne_bids.copyfiles import (copyfile_brainvision, copyfile_eeglab,
                                copyfile_ctf, copyfile_bti, copyfile_kit,
                                copyfile_edf)
from mne_bids.tsv_handler import (_from_tsv, _drop, _contains_row,
                                  _combine_rows)
from mne_bids.read import _find_matching_sidecar, _read_events

from mne_bids.config import (ORIENTATION, UNITS, MANUFACTURERS,
                             IGNORED_CHANNELS, ALLOWED_DATATYPE_EXTENSIONS,
                             BIDS_VERSION, REFERENCES, _map_options, reader,
                             ALLOWED_INPUT_EXTENSIONS, CONVERT_FORMATS)


def _is_numeric(n):
    return isinstance(n, (np.integer, np.floating, int, float))


def _channels_tsv(raw, fname, overwrite=False, verbose=True):
    """Create a channels.tsv file and save it.

    Parameters
    ----------
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    fname : str | mne_bids.BIDSPath
        Filename to save the channels.tsv to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to True or False.

    """
    # Get channel type mappings between BIDS and MNE nomenclatures
    map_chs = _get_ch_type_mapping(fro='mne', to='bids')

    # Prepare the descriptions for each channel type
    map_desc = defaultdict(lambda: 'Other type of channel')
    map_desc.update(meggradaxial='Axial Gradiometer',
                    megrefgradaxial='Axial Gradiometer Reference',
                    meggradplanar='Planar Gradiometer',
                    megmag='Magnetometer',
                    megrefmag='Magnetometer Reference',
                    stim='Trigger',
                    eeg='ElectroEncephaloGram',
                    ecog='Electrocorticography',
                    seeg='StereoEEG',
                    ecg='ElectroCardioGram',
                    eog='ElectroOculoGram',
                    emg='ElectroMyoGram',
                    misc='Miscellaneous',
                    bio='Biological',
                    ias='Internal Active Shielding',
                    dbs='Deep Brain Stimulation')
    get_specific = ('mag', 'ref_meg', 'grad')

    # get the manufacturer from the file in the Raw object
    _, ext = _parse_ext(raw.filenames[0], verbose=verbose)
    manufacturer = MANUFACTURERS[ext]

    ignored_channels = IGNORED_CHANNELS.get(manufacturer, list())

    status, ch_type, description = list(), list(), list()
    for idx, ch in enumerate(raw.info['ch_names']):
        status.append('bad' if ch in raw.info['bads'] else 'good')
        _channel_type = channel_type(raw.info, idx)
        if _channel_type in get_specific:
            _channel_type = coil_type(raw.info, idx, _channel_type)
        ch_type.append(map_chs[_channel_type])
        description.append(map_desc[_channel_type])
    low_cutoff, high_cutoff = (raw.info['highpass'], raw.info['lowpass'])
    if raw._orig_units:
        units = [raw._orig_units.get(ch, 'n/a') for ch in raw.ch_names]
    else:
        units = [_unit2human.get(ch_i['unit'], 'n/a')
                 for ch_i in raw.info['chs']]
        units = [u if u not in ['NA'] else 'n/a' for u in units]
    n_channels = raw.info['nchan']
    sfreq = raw.info['sfreq']

    # default to 'n/a' for status description
    # XXX: improve with API to modify the description
    status_description = ['n/a'] * len(status)

    ch_data = OrderedDict([
        ('name', raw.info['ch_names']),
        ('type', ch_type),
        ('units', units),
        ('low_cutoff', np.full((n_channels), low_cutoff)),
        ('high_cutoff', np.full((n_channels), high_cutoff)),
        ('description', description),
        ('sampling_frequency', np.full((n_channels), sfreq)),
        ('status', status),
        ('status_description', status_description)
    ])
    ch_data = _drop(ch_data, ignored_channels, 'name')

    _write_tsv(fname, ch_data, overwrite, verbose)


_cardinal_ident_mapping = {
    FIFF.FIFFV_POINT_NASION: 'nasion',
    FIFF.FIFFV_POINT_LPA: 'lpa',
    FIFF.FIFFV_POINT_RPA: 'rpa',
}


def _get_fid_coords(dig, raise_error=True):
    """Get the fiducial coordinates from a DigMontage.

    Parameters
    ----------
    dig : mne.channels.DigMontage
        The dig montage with the fiducial coordinates.
    raise_error : bool
        Whether to raise an error if the coordinates are missing or
        incorrectly formatted

    Returns
    -------
    fid_coords : mne.utils.Bunch
        The coordinates stored by fiducial name.
    coord_frame : int
        The integer key corresponding to the coordinate frame of the montage.
    """
    fid_coords = Bunch(nasion=None, lpa=None, rpa=None)
    fid_coord_frames = dict()

    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            key = _cardinal_ident_mapping[d['ident']]
            fid_coords[key] = d['r']
            fid_coord_frames[key] = d['coord_frame']

    if len(fid_coord_frames) > 0 and raise_error:
        if set(fid_coord_frames.keys()) != set(['nasion', 'lpa', 'rpa']):
            raise ValueError(
                f'Some fiducial points are missing, got {fid_coords.keys()}')

        if len(set(fid_coord_frames.values())) > 1:
            raise ValueError(
                'All fiducial points must be in the same coordinate system, '
                f'got {len(fid_coord_frames)})')

    coord_frame = fid_coord_frames.popitem()[1] if fid_coord_frames else None

    return fid_coords, coord_frame


def _events_tsv(events, durations, raw, fname, trial_type, overwrite=False,
                verbose=True):
    """Create an events.tsv file and save it.

    This function will write the mandatory 'onset', and 'duration' columns as
    well as the optional 'value' and 'sample'. The 'value'
    corresponds to the marker value as found in the TRIG channel of the
    recording. In addition, the 'trial_type' field can be written.

    Parameters
    ----------
    events : np.ndarray, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.
    durations : np.ndarray, shape (n_events,)
        The event durations in seconds.
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    fname : str | mne_bids.BIDSPath
        Filename to save the events.tsv to.
    trial_type : dict | None
        Dictionary mapping a brief description key to an event id (value). For
        example {'Go': 1, 'No Go': 2}.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to True or False.
    """
    # Start by filling all data that we know into an ordered dictionary
    first_samp = raw.first_samp
    sfreq = raw.info['sfreq']
    events = events.copy()
    events[:, 0] -= first_samp

    # Onset column needs to be specified in seconds
    data = OrderedDict([('onset', events[:, 0] / sfreq),
                        ('duration', durations),
                        ('trial_type', None),
                        ('value', events[:, 2]),
                        ('sample', events[:, 0])])

    # Now check if trial_type is specified or should be removed
    if trial_type:
        trial_type_map = {v: k for k, v in trial_type.items()}
        data['trial_type'] = [trial_type_map.get(i, 'n/a') for
                              i in events[:, 2]]
    else:
        del data['trial_type']

    _write_tsv(fname, data, overwrite, verbose)


def _readme(datatype, fname, overwrite=False, verbose=True):
    """Create a README file and save it.

    This will write a README file containing an MNE-BIDS citation.
    If a README already exists, the behavior depends on the
    `overwrite` parameter, as described below.

    Parameters
    ----------
    datatype : string
        The type of data contained in the raw file ('meg', 'eeg', 'ieeg')
    fname : str | mne_bids.BIDSPath
        Filename to save the README to.
    overwrite : bool
        Whether to overwrite the existing file (defaults to False).
        If overwrite is True, create a new README containing an
        MNE-BIDS citation. If overwrite is False, append an
        MNE-BIDS citation to the existing README, unless it
        already contains that citation.
    verbose : bool
        Set verbose output to True or False.
    """
    if os.path.isfile(fname) and not overwrite:
        with open(fname, 'r', encoding='utf-8-sig') as fid:
            orig_data = fid.read()
        mne_bids_ref = REFERENCES['mne-bids'] in orig_data
        datatype_ref = REFERENCES[datatype] in orig_data
        if mne_bids_ref and datatype_ref:
            return
        text = '{}References\n----------\n{}{}'.format(
            orig_data + '\n\n',
            '' if mne_bids_ref else REFERENCES['mne-bids'] + '\n\n',
            '' if datatype_ref else REFERENCES[datatype] + '\n')
    else:
        text = 'References\n----------\n{}{}'.format(
            REFERENCES['mne-bids'] + '\n\n', REFERENCES[datatype] + '\n')

    _write_text(fname, text, overwrite=True, verbose=verbose)


def _participants_tsv(raw, subject_id, fname, overwrite=False,
                      verbose=True):
    """Create a participants.tsv file and save it.

    This will append any new participant data to the current list if it
    exists. Otherwise a new file will be created with the provided information.

    Parameters
    ----------
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    subject_id : str
        The subject name in BIDS compatible format ('01', '02', etc.)
    fname : str | mne_bids.BIDSPath
        Filename to save the participants.tsv to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
        If there is already data for the given `subject_id` and overwrite is
        False, an error will be raised.
    verbose : bool
        Set verbose output to True or False.

    """
    subject_id = 'sub-' + subject_id
    data = OrderedDict(participant_id=[subject_id])

    subject_age = "n/a"
    sex = "n/a"
    hand = 'n/a'
    subject_info = raw.info.get('subject_info', None)
    if subject_info is not None:
        # add sex
        sex = _map_options(what='sex', key=subject_info.get('sex', 0),
                           fro='mne', to='bids')

        # add handedness
        hand = _map_options(what='hand', key=subject_info.get('hand', 0),
                            fro='mne', to='bids')

        # determine the age of the participant
        age = subject_info.get('birthday', None)
        meas_date = raw.info.get('meas_date', None)
        if isinstance(meas_date, (tuple, list, np.ndarray)):
            meas_date = meas_date[0]

        if meas_date is not None and age is not None:
            bday = datetime(age[0], age[1], age[2], tzinfo=timezone.utc)
            if isinstance(meas_date, datetime):
                meas_datetime = meas_date
            else:
                meas_datetime = datetime.fromtimestamp(meas_date,
                                                       tz=timezone.utc)
            subject_age = _age_on_date(bday, meas_datetime)
        else:
            subject_age = "n/a"

    data.update({'age': [subject_age], 'sex': [sex], 'hand': [hand]})

    if os.path.exists(fname):
        orig_data = _from_tsv(fname)
        # whether the new data exists identically in the previous data
        exact_included = _contains_row(orig_data,
                                       {'participant_id': subject_id,
                                        'age': subject_age,
                                        'sex': sex,
                                        'hand': hand})
        # whether the subject id is in the previous data
        sid_included = subject_id in orig_data['participant_id']
        # if the subject data provided is different to the currently existing
        # data and overwrite is not True raise an error
        if (sid_included and not exact_included) and not overwrite:
            raise FileExistsError(f'"{subject_id}" already exists in '  # noqa: E501 F821
                                  f'the participant list. Please set '
                                  f'overwrite to True.')

        # Append any columns the original data did not have
        # that mne-bids is trying to write. This handles
        # the edge case where users write participants data for
        # a subset of `hand`, `age` and `sex`.
        for key in data.keys():
            if key in orig_data:
                continue

            # add 'n/a' if any missing columns
            orig_data[key] = ['n/a'] * len(next(iter(data.values())))

        # Append any additional columns that original data had.
        # Keep the original order of the data by looping over
        # the original OrderedDict keys
        col_name = 'participant_id'
        for key in orig_data.keys():
            if key in data:
                continue

            # add original value for any user-appended columns
            # that were not handled by mne-bids
            p_id = data[col_name][0]
            if p_id in orig_data[col_name]:
                row_idx = orig_data[col_name].index(p_id)
                data[key] = [orig_data[key][row_idx]]

        # otherwise add the new data as new row
        data = _combine_rows(orig_data, data, 'participant_id')

    # overwrite is forced to True as all issues with overwrite == False have
    # been handled by this point
    _write_tsv(fname, data, True, verbose)


def _participants_json(fname, overwrite=False, verbose=True):
    """Create participants.json for non-default columns in accompanying TSV.

    Parameters
    ----------
    fname : str | mne_bids.BIDSPath
        Filename to save the scans.tsv to.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to True or False.

    """
    cols = OrderedDict()
    cols['participant_id'] = {'Description': 'Unique participant identifier'}
    cols['age'] = {'Description': 'Age of the participant at time of testing',
                   'Units': 'years'}
    cols['sex'] = {'Description': 'Biological sex of the participant',
                   'Levels': {'F': 'female', 'M': 'male'}}
    cols['hand'] = {'Description': 'Handedness of the participant',
                    'Levels': {'R': 'right', 'L': 'left', 'A': 'ambidextrous'}}

    # make sure to append any JSON fields added by the user
    # Note: mne-bids will overwrite age, sex and hand fields
    # if `overwrite` is True
    if op.exists(fname):
        with open(fname, 'r', encoding='utf-8-sig') as fin:
            orig_cols = json.load(fin, object_pairs_hook=OrderedDict)
        for key, val in orig_cols.items():
            if key not in cols:
                cols[key] = val

    _write_json(fname, cols, overwrite, verbose)


def _scans_tsv(raw, raw_fname, fname, overwrite=False, verbose=True):
    """Create a scans.tsv file and save it.

    Parameters
    ----------
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    raw_fname : str | mne_bids.BIDSPath
        Relative path to the raw data file.
    fname : str
        Filename to save the scans.tsv to.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to True or False.

    """
    # get measurement date in UTC from the data info
    meas_date = raw.info['meas_date']
    if meas_date is None:
        acq_time = 'n/a'
    # The "Z" indicates UTC time
    elif isinstance(meas_date, (tuple, list, np.ndarray)):  # pragma: no cover
        # for MNE < v0.20
        acq_time = _stamp_to_dt(meas_date).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    elif isinstance(meas_date, datetime):
        # for MNE >= v0.20
        acq_time = meas_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # for fif files check whether raw file is likely to be split
    raw_fnames = [raw_fname]
    if raw_fname.endswith('.fif'):
        # check whether fif files were split when saved
        # use the files in the target directory what should be written
        # to scans.tsv
        datatype, basename = raw_fname.split(os.sep)
        raw_dir = op.join(op.dirname(fname), datatype)
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.fif')]
        if basename not in raw_files:
            raw_fnames = []
            split_base = basename.replace('_meg.fif', '_split-{}')
            for raw_f in raw_files:
                if len(raw_f.split('_split-')) == 2:
                    if split_base.format(raw_f.split('_split-')[1]) == raw_f:
                        raw_fnames.append(op.join(datatype, raw_f))
            raw_fnames.sort()

    data = OrderedDict(
        [('filename', ['{:s}'.format(raw_f.replace(os.sep, '/'))
          for raw_f in raw_fnames]),
            ('acq_time', [acq_time] * len(raw_fnames))])

    if os.path.exists(fname):
        orig_data = _from_tsv(fname)
        # if the file name is already in the file raise an error
        if raw_fname in orig_data['filename'] and not overwrite:
            raise FileExistsError(f'"{raw_fname}" already exists in '
                                  f'the scans list. Please set '
                                  f'overwrite to True.')
        # otherwise add the new data
        data = _combine_rows(orig_data, data, 'filename')

    # overwrite is forced to True as all issues with overwrite == False have
    # been handled by this point
    _write_tsv(fname, data, True, verbose)


def _load_image(image, name='image'):
    if not has_nibabel():  # pragma: no cover
        raise ImportError('This function requires nibabel.')
    import nibabel as nib
    if type(image) not in nib.all_image_classes:
        try:
            image = _path_to_str(image)
        except ValueError:
            # image -> str conversion in the try block was successful,
            # so load the file from the specified location. We do this
            # here to keep the try block as short as possible.
            raise ValueError('`{}` must be a path to an MRI data '
                             'file or a nibabel image object, but it '
                             'is of type "{}"'.format(name, type(image)))
        else:
            image = nib.load(image)

    image = nib.Nifti1Image(image.dataobj, image.affine)
    # XYZT_UNITS = NIFT_UNITS_MM (10 in binary or 2 in decimal)
    # seems to be the default for Nifti files
    # https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/xyzt_units.html
    if image.header['xyzt_units'] == 0:
        image.header['xyzt_units'] = np.array(10, dtype='uint8')
    return image


def _meg_landmarks_to_mri_landmarks(meg_landmarks, trans):
    """Convert landmarks from head space to MRI space.

    Parameters
    ----------
    meg_landmarks : np.ndarray, shape (3, 3)
        The meg landmark data: rows LPA, NAS, RPA, columns x, y, z.
    trans : mne.transforms.Transform
        The transformation matrix from head coordinates to MRI coordinates.

    Returns
    -------
    mri_landmarks : np.ndarray, shape (3, 3)
        The mri RAS landmark data converted to from m to mm.
    """
    # Transform MEG landmarks into MRI space, adjust units by * 1e3
    return apply_trans(trans, meg_landmarks, move=True) * 1e3


def _mri_landmarks_to_mri_voxels(mri_landmarks, t1_mgh):
    """Convert landmarks from MRI surface RAS space to MRI voxel space.

    Parameters
    ----------
    mri_landmarks : np.ndarray, shape (3, 3)
        The MRI RAS landmark data: rows LPA, NAS, RPA, columns x, y, z.
    t1_mgh : nib.MGHImage
        The image data in MGH format.

    Returns
    -------
    vox_landmarks : np.ndarray, shape (3, 3)
        The MRI voxel-space landmark data.
    """
    # Get landmarks in voxel space, using the T1 data
    vox2ras_tkr = t1_mgh.header.get_vox2ras_tkr()
    ras2vox_tkr = linalg.inv(vox2ras_tkr)
    vox_landmarks = apply_trans(ras2vox_tkr, mri_landmarks)  # in vox
    return vox_landmarks


def _mri_voxels_to_mri_scanner_ras(mri_landmarks, img_mgh):
    """Convert landmarks from MRI voxel space to MRI scanner RAS space.

    Parameters
    ----------
    mri_landmarks : np.ndarray, shape (3, 3)
        The MRI RAS landmark data: rows LPA, NAS, RPA, columns x, y, z.
    img_mgh : nib.MGHImage
        The image data in MGH format.

    Returns
    -------
    ras_landmarks : np.ndarray, shape (3, 3)
        The MRI scanner RAS landmark data.
    """
    # Get landmarks in voxel space, using the T1 data
    vox2ras = img_mgh.header.get_vox2ras()
    ras_landmarks = apply_trans(vox2ras, mri_landmarks)  # in scanner RAS
    return ras_landmarks


def _mri_scanner_ras_to_mri_voxels(ras_landmarks, img_mgh):
    """Convert landmarks from MRI scanner RAS space to MRI to MRI voxel space.

    Parameters
    ----------
    ras_landmarks : np.ndarray, shape (3, 3)
        The MRI RAS landmark data: rows LPA, NAS, RPA, columns x, y, z.
    img_mgh : nib.MGHImage
        The image data in MGH format.

    Returns
    -------
    vox_landmarks : np.ndarray, shape (3, 3)
        The MRI voxel-space landmark data.
    """
    # Get landmarks in voxel space, using the T1 data
    vox2ras = img_mgh.header.get_vox2ras()
    ras2vox = linalg.inv(vox2ras)
    vox_landmarks = apply_trans(ras2vox, ras_landmarks)  # in vox
    return vox_landmarks


def _sidecar_json(raw, task, manufacturer, fname, datatype,
                  emptyroom_fname=None, overwrite=False, verbose=True):
    """Create a sidecar json file depending on the suffix and save it.

    The sidecar json file provides meta data about the data
    of a certain datatype.

    Parameters
    ----------
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    task : str
        Name of the task the data is based on.
    manufacturer : str
        Manufacturer of the acquisition system. For MEG also used to define the
        coordinate system for the MEG sensors.
    fname : str | mne_bids.BIDSPath
        Filename to save the sidecar json to.
    datatype : str
        Type of the data as in ALLOWED_ELECTROPHYSIO_DATATYPE.
    emptyroom_fname : str | mne_bids.BIDSPath
        For MEG recordings, the path to an empty-room data file to be
        associated with ``raw``. Only supported for MEG.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to True or False. Defaults to True.

    """
    sfreq = raw.info['sfreq']
    try:
        powerlinefrequency = raw.info['line_freq']
        powerlinefrequency = ('n/a' if powerlinefrequency is None else
                              powerlinefrequency)
    except KeyError:
        raise ValueError(
            "PowerLineFrequency parameter is required in the sidecar files. "
            "Please specify it in info['line_freq'] before saving to BIDS, "
            "e.g. by running: "
            "    raw.info['line_freq'] = 60"
            "in your script, or by passing: "
            "    --line_freq 60 "
            "in the command line for a 60 Hz line frequency. If the frequency "
            "is unknown, set it to None")

    if isinstance(raw, BaseRaw):
        rec_type = 'continuous'
    elif isinstance(raw, Epochs):
        rec_type = 'epoched'
    else:
        rec_type = 'n/a'

    # determine whether any channels have to be ignored:
    n_ignored = len([ch_name for ch_name in
                     IGNORED_CHANNELS.get(manufacturer, list()) if
                     ch_name in raw.ch_names])
    # all ignored channels are trigger channels at the moment...

    n_megchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_MEG_CH])
    n_megrefchan = len([ch for ch in raw.info['chs']
                        if ch['kind'] == FIFF.FIFFV_REF_MEG_CH])
    n_eegchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EEG_CH])
    n_ecogchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_ECOG_CH])
    n_seegchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_SEEG_CH])
    n_eogchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EOG_CH])
    n_ecgchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_ECG_CH])
    n_emgchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EMG_CH])
    n_miscchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_MISC_CH])
    n_stimchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_STIM_CH]) - n_ignored
    n_dbschan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_DBS_CH])

    # Set DigitizedLandmarks to True if any of LPA, RPA, NAS are found
    # Set DigitizedHeadPoints to True if any "Extra" points are found
    # (DigitizedHeadPoints done for Neuromag MEG files only)
    digitized_head_points = False
    digitized_landmark = False
    if datatype == 'meg' and raw.info['dig'] is not None:
        for dig_point in raw.info['dig']:
            if dig_point['kind'] in [FIFF.FIFFV_POINT_NASION,
                                     FIFF.FIFFV_POINT_RPA,
                                     FIFF.FIFFV_POINT_LPA]:
                digitized_landmark = True
            elif dig_point['kind'] == FIFF.FIFFV_POINT_EXTRA and \
                    raw.filenames[0].endswith('.fif'):
                digitized_head_points = True

    # Compile cHPI information, if any.
    from mne.io.ctf import RawCTF
    from mne.io.kit.kit import RawKIT

    chpi = False
    hpi_freqs = np.array([])
    if (datatype == 'meg' and
            parse_version(mne.__version__) > parse_version('0.23')):
        # We need to handle different data formats differently
        if isinstance(raw, RawCTF):
            try:
                mne.chpi.extract_chpi_locs_ctf(raw)
                chpi = True
            except RuntimeError:
                logger.info('Could not find cHPI information in raw data.')
        elif isinstance(raw, RawKIT):
            try:
                mne.chpi.extract_chpi_locs_kit(raw)
                chpi = True
            except (RuntimeError, ValueError):
                logger.info('Could not find cHPI information in raw data.')
        else:
            hpi_freqs, _, _ = mne.chpi.get_chpi_info(info=raw.info,
                                                     on_missing='ignore')
            if hpi_freqs.size > 0:
                chpi = True
    elif datatype == 'meg':
        logger.info('Cannot check for & write continuous head localization '
                    'information: requires MNE-Python >= 0.24')
        chpi = None

    # Define datatype-specific JSON dictionaries
    ch_info_json_common = [
        ('TaskName', task),
        ('Manufacturer', manufacturer),
        ('PowerLineFrequency', powerlinefrequency),
        ('SamplingFrequency', sfreq),
        ('SoftwareFilters', 'n/a'),
        ('RecordingDuration', raw.times[-1]),
        ('RecordingType', rec_type)]

    ch_info_json_meg = [
        ('DewarPosition', 'n/a'),
        ('DigitizedLandmarks', digitized_landmark),
        ('DigitizedHeadPoints', digitized_head_points),
        ('MEGChannelCount', n_megchan),
        ('MEGREFChannelCount', n_megrefchan)]

    if chpi is not None:
        ch_info_json_meg.append(('ContinuousHeadLocalization', chpi))
        ch_info_json_meg.append(('HeadCoilFrequency', list(hpi_freqs)))

    if emptyroom_fname is not None:
        ch_info_json_meg.append(('AssociatedEmptyRoom', str(emptyroom_fname)))

    ch_info_json_eeg = [
        ('EEGReference', 'n/a'),
        ('EEGGround', 'n/a'),
        ('EEGPlacementScheme', _infer_eeg_placement_scheme(raw)),
        ('Manufacturer', manufacturer)]

    ch_info_json_ieeg = [
        ('iEEGReference', 'n/a'),
        ('ECOGChannelCount', n_ecogchan),
        ('SEEGChannelCount', n_seegchan + n_dbschan)]
    ch_info_ch_counts = [
        ('EEGChannelCount', n_eegchan),
        ('EOGChannelCount', n_eogchan),
        ('ECGChannelCount', n_ecgchan),
        ('EMGChannelCount', n_emgchan),
        ('MiscChannelCount', n_miscchan),
        ('TriggerChannelCount', n_stimchan)]

    # Stitch together the complete JSON dictionary
    ch_info_json = ch_info_json_common
    if datatype == 'meg':
        append_datatype_json = ch_info_json_meg
    elif datatype == 'eeg':
        append_datatype_json = ch_info_json_eeg
    elif datatype == 'ieeg':
        append_datatype_json = ch_info_json_ieeg

    ch_info_json += append_datatype_json
    ch_info_json += ch_info_ch_counts
    ch_info_json = OrderedDict(ch_info_json)

    _write_json(fname, ch_info_json, overwrite, verbose)

    return fname


def _deface(image, landmarks, deface):
    if not has_nibabel():  # pragma: no cover
        raise ImportError('This function requires nibabel.')
    import nibabel as nib

    inset, theta = (5, 15.)
    if isinstance(deface, dict):
        if 'inset' in deface:
            inset = deface['inset']
        if 'theta' in deface:
            theta = deface['theta']

    if not _is_numeric(inset):
        raise ValueError('inset must be numeric (float, int). '
                         'Got %s' % type(inset))

    if not _is_numeric(theta):
        raise ValueError('theta must be numeric (float, int). '
                         'Got %s' % type(theta))

    if inset < 0:
        raise ValueError('inset should be positive, '
                         'Got %s' % inset)

    if not 0 <= theta < 90:
        raise ValueError('theta should be between 0 and 90 '
                         'degrees. Got %s' % theta)

    # get image data, make a copy
    image_data = image.get_fdata().copy()

    # make indices to move around so that the image doesn't have to
    idxs = np.meshgrid(np.arange(image_data.shape[0]),
                       np.arange(image_data.shape[1]),
                       np.arange(image_data.shape[2]),
                       indexing='ij')
    idxs = np.array(idxs)  # (3, *image_data.shape)
    idxs = np.transpose(idxs, [1, 2, 3, 0])  # (*image_data.shape, 3)
    idxs = idxs.reshape(-1, 3)  # (n_voxels, 3)

    # convert to RAS by applying affine
    idxs = nib.affines.apply_affine(image.affine, idxs)

    # now comes the actual defacing
    # 1. move center of voxels to (nasion - inset)
    # 2. rotate the head by theta from vertical
    x, y, z = nib.affines.apply_affine(image.affine, landmarks)[1]
    idxs = apply_trans(translation(x=-x, y=-y + inset, z=-z), idxs)
    idxs = apply_trans(rotation(x=-np.pi / 2 + np.deg2rad(theta)), idxs)
    idxs = idxs.reshape(image_data.shape + (3,))
    mask = (idxs[..., 2] < 0)  # z < middle
    image_data[mask] = 0.

    # smooth decided against for potential lack of anonymizaton
    # https://gist.github.com/alexrockhill/15043928b716a432db3a84a050b241ae

    image = nib.Nifti1Image(image_data, image.affine, image.header)
    return image


def _write_raw_fif(raw, bids_fname):
    """Save out the raw file in FIF.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw file to save out.
    bids_fname : str | mne_bids.BIDSPath
        The name of the BIDS-specified file where the raw object
        should be saved.

    """
    raw.save(bids_fname, fmt=raw.orig_format, split_naming='bids',
             overwrite=True)


def _write_raw_brainvision(raw, bids_fname, events):
    """Save out the raw file in BrainVision format.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw file to save out.
    bids_fname : str
        The name of the BIDS-specified file where the raw object
        should be saved.
    events : ndarray
        The events as MNE-Python format ndaray.

    """
    if not check_version('pybv', '0.4'):  # pragma: no cover
        raise ImportError('pybv >=0.4 is required for converting '
                          'file to BrainVision format')
    from pybv import write_brainvision
    # Subtract raw.first_samp because brainvision marks events starting from
    # the first available data point and ignores the raw.first_samp
    if events is not None:
        events[:, 0] -= raw.first_samp
        events = events[:, [0, 2]]  # reorder for pybv required order
    meas_date = raw.info['meas_date']
    if meas_date is not None:
        meas_date = _stamp_to_dt(meas_date)

    # pybv needs to know the units of the data for appropriate scaling
    # get voltage units as micro-volts and all other units "as is"
    unit = []
    for chs in raw.info['chs']:
        if chs['unit'] == FIFF.FIFF_UNIT_V:
            unit.append('µV')
        else:
            unit.append(_unit2human.get(chs['unit'], 'n/a'))
            unit = [u if u not in ['NA'] else 'n/a' for u in unit]

    # We enforce conversion to float32 format
    # XXX: pybv can also write to int16, to do that, we need to get
    # original units of data prior to conversion, and add an optimization
    # function to pybv that maximizes the resolution parameter while
    # ensuring that int16 can represent the data in original units.
    if raw.orig_format != 'single':
        warn(f'Encountered data in "{raw.orig_format}" format. '
             f'Converting to float32.', RuntimeWarning)

    # Writing to float32 µV with 0.1 resolution are the pybv defaults,
    # which guarantees accurate roundtrip for values >= 1e-7 µV
    fmt = 'binary_float32'
    resolution = 1e-1
    write_brainvision(data=raw.get_data(),
                      sfreq=raw.info['sfreq'],
                      ch_names=raw.ch_names,
                      fname_base=op.splitext(op.basename(bids_fname))[0],
                      folder_out=op.dirname(bids_fname),
                      events=events,
                      resolution=resolution,
                      unit=unit,
                      fmt=fmt,
                      meas_date=meas_date)


def make_dataset_description(path, name, data_license=None,
                             authors=None, acknowledgements=None,
                             how_to_acknowledge=None, funding=None,
                             references_and_links=None, doi=None,
                             dataset_type='raw',
                             overwrite=False, verbose=False):
    """Create json for a dataset description.

    BIDS datasets may have one or more fields, this function allows you to
    specify which you wish to include in the description. See the BIDS
    documentation for information about what each field means.

    Parameters
    ----------
    path : str
        A path to a folder where the description will be created.
    name : str
        The name of this BIDS dataset.
    data_license : str | None
        The license under which this dataset is published.
    authors : list | str | None
        List of individuals who contributed to the creation/curation of the
        dataset. Must be a list of str or a single comma separated str
        like ['a', 'b', 'c'].
    acknowledgements : list | str | None
        Either a str acknowledging individuals who contributed to the
        creation/curation of this dataset OR a list of the individuals'
        names as str.
    how_to_acknowledge : list | str | None
        Either a str describing how to acknowledge this dataset OR a list of
        publications that should be cited.
    funding : list | str | None
        List of sources of funding (e.g., grant numbers). Must be a list of
        str or a single comma separated str like ['a', 'b', 'c'].
    references_and_links : list | str | None
        List of references to publication that contain information on the
        dataset, or links.  Must be a list of str or a single comma
        separated str like ['a', 'b', 'c'].
    doi : str | None
        The DOI for the dataset.
    dataset_type : str
        Must be either "raw" or "derivative". Defaults to "raw".
    overwrite : bool
        Whether to overwrite existing files or data in files.
        Defaults to False.
        If overwrite is True, provided fields will overwrite previous data.
        If overwrite is False, no existing data will be overwritten or
        replaced.
    verbose : bool
        Set verbose output to True or False.

    Notes
    -----
    The required field BIDSVersion will be automatically filled by mne_bids.

    """
    # Put potential string input into list of strings
    if isinstance(authors, str):
        authors = authors.split(', ')
    if isinstance(funding, str):
        funding = funding.split(', ')
    if isinstance(references_and_links, str):
        references_and_links = references_and_links.split(', ')
    if dataset_type not in ['raw', 'derivative']:
        raise ValueError('`dataset_type` must be either "raw" or '
                         '"derivative."')

    fname = op.join(path, 'dataset_description.json')
    description = OrderedDict([('Name', name),
                               ('BIDSVersion', BIDS_VERSION),
                               ('DatasetType', dataset_type),
                               ('License', data_license),
                               ('Authors', authors),
                               ('Acknowledgements', acknowledgements),
                               ('HowToAcknowledge', how_to_acknowledge),
                               ('Funding', funding),
                               ('ReferencesAndLinks', references_and_links),
                               ('DatasetDOI', doi)])
    if op.isfile(fname):
        with open(fname, 'r', encoding='utf-8-sig') as fin:
            orig_cols = json.load(fin)
        if 'BIDSVersion' in orig_cols and \
                orig_cols['BIDSVersion'] != BIDS_VERSION:
            raise ValueError('Previous BIDS version used, please redo the '
                             'conversion to BIDS in a new directory '
                             'after ensuring all software is updated')
        for key, val in description.items():
            if description[key] is None or not overwrite:
                description[key] = orig_cols.get(key, None)
    # default author to make dataset description BIDS compliant
    # if the user passed an author don't overwrite,
    # if there was an author there, only overwrite if `overwrite=True`
    if authors is None and (description['Authors'] is None or overwrite):
        description['Authors'] = ["[Unspecified]"]

    pop_keys = [key for key, val in description.items() if val is None]
    for key in pop_keys:
        description.pop(key)
    _write_json(fname, description, overwrite=True, verbose=verbose)


def write_raw_bids(raw, bids_path, events_data=None, event_id=None,
                   anonymize=None, format='auto', symlink=False,
                   empty_room=None, allow_preload=False,
                   overwrite=False, verbose=True):
    """Save raw data to a BIDS-compliant folder structure.

    .. warning:: * The original file is simply copied over if the original
                   file format is BIDS-supported for that datatype. Otherwise,
                   this function will convert to a BIDS-supported file format
                   while warning the user. For EEG and iEEG data, conversion
                   will be to BrainVision format; for MEG, conversion will be
                   to FIFF.

                 * ``mne-bids`` will infer the manufacturer information
                   from the file extension. If your file format is non-standard
                   for the manufacturer, please update the manufacturer field
                   in the sidecars manually.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data. It must be an instance of `mne.io.Raw` that is not
        already loaded from disk unless ``allow_preload`` is explicitly set
        to ``True``. See warning for the ``allow_preload`` parameter.
    bids_path : mne_bids.BIDSPath
        The file to write. The `mne_bids.BIDSPath` instance passed here
        **must** have the ``.root`` attribute set. If the ``.datatype``
        attribute is not set, it will be inferred from the recording data type
        found in ``raw``. In case of multiple data types, the ``.datatype``
        attribute must be set.
        Example::

            bids_path = BIDSPath(subject='01', session='01', task='testing',
                                 acquisition='01', run='01', datatype='meg',
                                 root='/data/BIDS')

        This will write the following files in the correct subfolder ``root``::

            sub-01_ses-01_task-testing_acq-01_run-01_meg.fif
            sub-01_ses-01_task-testing_acq-01_run-01_meg.json
            sub-01_ses-01_task-testing_acq-01_run-01_channels.tsv
            sub-01_ses-01_acq-01_coordsystem.json

        and the following one if ``events_data`` is not ``None``::

            sub-01_ses-01_task-testing_acq-01_run-01_events.tsv

        and add a line to the following files::

            participants.tsv
            scans.tsv

        Note that the extension is automatically inferred from the raw
        object.
    events_data : path-like | np.ndarray | None
        Use this parameter to specify events to write to the ``*_events.tsv``
        sidecar file, additionally to the object's `mne.Annotations` (which
        are always written).
        If a path, specifies the location of an MNE events file.
        If an array, the MNE events array (shape: ``(n_events, 3)``).
        If a path or an array and ``raw.annotations`` exist, the union of
        ``event_data`` and ``raw.annotations`` will be written.
        Corresponding descriptions for all event IDs (listed in the third
        column of the MNE events array) must be specified via the ``event_id``
        parameter; otherwise, an exception is raised.
        If ``None``, events will only be inferred from the the raw object's
        `mne.Annotations`.

        .. note::
           If ``not None``, writes the union of ``events_data`` and
           ``raw.annotations``. If you wish to **only** write
           ``raw.annotations``, pass ``events_data=None``. If you want to
           **exclude** the events in ``raw.annotations`` from being written,
           call ``raw.set_annotations(None)`` before invoking this function.

        .. note::
           Descriptions of all event IDs must be specified via the ``event_id``
           parameter.

    event_id : dict | None
        Descriptions of all event IDs, if you passed ``events_data``.
        The descriptions will be written to the ``trial_type`` column in
        ``*_events.tsv``. The dictionary keys correspond to the event
        descriptions and the values to the event IDs. You must specify a
        description for all event IDs in ``events_data``.
    anonymize : dict | None
        If `None` (default), no anonymization is performed.
        If a dictionary, data will be anonymized depending on the dictionary
        keys: ``daysback`` is a required key, ``keep_his`` is optional.

        ``daysback`` : int
            Number of days by which to move back the recording date in time.
            In studies with multiple subjects the relative recording date
            differences between subjects can be kept by using the same number
            of ``daysback`` for all subject anonymizations. ``daysback`` should
            be great enough to shift the date prior to 1925 to conform with
            BIDS anonymization rules.

        ``keep_his`` : bool
            If ``False`` (default), all subject information next to the
            recording date will be overwritten as well. If True, keep subject
            information apart from the recording date.

    format : 'auto' | 'BrainVision' | 'FIF'
        Controls the file format of the data after BIDS conversion. If
        ``'auto'``, MNE-BIDS will attempt to convert the input data to BIDS
        without a change of the original file format. A conversion to a
        different file format (BrainVision, or FIF) will only take place when
        the original file format lacks some necessary features. When a str is
        passed, a conversion can be forced to the BrainVision format for EEG,
        or the FIF format for MEG data.
    symlink : bool
        Instead of copying the source files, only create symbolic links to
        preserve storage space. This is only allowed when not anonymizing the
        data (i.e., ``anonymize`` must be ``None``).

        .. note::
           Symlinks currently only work with FIFF files. In case of split
           files, only a link to the first file will be created, and
           :func:`mne_bids.read_raw_bids` will correctly handle reading the
           data again.

        .. note::
           Symlinks are currently only supported on macOS and Linux. We will
           add support for Windows 10 at a later time.

    empty_room : mne_bids.BIDSPath | None
        The empty-room recording to be associated with this file. This is
        only supported for MEG data, and only if the ``root`` attributes of
        ``bids_path`` and ``empty_room`` are the same. Pass ``None``
        (default) if you do not wish to specify an associated empty-room
        recording.
    allow_preload : bool
        If ``True``, allow writing of preloaded raw objects (i.e.,
        ``raw.preload`` is ``True``). Because the original file is ignored, you
        must specify what ``format`` to write (not ``auto``).

        .. warning::
            BIDS was originally designed for unprocessed or minimally processed
            data. For this reason, by default, we prevent writing of preloaded
            data that may have been modified. Only use this option when
            absolutely necessary: for example, manually converting from file
            formats not supported by MNE or writing preprocessed derivatives.
            Be aware that these use cases are not fully supported.

    overwrite : bool
        Whether to overwrite existing files or data in files.
        Defaults to ``False``.

        If ``True``, any existing files with the same BIDS parameters
        will be overwritten with the exception of the ``*_participants.tsv``
        and ``*_scans.tsv`` files. For these files, parts of pre-existing data
        that match the current data will be replaced. For
        ``*_participants.tsv``, specifically, age, sex and hand fields will be
        overwritten, while any manually added fields in ``participants.json``
        and ``participants.tsv`` by a user will be retained.
        If ``False``, no existing data will be overwritten or
        replaced.
    verbose : bool
        If ``True``, this will print a snippet of the sidecar files. Otherwise,
        no content will be printed.

    Returns
    -------
    bids_path : mne_bids.BIDSPath
        The path of the created data file.

    Notes
    -----
    You should ensure that ``raw.info['subject_info']`` and
    ``raw.info['meas_date']`` are set to proper (not-``None``) values to allow
    for the correct computation of each participant's age when creating
    ``*_participants.tsv``.

    This function will convert existing `mne.Annotations` from
    ``raw.annotations`` to events. Additionally, any events supplied via
    ``events_data`` will be written too. To avoid writing of annotations,
    remove them from the raw file via ``raw.set_annotations(None)`` before
    invoking ``write_raw_bids``.

    To write events encoded in a ``STIM`` channel, you first need to create the
    events array manually and pass it to this function:

    ..
        events = mne.find_events(raw, min_duration=0.002)
        write_raw_bids(..., events_data=events)

    See the documentation of :func:`mne.find_events` for more information on
    event extraction from ``STIM`` channels.

    When anonymizing ``.edf`` files, then the file format for EDF limits
    how far back we can set the recording date. Therefore, all anonymized
    EDF datasets will have an internal recording date of ``01-01-1985``,
    and the actual recording date will be stored in the ``scans.tsv``
    file's ``acq_time`` column.

    ``write_raw_bids`` will generate a ``dataset_description.json`` file
    if it does not already exist. Minimal metadata will be written there.
    If one sets ``overwrite`` to ``True`` here, it will not overwrite an
    existing ``dataset_description.json`` file.
    If you need to add more data there, or overwrite it, then you should
    call :func:`mne_bids.make_dataset_description` directly.

    See Also
    --------
    mne.io.Raw.anonymize
    mne.find_events
    mne.Annotations
    mne.events_from_annotations

    """
    if not isinstance(raw, BaseRaw):
        raise ValueError('raw_file must be an instance of BaseRaw, '
                         'got %s' % type(raw))

    if raw.preload is not False and not allow_preload:
        raise ValueError('The data is already loaded from disk and may be '
                         'altered. See warning for "allow_preload".')

    if not isinstance(bids_path, BIDSPath):
        raise RuntimeError('"bids_path" must be a BIDSPath object. Please '
                           'instantiate using mne_bids.BIDSPath().')

    _validate_type(events_data, types=('path-like', np.ndarray, None),
                   item_name='events_data',
                   type_name='path-like, NumPy array, or None')

    if symlink and sys.platform in ('win32', 'cygwin'):
        raise NotImplementedError('Symbolic links are currently not supported '
                                  'by MNE-BIDS on Windows operating systems.')

    if symlink and anonymize is not None:
        raise ValueError('Cannot create symlinks when anonymizing data.')

    # Check if the root is available
    if bids_path.root is None:
        raise ValueError('The root of the "bids_path" must be set. '
                         'Please use `bids_path.update(root="<root>")` '
                         'to set the root of the BIDS folder to read.')

    if events_data is not None and event_id is None:
        raise RuntimeError('You passed events_data, but no event_id '
                           'dictionary. You need to pass both, or neither.')

    if event_id is not None and events_data is None:
        raise RuntimeError('You passed event_id, but no events_data NumPy '
                           'array. You need to pass both, or neither.')

    _validate_type(item=empty_room, item_name='empty_room',
                   types=(BIDSPath, None))

    raw = raw.copy()
    convert = False  # flag if converting not copying

    # Load file, filename, extension
    if not allow_preload:
        raw_fname = raw.filenames[0]
        if '.ds' in op.dirname(raw.filenames[0]):
            raw_fname = op.dirname(raw.filenames[0])
        # point to file containing header info for multifile systems
        raw_fname = raw_fname.replace('.eeg', '.vhdr')
        raw_fname = raw_fname.replace('.fdt', '.set')
        raw_fname = raw_fname.replace('.dat', '.lay')
        _, ext = _parse_ext(raw_fname, verbose=verbose)

        if ext not in ALLOWED_INPUT_EXTENSIONS:
            raise ValueError(f'Unrecognized file format {ext}')

        if symlink and ext != '.fif':
            raise NotImplementedError('Symlinks are currently only supported '
                                      'for FIFF files.')

        raw_orig = reader[ext](**raw._init_kwargs)
    else:
        if format == 'BrainVision':
            ext = '.vhdr'
        elif format == 'FIF':
            ext = '.fif'
        else:
            raise ValueError('For preloaded data, you must specify a valid '
                             'format. See "allow_preload".')
        raw_orig = raw

    # Check times
    if not np.array_equal(raw.times, raw_orig.times):
        if len(raw.times) == len(raw_orig.times):
            msg = ("raw.times has changed since reading from disk, but "
                   "write_raw_bids() doesn't allow writing modified data.")
        else:
            msg = ("The raw data you want to write contains {comp} time "
                   "points than the raw data on disk. It is possible that you "
                   "{guess} your data, which write_raw_bids() won't accept.")
            if len(raw.times) < len(raw_orig.times):
                msg = msg.format(comp='fewer', guess='cropped')
            elif len(raw.times) > len(raw_orig.times):
                msg = msg.format(comp='more', guess='concatenated')

        msg += (' If you believe you have a valid use case that should be '
                'supported, please reach out to the developers at '
                'https://github.com/mne-tools/mne-bids/issues')
        raise ValueError(msg)

    # Initialize BIDS path
    datatype = _handle_datatype(raw, bids_path.datatype, verbose)
    bids_path = (bids_path.copy()
                 .update(datatype=datatype, suffix=datatype, extension=ext))

    # Check whether provided info and raw indicates valid MEG emptyroom data
    data_is_emptyroom = False
    if (bids_path.datatype == 'meg' and bids_path.subject == 'emptyroom' and
            bids_path.task == 'noise'):
        data_is_emptyroom = True
        # check the session date provided is consistent with the value in raw
        meas_date = raw.info.get('meas_date', None)
        if meas_date is not None:
            if not isinstance(meas_date, datetime):
                meas_date = datetime.fromtimestamp(meas_date[0],
                                                   tz=timezone.utc)
            er_date = meas_date.strftime('%Y%m%d')
            if er_date != bids_path.session:
                raise ValueError(
                    f"The date provided for the empty-room session "
                    f"({bids_path.session}) doesn't match the empty-room "
                    f"recording date found in the data's info structure "
                    f"({er_date})."
                )
            if anonymize is not None and 'daysback' in anonymize:
                meas_date = meas_date - timedelta(anonymize['daysback'])
                session = meas_date.strftime('%Y%m%d')
                bids_path = bids_path.copy().update(session=session)

    associated_er_path = None
    if empty_room is not None:
        if bids_path.datatype != 'meg':
            raise ValueError('"empty_room" is only supported for '
                             'MEG data.')
        if data_is_emptyroom:
            raise ValueError('You cannot write empty-room data and pass '
                             '"empty_room" at the same time.')
        if bids_path.root != empty_room.root:
            raise ValueError('The MEG data and its associated empty-room '
                             'recording must share the same BIDS root.')

        associated_er_path = empty_room.fpath
        if not associated_er_path.exists():
            raise FileNotFoundError(f'Empty-room data file not found: '
                                    f'{associated_er_path}')

        # Turn it into a path relative to the BIDS root
        associated_er_path = Path(str(associated_er_path)
                                  .replace(str(empty_room.root), ''))
        # Ensure it works on Windows too
        associated_er_path = associated_er_path.as_posix()

    # In case of an "emptyroom" subject, BIDSPath() will raise
    # an exception if we don't provide a valid task ("noise"). Now,
    # scans_fname, electrodes_fname, and coordsystem_fname must NOT include
    # the task entity. Therefore, we cannot generate them with
    # BIDSPath() directly. Instead, we use BIDSPath() directly
    # as it does not make any advanced check.

    data_path = bids_path.mkdir().directory

    # create *_scans.tsv
    session_path = BIDSPath(subject=bids_path.subject,
                            session=bids_path.session, root=bids_path.root)
    scans_path = session_path.copy().update(suffix='scans', extension='.tsv')

    # create *_coordsystem.json
    coordsystem_path = session_path.copy().update(
        acquisition=bids_path.acquisition, space=bids_path.space,
        datatype=bids_path.datatype, suffix='coordsystem', extension='.json')

    # For the remaining files, we can use BIDSPath to alter.
    readme_fname = op.join(bids_path.root, 'README')
    participants_tsv_fname = op.join(bids_path.root, 'participants.tsv')
    participants_json_fname = participants_tsv_fname.replace('.tsv',
                                                             '.json')

    sidecar_path = bids_path.copy().update(suffix=bids_path.datatype,
                                           extension='.json')
    events_path = bids_path.copy().update(suffix='events', extension='.tsv')
    channels_path = bids_path.copy().update(
        suffix='channels', extension='.tsv')

    # Anonymize
    if anonymize is not None:
        daysback, keep_his = _check_anonymize(anonymize, raw, ext)
        raw.anonymize(daysback=daysback, keep_his=keep_his, verbose=verbose)

        if bids_path.datatype == 'meg' and ext != '.fif':
            if verbose:
                warn('Converting to FIF for anonymization')
            convert = True
            bids_path.update(extension='.fif')
        elif bids_path.datatype in ['eeg', 'ieeg']:
            if ext not in ['.vhdr', '.edf', '.bdf']:
                if verbose:
                    warn('Converting data files to BrainVision format '
                         'for anonymization')
                convert = True
                bids_path.update(extension='.vhdr')

    # Read in Raw object and extract metadata from Raw object if needed
    orient = ORIENTATION.get(ext, 'n/a')
    unit = UNITS.get(ext, 'n/a')
    manufacturer = MANUFACTURERS.get(ext, 'n/a')

    # save readme file unless it already exists
    # XXX: can include README overwrite in future if using a template API
    # XXX: see https://github.com/mne-tools/mne-bids/issues/551
    _readme(bids_path.datatype, readme_fname, False, verbose)

    # save all participants meta data
    _participants_tsv(raw, bids_path.subject, participants_tsv_fname,
                      overwrite, verbose)
    _participants_json(participants_json_fname, True, verbose)

    # for MEG, we only write coordinate system
    if bids_path.datatype == 'meg' and not data_is_emptyroom:
        _write_coordsystem_json(raw=raw, unit=unit, hpi_coord_system=orient,
                                sensor_coord_system=orient,
                                fname=coordsystem_path.fpath,
                                datatype=bids_path.datatype,
                                overwrite=overwrite, verbose=verbose)
    elif bids_path.datatype in ['eeg', 'ieeg']:
        # We only write electrodes.tsv and accompanying coordsystem.json
        # if we have an available DigMontage
        if raw.info['dig'] is not None and raw.info['dig']:
            _write_dig_bids(bids_path, raw, overwrite, verbose)
    else:
        logger.warning(f'Writing of electrodes.tsv is not supported '
                       f'for data type "{bids_path.datatype}". Skipping ...')

    # Write events.
    if not data_is_emptyroom:
        events_array, event_dur, event_desc_id_map = _read_events(
            events_data, event_id, raw, task=bids_path.task, verbose=False
        )
        if events_array.size != 0:
            _events_tsv(events=events_array, durations=event_dur, raw=raw,
                        fname=events_path.fpath, trial_type=event_desc_id_map,
                        overwrite=overwrite, verbose=verbose)
        # Kepp events_array around for BrainVision writing below.
        del event_desc_id_map, events_data, event_id, event_dur

    # make dataset description and add template data if it does not
    # already exist. Always set overwrite to False here. If users
    # want to edit their dataset_description, they can directly call
    # this function.
    make_dataset_description(bids_path.root, name=" ", overwrite=False,
                             verbose=verbose)

    _sidecar_json(raw, task=bids_path.task, manufacturer=manufacturer,
                  fname=sidecar_path.fpath, datatype=bids_path.datatype,
                  emptyroom_fname=associated_er_path,
                  overwrite=overwrite, verbose=verbose)
    _channels_tsv(raw, channels_path.fpath, overwrite, verbose)

    # create parent directories if needed
    _mkdir_p(os.path.dirname(data_path))

    if os.path.exists(bids_path.fpath):
        if overwrite:
            # Need to load data before removing its source
            raw.load_data()
            if bids_path.fpath.is_dir():
                shutil.rmtree(bids_path.fpath)
            else:
                bids_path.fpath.unlink()
        else:
            raise FileExistsError(
                f'"{bids_path.fpath}" already exists. '  # noqa: F821
                'Please set overwrite to True.')

    # If not already converting for anonymization, we may still need to do it
    # if current format not BIDS compliant
    if not convert:
        convert = ext not in ALLOWED_DATATYPE_EXTENSIONS[bids_path.datatype]

        if convert and symlink:
            raise RuntimeError(
                'The input file format is not supported by the BIDS standard. '
                'To store your data, MNE-BIDS would have to convert it. '
                'However, this is not possible since you set symlink=True. '
                'Deactivate symbolic links by passing symlink=False to allow '
                'file format conversion.')

    # check if there is an BIDS-unsupported MEG format
    if bids_path.datatype == 'meg' and convert and not anonymize:
        raise ValueError(
            f"Got file extension {ext} for MEG data, "
            f"expected one of "
            f"{', '.join(sorted(ALLOWED_DATATYPE_EXTENSIONS['meg']))}")

    if not convert and verbose:
        logger.info(f'Copying data files to {bids_path.fpath.name}')

    # If users desire a certain format, will handle auto-conversion
    if format != 'auto':
        if format == 'BrainVision' and bids_path.datatype in ['ieeg', 'eeg']:
            convert = True
            bids_path.update(extension='.vhdr')
        elif format == 'FIF' and bids_path.datatype == 'meg':
            convert = True
            bids_path.update(extension='.fif')
        elif all(format not in values for values in CONVERT_FORMATS.values()):
            raise ValueError(f'The input "format" {format} is not an '
                             f'accepted input format for `write_raw_bids`. '
                             f'Please use one of {CONVERT_FORMATS[datatype]} '
                             f'for {datatype} datatype.')
        elif format not in CONVERT_FORMATS[datatype]:
            raise ValueError(f'The input "format" {format} is not an '
                             f'accepted input format for {datatype} datatype. '
                             f'Please use one of {CONVERT_FORMATS[datatype]} '
                             f'for {datatype} datatype.')

    # File saving branching logic
    if convert:
        if bids_path.datatype == 'meg':
            _write_raw_fif(
                raw, (op.join(data_path, bids_path.basename)
                      if ext == '.pdf' else bids_path.fpath))
        else:
            if verbose:
                warn('Converting data files to BrainVision format')
            bids_path.update(suffix=bids_path.datatype, extension='.vhdr')
            # XXX Should we write durations here too?
            _write_raw_brainvision(raw, bids_path.fpath, events=events_array)
    elif ext == '.fif':
        if symlink:
            link_target = Path(raw.filenames[0])
            link_path = bids_path.fpath
            link_path.symlink_to(link_target)
        else:
            _write_raw_fif(raw, bids_path)
    # CTF data is saved and renamed in a directory
    elif ext == '.ds':
        copyfile_ctf(raw_fname, bids_path)
    # BrainVision is multifile, copy over all of them and fix pointers
    elif ext == '.vhdr':
        copyfile_brainvision(raw_fname, bids_path, anonymize=anonymize)
    elif ext in ['.edf', '.bdf']:
        if anonymize is not None:
            warn("EDF/EDF+/BDF files contain two fields for recording dates."
                 "Due to file format limitations, one of these fields only "
                 "supports 2-digit years. The date for that field will be "
                 "set to 85 (i.e., 1985), the earliest possible date. "
                 "The true anonymized date is stored in the scans.tsv file.")
        copyfile_edf(raw_fname, bids_path, anonymize=anonymize)
    # EEGLAB .set might be accompanied by a .fdt - find out and copy it too
    elif ext == '.set':
        copyfile_eeglab(raw_fname, bids_path)
    elif ext == '.pdf':
        raw_dir = op.join(data_path, op.splitext(bids_path.basename)[0])
        _mkdir_p(raw_dir)
        copyfile_bti(raw_orig, raw_dir)
    elif ext in ['.con', '.sqd']:
        copyfile_kit(raw_fname, bids_path.fpath, bids_path.subject,
                     bids_path.session, bids_path.task, bids_path.run,
                     raw._init_kwargs)
    else:
        shutil.copyfile(raw_fname, bids_path)

    # write to the scans.tsv file the output file written
    scan_relative_fpath = op.join(bids_path.datatype, bids_path.fpath.name)
    _scans_tsv(raw, scan_relative_fpath, scans_path.fpath, overwrite, verbose)
    if verbose:
        print(f'Wrote {scans_path.fpath} entry with {scan_relative_fpath}.')

    return bids_path


def get_anat_landmarks(image, info, trans, fs_subject, fs_subjects_dir=None):
    """Get anatomical landmarks in MRI voxel coordinates.

    This function transforms the fiducial points from "head" to MRI "voxel"
    coordinate space. The landmarks obtained are defined w.r.t. the MRI passed
    via the ``image`` parameter.

    Parameters
    ----------
    image : str | pathlib.Path | NibabelImageObject
        Path to an MRI scan (e.g. T1w) of the subject. Can be in any format
        readable by nibabel. Can also be a nibabel image object of an
        MRI scan. Will be written as a .nii.gz file.
    info : mne.Info
        The measurement information from an electrophysiology recording of
        the subject with the anatomical landmarks stored in its
        :class:`mne.channels.DigMontage`.
    trans : mne.transforms.Transform | str
        The transformation matrix from head to MRI coordinates. Can
        also be a string pointing to a ``.trans`` file containing the
        transformation matrix. If ``None`` and no ``landmarks`` parameter is
        passed, no sidecar JSON file will be created.
    fs_subject : str
        The subject identifier used for FreeSurfer. If ``None``, defaults to
        the ``subject`` entity in ``bids_path``. Must be provided to write
        the anatomical landmarks if they are not provided in MRI voxel space.
        This is because the head coordinate of a
        :class:`mne.channels.DigMontage` is aligned using FreeSurfer surfaces.
    fs_subjects_dir : str | pathlib.Path | None
        The FreeSurfer subjects directory. If ``None``, defaults to the
        ``SUBJECTS_DIR`` environment variable. Must be provided to write
        anatomical landmarks if they are not provided in MRI voxel space.

    Returns
    -------
    landmarks : mne.channels.DigMontage
        A montage with the landmarks in MRI voxel space.
    """
    if not has_nibabel():  # pragma: no cover
        raise ImportError('This function requires nibabel.')
    import nibabel as nib
    coords_dict, coord_frame = _get_fid_coords(info['dig'])
    if coord_frame != FIFF.FIFFV_COORD_HEAD:
        raise ValueError('Fiducial coordinates in `info` must be in '
                         f'the head coordinate frame, got {coord_frame}')
    landmarks = np.asarray((coords_dict['lpa'],
                            coords_dict['nasion'],
                            coords_dict['rpa']))
    # get trans and ensure it is from head to MRI
    trans, _ = _get_trans(trans, fro='head', to='mri')
    landmarks = _meg_landmarks_to_mri_landmarks(landmarks, trans)
    fs_subjects_dir = get_subjects_dir(fs_subjects_dir, raise_error=True)
    t1_fname = Path(fs_subjects_dir) / fs_subject / 'mri' / 'T1.mgz'
    if not t1_fname.exists():
        raise ValueError('Freesurfer recon-all subject folder '
                         'is incorrect or improperly formatted, '
                         f'got {Path(fs_subjects_dir) / fs_subject}')
    t1w_img = _load_image(str(t1_fname), name='T1.mgz')
    t1w_mgh = nib.MGHImage(t1w_img.dataobj, t1w_img.affine)
    # go to T1 voxel space from surface RAS/TkReg RAS/freesurfer
    landmarks = _mri_landmarks_to_mri_voxels(landmarks, t1w_mgh)
    # go to T1 scanner space from T1 voxel space
    landmarks = _mri_voxels_to_mri_scanner_ras(landmarks, t1w_mgh)
    img_nii = _load_image(image, name='image')
    img_mgh = nib.MGHImage(img_nii.dataobj, img_nii.affine)
    landmarks = _mri_scanner_ras_to_mri_voxels(landmarks, img_mgh)
    landmarks = mne.channels.make_dig_montage(
        lpa=landmarks[0], nasion=landmarks[1], rpa=landmarks[2],
        coord_frame='mri_voxel')
    return landmarks


def write_anat(image, bids_path, landmarks=None, deface=False,
               raw=None, trans=None, t1w=None, overwrite=False, verbose=False):
    """Put anatomical MRI data into a BIDS format.

    Given an MRI scan, format and store the MR data according to BIDS in the
    correct location inside the specified :class:`mne_bids.BIDSPath`. If a
    transformation matrix is supplied, this information will be stored in a
    sidecar JSON file.

    .. note:: To generate the JSON sidecar with anatomical landmark
              coordinates ("fiducials"), you need to pass the landmarks via
              the ``landmarks`` parameter. :func:`mne_bids.get_anat_landmarks`
              may be useful for getting the ``landmarks``.

    Parameters
    ----------
    image : str | pathlib.Path | NibabelImageObject
        Path to an MRI scan (e.g. T1w) of the subject. Can be in any format
        readable by nibabel. Can also be a nibabel image object of an
        MRI scan. Will be written as a .nii.gz file.
    bids_path : mne_bids.BIDSPath
        The file to write. The `mne_bids.BIDSPath` instance passed here
        **must** have the ``root`` and ``subject`` attributes set.
        The suffix is assumed to be ``'T1w'`` if not present. It can
        also be ``'FLASH'``, for example, to indicate FLASH MRI.
    landmarks : mne.channels.DigMontage | str | None
        The DigMontage or filepath to a DigMontage with landmarks that can be
        passed to provide information for defacing. Landmarks can be determined
        from the head model using `mne coreg` GUI, or they can be determined
        from the MRI using freeview.  If ``None`` and no ``trans`` parameter
        is passed, no sidecar JSON file will be created.
    deface : bool | dict
        If False, no defacing is performed.
        If True, deface with default parameters.
        `trans` and `raw` must not be `None` if True.
        If dict, accepts the following keys:

        - `inset`: how far back in voxels to start defacing
          relative to the nasion (default 5)

        - `theta`: is the angle of the defacing shear in degrees relative
          to vertical (default 15).

    raw : mne.io.Raw | None
        The raw data of ``subject`` corresponding to the MR scan in ``image``.
        If ``None``, ``trans`` has to be ``None`` as well.

        Deprecated in v0.8, use :func:`mne_bids.get_anat_landmarks` instead.

    trans : mne.transforms.Transform | str | None
        The transformation matrix from head to MRI coordinates. Can
        also be a string pointing to a ``.trans`` file containing the
        transformation matrix. If ``None`` and no ``landmarks`` parameter is
        passed, no sidecar JSON file will be created.

        Deprecated in v0.8, use :func:`mne_bids.get_anat_landmarks` instead.

    t1w : str | pathlib.Path | NibabelImageObject | None
        This parameter is useful if image written is not already a T1 image.
        If the image written is to have a sidecar or be defaced,
        this can be done using `raw`, `trans` and `t1w`. The T1 must be
        passed here because the coregistration uses freesurfer surfaces which
        are in T1 space.

        Deprecated in v0.8, use :func:`mne_bids.get_anat_landmarks` instead.

    overwrite : bool
        Whether to overwrite existing files or data in files.
        Defaults to False.
        If overwrite is True, any existing files with the same BIDS parameters
        will be overwritten with the exception of the `participants.tsv` and
        `scans.tsv` files. For these files, parts of pre-existing data that
        match the current data will be replaced.
        If overwrite is False, no existing data will be overwritten or
        replaced.

    verbose : bool
        If ``True``, this will print a snippet of the sidecar files. If
        ``False``, no content will be printed.

    Returns
    -------
    bids_path : mne_bids.BIDSPath
        Path to the written MRI data.
    """
    if not has_nibabel():  # pragma: no cover
        raise ImportError('This function requires nibabel.')
    import nibabel as nib

    if raw is not None or trans is not None or t1w is not None:
        raise ValueError('`raw`, `trans` and `t1w` are depreciated '
                         'use `mne_bids.get_anat_landmarks` instead')

    del raw, trans, t1w

    write_sidecar = landmarks is not None

    if deface and landmarks is None:
        raise ValueError('`landmarks` must be provided to deface the image')

    # Check if the root is available
    if bids_path.root is None:
        raise ValueError('The root of the "bids_path" must be set. '
                         'Please use `bids_path.update(root="<root>")` '
                         'to set the root of the BIDS folder to read.')
    # create a copy
    bids_path = bids_path.copy()

    # BIDS demands anatomical scans have no task associated with them
    bids_path.update(task=None)

    # XXX For now, only support writing a single run.
    bids_path.update(run=None)

    # this file is anat
    if bids_path.datatype is None:
        bids_path.update(datatype='anat')

    # default to T1w
    if not bids_path.suffix:
        bids_path.update(suffix='T1w')

    #  data is compressed Nifti
    bids_path.update(extension='.nii.gz')

    # create the directory for the MRI data
    bids_path.directory.mkdir(exist_ok=True, parents=True)

    # Try to read our MRI file and convert to MGH representation
    image_nii = _load_image(image)

    # Check if we have necessary conditions for writing a sidecar JSON
    if write_sidecar:
        if isinstance(landmarks, str):
            landmarks, coord_frame = read_fiducials(landmarks)
            landmarks = np.array([landmark['r'] for landmark in
                                  landmarks], dtype=float)  # unpack
        else:
            # Prepare to write the sidecar JSON, extract MEG landmarks
            coords_dict, coord_frame = _get_fid_coords(landmarks.dig)
            landmarks = np.asarray((coords_dict['lpa'],
                                    coords_dict['nasion'],
                                    coords_dict['rpa']))

        # check if coord frame is supported
        if coord_frame not in (FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                               FIFF.FIFFV_MNE_COORD_RAS):
            raise ValueError(f'Coordinate frame not supported: {coord_frame}')

        # convert to voxels from scanner RAS to voxels
        if coord_frame == FIFF.FIFFV_MNE_COORD_RAS:
            # Make MGH image for header properties
            img_mgh = nib.MGHImage(image_nii.dataobj, image_nii.affine)
            landmarks = _mri_scanner_ras_to_mri_voxels(
                landmarks * 1e3, img_mgh)

        # Write sidecar.json
        img_json = dict()
        img_json['AnatomicalLandmarkCoordinates'] = \
            {'LPA': list(landmarks[0, :]),
             'NAS': list(landmarks[1, :]),
             'RPA': list(landmarks[2, :])}
        fname = bids_path.copy().update(extension='.json')
        if op.isfile(fname) and not overwrite:
            raise IOError('Wanted to write a file but it already exists and '
                          '`overwrite` is set to False. File: "{}"'
                          .format(fname))
        _write_json(fname, img_json, overwrite, verbose)

        if deface:
            image_nii = _deface(image_nii, landmarks, deface)

    # Save anatomical data
    if op.exists(bids_path):
        if overwrite:
            os.remove(bids_path)
        else:
            raise IOError(f'Wanted to write a file but it already exists and '
                          f'`overwrite` is set to False. File: "{bids_path}"')

    nib.save(image_nii, bids_path.fpath)

    return bids_path


def mark_bad_channels(ch_names, descriptions=None, *, bids_path,
                      overwrite=False, verbose=True):
    """Update which channels are marked as "bad" in an existing BIDS dataset.

    Parameters
    ----------
    ch_names : str | list of str
        The names of the channel(s) to mark as bad. Pass an empty list in
        combination with ``overwrite=True`` to mark all channels as good.
    descriptions : None | str | list of str
        Descriptions of the reasons that lead to the exclusion of the
        channel(s). If a list, it must match the length of ``ch_names``.
        If ``None``, no descriptions are added.
    bids_path : mne_bids.BIDSPath
        The recording to update. The :class:`mne_bids.BIDSPath` instance passed
        here **must** have the ``.root`` attribute set. The ``.datatype``
        attribute **may** be set. If ``.datatype`` is not set and only one data
        type (e.g., only EEG or MEG data) is present in the dataset, it will be
        selected automatically.
    overwrite : bool
        If ``False``, only update the information of the channels passed via
        ``ch_names``, and leave the rest untouched. If ``True``, update the
        information of **all** channels: mark the channels passed via
        ``ch_names`` as bad, and all remaining channels as good, also
        discarding their descriptions.
    verbose : bool
        The verbosity level.

    Examples
    --------
    Mark a single channel as bad.

    >>> root = Path('./mne_bids/tests/data/tiny_bids').absolute()
    >>> bids_path = BIDSPath(subject='01', task='rest', session='eeg',
    ...                      datatype='eeg', root=root)
    >>> mark_bad_channels('C4', bids_path=bids_path, verbose=False)
    Processing channel C4:
        status: bad
        description: n/a

    Mark multiple channels as bad, and add a description as to why.

    >>> bads = ['C3', 'PO10']
    >>> descriptions = ['very noisy', 'continuously flat']
    >>> mark_bad_channels(bads, descriptions, bids_path=bids_path,
    ...                   verbose=False)
    Processing channel C3:
        status: bad
        description: very noisy
    Processing channel PO10:
        status: bad
        description: continuously flat

    Mark two channels as bad, and mark all others as good by setting
    ``overwrite=True``.

    >>> bads = ['C3', 'C4']
    >>> mark_bad_channels(bads, bids_path=bids_path, # doctest: +SKIP
    ...                   overwrite=True, verbose=False)

    Mark all channels as good by passing an empty list of bad channels, and
    setting ``overwrite=True``.

    >>> mark_bad_channels([], bids_path=bids_path, overwrite=True,
    ...                   verbose=False)
    Resetting status and description for all channels.
    """
    if not ch_names and not overwrite:
        raise ValueError('You did not pass a channel name, but set '
                         'overwrite=False. If you wish to mark all channels '
                         'as good, please pass overwrite=True')

    if descriptions and not ch_names:
        raise ValueError('You passed descriptions, but no channels.')

    if not ch_names:
        descriptions = []
    if isinstance(ch_names, str):
        ch_names = [ch_names]
    if isinstance(descriptions, str):
        descriptions = [descriptions]
    elif not descriptions:
        descriptions = ['n/a'] * len(ch_names)
    if len(ch_names) != len(descriptions):
        raise ValueError('Number of channels and descriptions must match.')

    if not isinstance(bids_path, BIDSPath):
        raise RuntimeError('"bids_path" must be a BIDSPath object. Please '
                           'instantiate using mne_bids.BIDSPath().')

    if bids_path.root is None:
        raise ValueError('The root of the "bids_path" must be set. '
                         'Please use `bids_path.update(root="<root>")` '
                         'to set the root of the BIDS folder to read.')

    # Read sidecar file.
    channels_fname = _find_matching_sidecar(bids_path, suffix='channels',
                                            extension='.tsv')
    tsv_data = _from_tsv(channels_fname)

    # Read sidecar and create required columns if they do not exist.
    if 'status' not in tsv_data:
        logger.info('No "status" column found in input file. Creating.')
        tsv_data['status'] = ['good'] * len(tsv_data['name'])

    if 'status_description' not in tsv_data:
        logger.info('No "status_description" column found in input file. '
                    'Creating.')
        tsv_data['status_description'] = ['n/a'] * len(tsv_data['name'])

    # Update the sidecar data.
    if overwrite:
        # In cases where the "status" and / or "status_description"
        # columns were just created by us, we overwrite them again
        # here. This is not optimal in terms of performance, but
        # probably doesn't hurt anyone.
        logger.info('Resetting status and description for all channels.')
        tsv_data['status'] = ['good'] * len(tsv_data['name'])
        tsv_data['status_description'] = ['n/a'] * len(tsv_data['name'])

    # Now actually mark the user-requested channels as bad.
    for ch_name, description in zip(ch_names, descriptions):
        if ch_name not in tsv_data['name']:
            raise ValueError(f'Channel {ch_name} not found in dataset!')

        idx = tsv_data['name'].index(ch_name)
        logger.info(f'Processing channel {ch_name}:\n'
                    f'    status: bad\n'
                    f'    description: {description}')
        tsv_data['status'][idx] = 'bad'
        tsv_data['status_description'][idx] = description

    _write_tsv(channels_fname, tsv_data, overwrite=True, verbose=verbose)


def write_meg_calibration(calibration, bids_path, verbose=None):
    """Write the Elekta/Neuromag/MEGIN fine-calibration matrix to disk.

    Parameters
    ----------
    calibration : path-like | dict
        Either the path of the ``.dat`` file containing the file-calibration
        matrix, or the dictionary returned by
        :func:`mne.preprocessing.read_fine_calibration`.
    bids_path : mne_bids.BIDSPath
        A :class:`mne_bids.BIDSPath` instance with at least ``root`` and
        ``subject`` set,  and that ``datatype`` is either ``'meg'`` or
        ``None``.
    verbose : bool | None
         If a boolean, whether or not to produce verbose output. If ``None``,
         use the default log level.

    Examples
    --------
    >>> data_path = mne.datasets.testing.data_path(download=False)
    >>> calibration_fname = op.join(data_path, 'SSS', 'sss_cal_3053.dat')
    >>> bids_path = BIDSPath(subject='01', session='test',
    ...                      root=op.join(data_path, 'mne_bids'))
    >>> write_meg_calibration(calibration_fname, bids_path) # doctest: +ELLIPSIS
    Writing fine-calibration file to ...sub-01_ses-test_acq-calibration_meg.dat...
    """  # noqa: E501
    if bids_path.root is None or bids_path.subject is None:
        raise ValueError('bids_path must have root and subject set.')
    if bids_path.datatype not in (None, 'meg'):
        raise ValueError('Can only write fine-calibration information for MEG '
                         'datasets.')

    _validate_type(calibration, types=('path-like', dict),
                   item_name='calibration',
                   type_name='path or dictionary')

    if (isinstance(calibration, dict) and
            ('ch_names' not in calibration or
             'locs' not in calibration or
             'imb_cals' not in calibration)):
        raise ValueError('The dictionary you passed does not appear to be a '
                         'proper fine-calibration dict. Please only pass the '
                         'output of '
                         'mne.preprocessing.read_fine_calibration(), or a '
                         'filename.')

    if not isinstance(calibration, dict):
        calibration = mne.preprocessing.read_fine_calibration(calibration)

    out_path = BIDSPath(subject=bids_path.subject, session=bids_path.session,
                        acquisition='calibration', suffix='meg',
                        extension='.dat', datatype='meg', root=bids_path.root)

    logger.info(f'Writing fine-calibration file to {out_path}')
    out_path.mkdir()
    mne.preprocessing.write_fine_calibration(fname=str(out_path),
                                             calibration=calibration)


def write_meg_crosstalk(fname, bids_path, verbose=None):
    """Write the Elekta/Neuromag/MEGIN crosstalk information to disk.

    Parameters
    ----------
    fname : path-like
        The path of the ``FIFF`` file containing the crosstalk information.
    bids_path : mne_bids.BIDSPath
        A :class:`mne_bids.BIDSPath` instance with at least ``root`` and
        ``subject`` set,  and that ``datatype`` is either ``'meg'`` or
        ``None``.
    verbose : bool | None
         If a boolean, whether or not to produce verbose output. If ``None``,
         use the default log level.

    Examples
    --------
    >>> data_path = mne.datasets.testing.data_path(download=False)
    >>> crosstalk_fname = op.join(data_path, 'SSS', 'ct_sparse.fif')
    >>> bids_path = BIDSPath(subject='01', session='test',
    ...                      root=op.join(data_path, 'mne_bids'))
    >>> write_meg_crosstalk(crosstalk_fname, bids_path) # doctest: +ELLIPSIS
    Writing crosstalk file to ...sub-01_ses-test_acq-crosstalk_meg.fif
    """
    if bids_path.root is None or bids_path.subject is None:
        raise ValueError('bids_path must have root and subject set.')
    if bids_path.datatype not in (None, 'meg'):
        raise ValueError('Can only write fine-calibration information for MEG '
                         'datasets.')

    _validate_type(fname, types=('path-like',), item_name='fname')

    # MNE doesn't have public reader and writer functions for crosstalk data,
    # so just copy the original file. Use shutil.copyfile() to only copy file
    # contents, but not metadata & permissions.
    out_path = BIDSPath(subject=bids_path.subject, session=bids_path.session,
                        acquisition='crosstalk', suffix='meg',
                        extension='.fif', datatype='meg', root=bids_path.root)

    logger.info(f'Writing crosstalk file to {out_path}')
    out_path.mkdir()
    shutil.copyfile(src=fname, dst=str(out_path))
