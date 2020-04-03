"""Make BIDS compatible directory structures and infer meta data from MNE."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)
import os
import os.path as op
from datetime import datetime, date, timedelta, timezone
from warnings import warn
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
from mne.utils import check_version, has_nibabel, _check_ch_locs

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
                             BIDS_VERSION)


def _is_numeric(n):
    return isinstance(n, (np.integer, np.floating, int, float))


def _stamp_to_dt(utc_stamp):
    """Convert POSIX timestamp to datetime object in Windows-friendly way."""
    # This is a windows datetime bug for timestamp < 0. A negative value
    # is needed for anonymization which requires the date to be moved back
    # to before 1925. This then requires a negative value of daysback
    # compared the 1970 reference date.
    if isinstance(utc_stamp, datetime):
        return utc_stamp
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def _channels_tsv(raw, fname, overwrite=False, verbose=True):
    """Create a channels.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    fname : str
        Filename to save the channels.tsv to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false.

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
                    misc='Miscellaneous')
    get_specific = ('mag', 'ref_meg', 'grad')

    # get the manufacturer from the file in the Raw object
    manufacturer = None

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

    ch_data = OrderedDict([
        ('name', raw.info['ch_names']),
        ('type', ch_type),
        ('units', units),
        ('low_cutoff', np.full((n_channels), low_cutoff)),
        ('high_cutoff', np.full((n_channels), high_cutoff)),
        ('description', description),
        ('sampling_frequency', np.full((n_channels), sfreq)),
        ('status', status)])
    ch_data = _drop(ch_data, ignored_channels, 'name')

    _write_tsv(fname, ch_data, overwrite, verbose)

    return fname


def _electrodes_tsv(raw, fname, kind, overwrite=False, verbose=True):
    """
    Create an electrodes.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    fname : str
        Filename to save the electrodes.tsv to.
    kind : str
        Acquisition type to save electrodes in. For iEEG, requires size.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to true or false.
    """
    x, y, z, names = list(), list(), list(), list()
    for ch in raw.info['chs']:
        if _check_ch_locs([ch]):
            x.append(ch['loc'][0])
            y.append(ch['loc'][1])
            z.append(ch['loc'][2])
        else:
            x.append('n/a')
            y.append('n/a')
            z.append('n/a')
        names.append(ch['ch_name'])

    if kind == "ieeg":
        sizes = ['n/a'] * len(names)
        data = OrderedDict([('name', names),
                            ('x', x),
                            ('y', y),
                            ('z', z),
                            ('size', sizes),
                            ])
    else:
        data = OrderedDict([('name', names),
                            ('x', x),
                            ('y', y),
                            ('z', z),
                            ])
    _write_tsv(fname, data, overwrite=overwrite, verbose=verbose)
    return fname


def _events_tsv(events, raw, fname, trial_type, overwrite=False,
                verbose=True):
    """Create an events.tsv file and save it.

    This function will write the mandatory 'onset', and 'duration' columns as
    well as the optional 'value' and 'sample'. The 'value'
    corresponds to the marker value as found in the TRIG channel of the
    recording. In addition, the 'trial_type' field can be written.

    Parameters
    ----------
    events : array, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.
    raw : instance of Raw
        The data as MNE-Python Raw object.
    fname : str
        Filename to save the events.tsv to.
    trial_type : dict | None
        Dictionary mapping a brief description key to an event id (value). For
        example {'Go': 1, 'No Go': 2}.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false.

    Notes
    -----
    The function writes durations of zero for each event.

    """
    # Start by filling all data that we know into an ordered dictionary
    first_samp = raw.first_samp
    sfreq = raw.info['sfreq']
    events[:, 0] -= first_samp

    # Onset column needs to be specified in seconds
    data = OrderedDict([('onset', events[:, 0] / sfreq),
                        ('duration', np.zeros(events.shape[0])),
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

    return fname


def _participants_tsv(raw, subject_id, fname, overwrite=False,
                      verbose=True):
    """Create a participants.tsv file and save it.

    This will append any new participant data to the current list if it
    exists. Otherwise a new file will be created with the provided information.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    subject_id : str
        The subject name in BIDS compatible format ('01', '02', etc.)
    fname : str
        Filename to save the participants.tsv to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
        If there is already data for the given `subject_id` and overwrite is
        False, an error will be raised.
    verbose : bool
        Set verbose output to true or false.

    """
    subject_id = 'sub-' + subject_id
    data = OrderedDict(participant_id=[subject_id])

    subject_age = "n/a"
    sex = "n/a"
    hand = 'n/a'
    subject_info = raw.info['subject_info']
    if subject_info is not None:
        sexes = {0: 'n/a', 1: 'M', 2: 'F'}
        sex = sexes[subject_info.get('sex', 0)]

        # add handedness
        # XXX: MNE currently only handles R/L,
        # follow https://github.com/mne-tools/mne-python/issues/7347
        hand_options = {0: 'n/a', 1: 'R', 2: 'L', 3: 'A'}
        hand = hand_options[subject_info.get('hand', 0)]

        # determine the age of the participant
        age = subject_info.get('birthday', None)
        meas_date = raw.info.get('meas_date', None)
        if isinstance(meas_date, (tuple, list, np.ndarray)):
            meas_date = meas_date[0]

        if meas_date is not None and age is not None:
            bday = datetime(age[0], age[1], age[2], tzinfo=timezone.utc)
            meas_datetime = meas_date
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
            raise FileExistsError('"%s" already exists in the participant '  # noqa: E501 F821
                                  'list. Please set overwrite to '
                                  'True.' % subject_id)
        # otherwise add the new data
        data = _combine(orig_data, data, 'participant_id')

    # overwrite is forced to True as all issues with overwrite == False have
    # been handled by this point
    _write_tsv(fname, data, True, verbose)

    return fname


def _participants_json(fname, overwrite=False, verbose=True):
    """Create participants.json for non-default columns in accompanying TSV.

    Parameters
    ----------
    fname : str
        Filename to save the scans.tsv to.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to true or false.

    """
    cols = OrderedDict()
    cols['participant_id'] = {'Description': 'Unique participant identifier'}
    cols['age'] = {'Description': 'Age of the participant at time of testing',
                   'Units': 'years'}
    cols['sex'] = {'Description': 'Biological sex of the participant',
                   'Levels': {'F': 'female', 'M': 'male'}}
    cols['hand'] = {'Description': 'Handedness of the participant',
                    'Levels': {'R': 'right', 'L': 'left', 'A': 'ambidextrous'}}

    _write_json(fname, cols, overwrite, verbose)

    return fname


def _coordsystem_json(raw, unit, orient, coordsystem_name, fname,
                      kind, overwrite=False, verbose=True):
    """Create a coordsystem.json file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    unit : str
        Units to be used in the coordsystem specification.
    orient : str
        Used to define the coordinate system for the head coils.
    coordsystem_name : str
        Name of the coordinate system for the sensor positions.
    fname : str
        Filename to save the coordsystem.json to.
    kind : str
        Type of the data as in ALLOWED_KINDS.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false.

    """
    dig = raw.info['dig']
    coords = _extract_landmarks(dig)
    hpi = {d['ident']: d for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI}
    if hpi:
        for ident in hpi.keys():
            coords['coil%d' % ident] = hpi[ident]['r'].tolist()

    coord_frame = set([dig[ii]['coord_frame'] for ii in range(len(dig))])
    if len(coord_frame) > 1:  # noqa E501
        raise ValueError('All HPI, electrodes, and fiducials must be in the '
                         'same coordinate frame. Found: "{}"'
                         .format(coord_frame))

    if kind == 'meg':
        hpi = {d['ident']: d for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI}
        if hpi:
            for ident in hpi.keys():
                coords['coil%d' % ident] = hpi[ident]['r'].tolist()

        fid_json = {'MEGCoordinateSystem': coordsystem_name,
                    'MEGCoordinateUnits': unit,  # XXX validate this
                    'HeadCoilCoordinates': coords,
                    'HeadCoilCoordinateSystem': orient,
                    'HeadCoilCoordinateUnits': unit  # XXX validate this
                    }
    elif kind == "ieeg":
        fid_json = {
            'iEEGCoordinateSystem': coordsystem_name,  # MRI, Pixels, or ACPC
            'iEEGCoordinateUnits': unit,  # m (MNE), mm, cm , or pixels
        }

    _write_json(fname, fid_json, overwrite, verbose)

    return fname


def _scans_tsv(raw, raw_fname, fname, overwrite=False, verbose=True):
    """Create a scans.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    raw_fname : str
        Relative path to the raw data file.
    fname : str
        Filename to save the scans.tsv to.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to true or false.

    """
    # get measurement date from the data info
    meas_date = raw.info['meas_date']
    if meas_date is None:
        acq_time = 'n/a'
    elif isinstance(meas_date, (tuple, list, np.ndarray)):  # noqa: E501
        # for MNE < v0.20
        acq_time = _stamp_to_dt(meas_date).strftime('%Y-%m-%dT%H:%M:%S')
    elif isinstance(meas_date, datetime):
        # for MNE >= v0.20
        acq_time = meas_date.strftime('%Y-%m-%dT%H:%M:%S')

    data = OrderedDict([('filename', ['%s' % raw_fname.replace(os.sep, '/')]),
                       ('acq_time', [acq_time])])

    if os.path.exists(fname):
        orig_data = _from_tsv(fname)
        # if the file name is already in the file raise an error
        if raw_fname in orig_data['filename'] and not overwrite:
            raise FileExistsError('"%s" already exists in the scans list. '  # noqa: E501 F821
                                  'Please set overwrite to True.' % raw_fname)
        # otherwise add the new data
        data = _combine(orig_data, data, 'filename')

    # overwrite is forced to True as all issues with overwrite == False have
    # been handled by this point
    _write_tsv(fname, data, True, verbose)

    return fname


def _meg_landmarks_to_mri_landmarks(meg_landmarks, trans):
    """Convert landmarks from head space to MRI space.

    Parameters
    ----------
    meg_landmarks : array, shape (3, 3)
        The meg landmark data: rows LPA, NAS, RPA, columns x, y, z.
    trans : instance of mne.transforms.Transform
        The transformation matrix from head coordinates to MRI coordinates.

    Returns
    -------
    mri_landmarks : array, shape (3, 3)
        The mri ras landmark data converted to from m to mm.
    """
    # Transform MEG landmarks into MRI space, adjust units by * 1e3
    return apply_trans(trans, meg_landmarks, move=True) * 1e3


def _mri_landmarks_to_mri_voxels(mri_landmarks, t1_mgh):
    """Convert landmarks from MRI RAS space to MRI voxel space.

    Parameters
    ----------
    mri_landmarks : array, shape (3, 3)
        The MRI RAS landmark data: rows LPA, NAS, RPA, columns x, y, z.
    t1_mgh : nib.MGHImage
        The image data in MGH format.

    Returns
    -------
    mri_landmarks : array, shape (3, 3)
        The MRI voxel-space landmark data.
    """
    # Get landmarks in voxel space, using the T1 data
    vox2ras_tkr = t1_mgh.header.get_vox2ras_tkr()
    ras2vox_tkr = linalg.inv(vox2ras_tkr)
    mri_landmarks = apply_trans(ras2vox_tkr, mri_landmarks)  # in vox
    return mri_landmarks


def _sidecar_json(raw, task, manufacturer, fname, kind, overwrite=False,
                  verbose=True):
    """Create a sidecar json file depending on the kind and save it.

    The sidecar json file provides meta data about the data of a certain kind.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    task : str
        Name of the task the data is based on.
    manufacturer : str
        Manufacturer of the acquisition system. For MEG also used to define the
        coordinate system for the MEG sensors.
    fname : str
        Filename to save the sidecar json to.
    kind : str
        Type of the data as in ALLOWED_KINDS.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false. Defaults to true.

    """
    sfreq = raw.info['sfreq']
    powerlinefrequency = raw.info.get('line_freq', None)
    if powerlinefrequency is None:
        powerlinefrequency = _estimate_line_freq(raw)
        warn('No line frequency found, defaulting to {} Hz '
             'estimated from multi-taper FFT '
             'on 10 seconds of data.'.format(powerlinefrequency))

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

    # Define modality-specific JSON dictionaries
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
        ('DigitizedLandmarks', False),
        ('DigitizedHeadPoints', False),
        ('MEGChannelCount', n_megchan),
        ('MEGREFChannelCount', n_megrefchan)]
    ch_info_json_eeg = [
        ('EEGReference', 'n/a'),
        ('EEGGround', 'n/a'),
        ('EEGPlacementScheme', _infer_eeg_placement_scheme(raw)),
        ('Manufacturer', manufacturer)]
    ch_info_json_ieeg = [
        ('iEEGReference', 'n/a'),
        ('ECOGChannelCount', n_ecogchan),
        ('SEEGChannelCount', n_seegchan)]
    ch_info_ch_counts = [
        ('EEGChannelCount', n_eegchan),
        ('EOGChannelCount', n_eogchan),
        ('ECGChannelCount', n_ecgchan),
        ('EMGChannelCount', n_emgchan),
        ('MiscChannelCount', n_miscchan),
        ('TriggerChannelCount', n_stimchan)]

    # Stitch together the complete JSON dictionary
    ch_info_json = ch_info_json_common
    if kind == 'meg':
        append_kind_json = ch_info_json_meg
    elif kind == 'eeg':
        append_kind_json = ch_info_json_eeg
    elif kind == 'ieeg':
        append_kind_json = ch_info_json_ieeg

    ch_info_json += append_kind_json
    ch_info_json += ch_info_ch_counts
    ch_info_json = OrderedDict(ch_info_json)

    _write_json(fname, ch_info_json, overwrite, verbose)

    return fname


def _deface(t1w, mri_landmarks, deface):
    if not has_nibabel():  # pragma: no cover
        raise ImportError('This function requires nibabel.')
    import nibabel as nib

    inset, theta = (20, 35.)
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

    if not 0 < theta < 90:
        raise ValueError('theta should be between 0 and 90 '
                         'degrees. Got %s' % theta)

    # x: L/R L+, y: S/I I+, z: A/P A+
    t1w_data = t1w.get_fdata().copy()
    idxs_vox = np.meshgrid(np.arange(t1w_data.shape[0]),
                           np.arange(t1w_data.shape[1]),
                           np.arange(t1w_data.shape[2]),
                           indexing='ij')
    idxs_vox = np.array(idxs_vox)  # (3, *t1w_data.shape)
    idxs_vox = np.transpose(idxs_vox,
                            [1, 2, 3, 0])  # (*t1w_data.shape, 3)
    idxs_vox = idxs_vox.reshape(-1, 3)  # (n_voxels, 3)

    mri_landmarks_ras = apply_trans(t1w.affine, mri_landmarks)
    ras_meg_t = \
        get_ras_to_neuromag_trans(*mri_landmarks_ras[[1, 0, 2]])

    idxs_ras = apply_trans(t1w.affine, idxs_vox)
    idxs_meg = apply_trans(ras_meg_t, idxs_ras)

    # now comes the actual defacing
    # 1. move center of voxels to (nasion - inset)
    # 2. rotate the head by theta from the normal to the plane passing
    # through anatomical coordinates
    trans_y = -mri_landmarks_ras[1, 1] + inset
    idxs_meg = apply_trans(translation(y=trans_y), idxs_meg)
    idxs_meg = apply_trans(rotation(x=-np.deg2rad(theta)), idxs_meg)
    coords = idxs_meg.reshape(t1w.shape + (3,))  # (*t1w_data.shape, 3)
    mask = (coords[..., 2] < 0)   # z < 0

    t1w_data[mask] = 0.
    # smooth decided against for potential lack of anonymizaton
    # https://gist.github.com/alexrockhill/15043928b716a432db3a84a050b241ae

    t1w = nib.Nifti1Image(t1w_data, t1w.affine, t1w.header)
    return t1w


def _write_raw_fif(raw, bids_fname):
    """Save out the raw file in FIF.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw file to save out.
    bids_fname : str
        The name of the BIDS-specified file where the raw object
        should be saved.

    """
    n_rawfiles = len(raw.filenames)
    if n_rawfiles > 1:
        split_naming = 'bids'
        raw.save(bids_fname, split_naming=split_naming, overwrite=True)
    else:
        # This ensures that single FIF files do not have the part param
        raw.save(bids_fname, split_naming='neuromag', overwrite=True)


def _write_raw_brainvision(raw, bids_fname, events):
    """Save out the raw file in BrainVision format.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw file to save out.
    bids_fname : str
        The name of the BIDS-specified file where the raw object
        should be saved.

    """
    if not check_version('pybv', '0.2'):
        raise ImportError('pybv >=0.2.0 is required for converting ' +
                          'file to Brainvision format')
    from pybv import write_brainvision
    # Subtract raw.first_samp because brainvision marks events starting from
    # the first available data point and ignores the raw.first_samp
    if events is not None:
        events[:, 0] -= raw.first_samp
        events = events[:, [0, 2]]  # reorder for pybv required order
    meas_date = raw.info['meas_date']
    if meas_date is not None:
        meas_date = _stamp_to_dt(meas_date)
    write_brainvision(data=raw.get_data(), sfreq=raw.info['sfreq'],
                      ch_names=raw.ch_names,
                      fname_base=op.splitext(op.basename(bids_fname))[0],
                      folder_out=op.dirname(bids_fname),
                      events=events,
                      resolution=1e-9,
                      meas_date=meas_date)


def _get_anonymization_daysback(raw):
    """Get the min and max number of daysback necessary to satisfy BIDS specs.

    .. warning:: It is important that you remember the anonymization
                 number if you would ever like to de-anonymize but
                 that it is not included in the code publication
                 as that would break the anonymization.

    BIDS requires that anonymized dates be before 1925. In order to
    preserve the longitudinal structure and ensure anonymization, the
    user is asked to provide the same `daysback` argument to each call
    of `write_raw_bids`. To determine the miniumum number of daysback
    necessary, this function will calculate the minimum number based on
    the most recent measurement date of raw objects.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        The raw data. It must be an instance or list of instances of mne.Raw.

    Returns
    -------
    daysback_min : int
        The minimum number of daysback necessary to be compatible with BIDS.
    daysback_max : int
        The maximum number of daysback that MNE can store.
    """
    this_date = _stamp_to_dt(raw.info['meas_date']).date()
    daysback_min = (this_date - date(year=1924, month=12, day=31)).days
    daysback_max = (this_date - datetime.fromtimestamp(0).date() +
                    timedelta(seconds=np.iinfo('>i4').max)).days
    return daysback_min, daysback_max


def get_anonymization_daysback(raws):
    """Get the group min and max number of daysback necessary for BIDS specs.

    .. warning:: It is important that you remember the anonymization
                 number if you would ever like to de-anonymize but
                 that it is not included in the code publication
                 as that would break the anonymization.

    BIDS requires that anonymized dates be before 1925. In order to
    preserve the longitudinal structure and ensure anonymization, the
    user is asked to provide the same `daysback` argument to each call
    of `write_raw_bids`. To determine the miniumum number of daysback
    necessary, this function will calculate the minimum number based on
    the most recent measurement date of raw objects.

    Parameters
    ----------
    raw : mne.io.Raw | list of Raw
        The group raw data. It must be a list of instances of mne.Raw.

    Returns
    -------
    daysback_min : int
        The minimum number of daysback necessary to be compatible with BIDS.
    daysback_max : int
        The maximum number of daysback that MNE can store.
    """
    if not isinstance(raws, list):
        raws = list([raws])
    daysback_min_list = list()
    daysback_max_list = list()
    for raw in raws:
        if raw.info['meas_date'] is not None:
            daysback_min, daysback_max = _get_anonymization_daysback(raw)
            daysback_min_list.append(daysback_min)
            daysback_max_list.append(daysback_max)
    if not daysback_min_list or not daysback_max_list:
        raise ValueError('All measurement dates are None, ' +
                         'pass any `daysback` value to anonymize.')
    daysback_min = max(daysback_min_list)
    daysback_max = min(daysback_max_list)
    if daysback_min > daysback_max:
        raise ValueError('The dataset spans more time than can be ' +
                         'accomodated by MNE, you may have to ' +
                         'not follow BIDS recommendations and use' +
                         'anonymized dates after 1925')
    return daysback_min, daysback_max


def make_bids_basename(subject=None, session=None, task=None,
                       acquisition=None, run=None, processing=None,
                       recording=None, space=None, prefix=None, suffix=None):
    """Create a partial/full BIDS filename from its component parts.

    BIDS filename prefixes have one or more pieces of metadata in them. They
    must follow a particular order, which is followed by this function. This
    will generate the *prefix* for a BIDS filename that can be used with many
    subsequent files, or you may also give a suffix that will then complete
    the file name.

    Note that all parameters are not applicable to each kind of data. For
    example, electrode location TSV files do not need a task field.

    Parameters
    ----------
    subject : str | None
        The subject ID. Corresponds to "sub".
    session : str | None
        The session identifier. Corresponds to "ses". Must be a date in
        format "YYYYMMDD" if subject is "emptyroom".
    task : str | None
        The task identifier. Corresponds to "task". Must be "noise" if
        subject is "emptyroom".
    acquisition: str | None
        The acquisition parameters. Corresponds to "acq".
    run : int | None
        The run number. Corresponds to "run".
    processing : str | None
        The processing label. Corresponds to "proc".
    recording : str | None
        The recording name. Corresponds to "recording".
    space : str | None
        The coordinate space for an anatomical file. Corresponds to "space".
    prefix : str | None
        The prefix for the filename to be created. E.g., a path to the folder
        in which you wish to create a file with this name.
    suffix : str | None
        The suffix for the filename to be created. E.g., 'audio.wav'.

    Returns
    -------
    filename : str
        The BIDS filename you wish to create.

    Examples
    --------
    >>> print(make_bids_basename(subject='test', session='two', task='mytask', suffix='data.csv')) # noqa: E501
    sub-test_ses-two_task-mytask_data.csv

    """
    order = OrderedDict([('sub', subject),
                         ('ses', session),
                         ('task', task),
                         ('acq', acquisition),
                         ('run', run),
                         ('proc', processing),
                         ('space', space),
                         ('recording', recording)])
    if order['run'] is not None and not isinstance(order['run'], str):
        # Ensure that run is a string
        order['run'] = '{:02}'.format(order['run'])

    _check_types(order.values())

    if (all(ii is None for ii in order.values()) and suffix is None and
            prefix is None):
        raise ValueError("At least one parameter must be given.")

    if subject == 'emptyroom':
        if suffix is None and task != 'noise':
            raise ValueError('task must be ''noise'' if subject is'
                             ' ''emptyroom''')
        try:
            datetime.strptime(session, '%Y%m%d')
        except ValueError:
            raise ValueError("session must be string of format YYYYMMDD")

    filename = []
    for key, val in order.items():
        if val is not None:
            _check_key_val(key, val)
            filename.append('%s-%s' % (key, val))

    if isinstance(suffix, str):
        filename.append(suffix)

    filename = '_'.join(filename)
    if isinstance(prefix, str):
        filename = op.join(prefix, filename)
    return filename


def make_dataset_description(path, name, data_license=None,
                             authors=None, acknowledgements=None,
                             how_to_acknowledge=None, funding=None,
                             references_and_links=None, doi=None,
                             verbose=False):
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
        The license under which this datset is published.
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

    Notes
    -----
    The required field BIDSVersion will be automatically filled by mne_bids.

    """
    # default author to make dataset description BIDS compliant
    if authors is None:
        authors = "MNE-BIDS"

    # Put potential string input into list of strings
    if isinstance(authors, str):
        authors = authors.split(', ')
    if isinstance(funding, str):
        funding = funding.split(', ')
    if isinstance(references_and_links, str):
        references_and_links = references_and_links.split(', ')

    fname = op.join(path, 'dataset_description.json')
    description = OrderedDict([('Name', name),
                               ('BIDSVersion', BIDS_VERSION),
                               ('License', data_license),
                               ('Authors', authors),
                               ('Acknowledgements', acknowledgements),
                               ('HowToAcknowledge', how_to_acknowledge),
                               ('Funding', funding),
                               ('ReferencesAndLinks', references_and_links),
                               ('DatasetDOI', doi)])
    pop_keys = [key for key, val in description.items() if val is None]
    for key in pop_keys:
        description.pop(key)
    _write_json(fname, description, overwrite=True, verbose=verbose)


def write_raw_bids(raw, bids_basename, bids_root, events_data=None,
                   event_id=None, anonymize=None,
                   overwrite=False, verbose=True):
    """Save raw data to a BIDS-compliant folder structure.

    .. warning:: * The original file is simply copied over if the original
                   file format is BIDS-supported for that modality. Otherwise,
                   this function will convert to a BIDS-supported file format
                   while warning the user. For EEG and iEEG data, conversion
                   will be to BrainVision format, for MEG conversion will be
                   to FIF.

                 * ``mne-bids`` will infer the manufacturer information
                   from the file extension. If your file format is non-standard
                   for the manufacturer, please update the manufacturer field
                   in the sidecars manually.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        The raw data. It must be an instance of mne.Raw. The data should not be
        loaded from disk, i.e., raw.preload must be False.
    bids_basename : str
        The base filename of the BIDS compatible files. Typically, this can be
        generated using make_bids_basename.
        Example: `sub-01_ses-01_task-testing_acq-01_run-01`.
        This will write the following files in the correct subfolder of the
        bids_root::

            sub-01_ses-01_task-testing_acq-01_run-01_meg.fif
            sub-01_ses-01_task-testing_acq-01_run-01_meg.json
            sub-01_ses-01_task-testing_acq-01_run-01_channels.tsv
            sub-01_ses-01_task-testing_acq-01_run-01_coordsystem.json

        and the following one if events_data is not None::

            sub-01_ses-01_task-testing_acq-01_run-01_events.tsv

        and add a line to the following files::

            participants.tsv
            scans.tsv

        Note that the modality 'meg' is automatically inferred from the raw
        object and extension '.fif' is copied from raw.filenames.
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder. The session and
        subject specific folders will be populated automatically by parsing
        bids_basename.
    events_data : str | pathlib.Path | array | None
        The events file. If a string or a Path object, specifies the path of
        the events file. If an array, the MNE events array (shape n_events, 3).
        If None, events will be inferred from the stim channel using
        `mne.find_events`.
    event_id : dict | None
        The event id dict used to create a 'trial_type' column in events.tsv
    anonymize : dict | None
        If None is provided (default) no anonymization is performed.
        If a dictionary is passed, data will be anonymized; identifying data
        structures such as study date and time will be changed.
        `daysback` is a required argument and `keep_his` is optional, these
        arguments are passed to :func:`mne.io.anonymize_info`.

        `daysback` : int
            Number of days to move back the date. To keep relative dates for
            a subject use the same `daysback`. According to BIDS
            specifications, the number of days back must be great enough
            that the date is before 1925.

        `keep_his` : bool
            If True info['subject_info']['his_id'] of subject_info will NOT be
            overwritten. Defaults to False.

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
        If verbose is True, this will print a snippet of the sidecar files. If
        False, no content will be printed.

    Returns
    -------
    bids_root : str
        The path of the root of the BIDS compatible folder.

    Notes
    -----
    For the participants.tsv file, the raw.info['subject_info'] should be
    updated and raw.info['meas_date'] should not be None to compute the age
    of the participant correctly.

    """
    if not check_version('mne', '0.17'):
        raise ValueError('Your version of MNE is too old. '
                         'Please update to 0.17 or newer.')

    if not check_version('mne', '0.20') and anonymize:
        raise ValueError('Your version of MNE is too old to be able to '
                         'anonymize the data on write. Please update to '
                         '0.20.dev version or newer.')

    if not isinstance(raw, BaseRaw):
        raise ValueError('raw_file must be an instance of BaseRaw, '
                         'got %s' % type(raw))

    if not hasattr(raw, 'filenames') or raw.filenames[0] is None:
        raise ValueError('raw.filenames is missing. Please set raw.filenames'
                         'as a list with the full path of original raw file.')

    if raw.preload is not False:
        raise ValueError('The data should not be preloaded.')

    bids_root = _path_to_str(bids_root)
    raw = raw.copy()

    raw_fname = raw.filenames[0]
    if '.ds' in op.dirname(raw.filenames[0]):
        raw_fname = op.dirname(raw.filenames[0])
    # point to file containing header info for multifile systems
    raw_fname = raw_fname.replace('.eeg', '.vhdr')
    raw_fname = raw_fname.replace('.fdt', '.set')
    _, ext = _parse_ext(raw_fname, verbose=verbose)

    if ext not in [this_ext for data_type in ALLOWED_EXTENSIONS
                   for this_ext in ALLOWED_EXTENSIONS[data_type]]:
        raise ValueError('Unrecognized file format %s' % ext)

    raw_orig = reader[ext](**raw._init_kwargs)
    assert_array_equal(raw.times, raw_orig.times,
                       "raw.times should not have changed since reading"
                       " in from the file. It may have been cropped.")

    params = _parse_bids_filename(bids_basename, verbose)
    subject_id, session_id = params['sub'], params['ses']
    acquisition, task, run = params['acq'], params['task'], params['run']
    kind = _handle_kind(raw)

    bids_fname = bids_basename + '_%s%s' % (kind, ext)

    # check whether the info provided indicates that the data is emptyroom
    # data
    emptyroom = False
    if subject_id == 'emptyroom' and task == 'noise':
        emptyroom = True
        # check the session date provided is consistent with the value in raw
        meas_date = raw.info.get('meas_date', None)
        if meas_date is not None:
            if not isinstance(meas_date, datetime):
                meas_date = datetime.fromtimestamp(meas_date[0],
                                                   tz=timezone.utc)
            er_date = meas_date.strftime('%Y%m%d')
            if er_date != session_id:
                raise ValueError("Date provided for session doesn't match "
                                 "session date.")

    data_path = make_bids_folders(subject=subject_id, session=session_id,
                                  kind=kind, bids_root=bids_root,
                                  overwrite=False, verbose=verbose)
    if session_id is None:
        ses_path = os.sep.join(data_path.split(os.sep)[:-1])
    else:
        ses_path = make_bids_folders(subject=subject_id, session=session_id,
                                     bids_root=bids_root, make_dir=False,
                                     overwrite=False, verbose=verbose)

    # create filenames
    scans_fname = make_bids_basename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=ses_path)
    participants_tsv_fname = make_bids_basename(prefix=bids_root,
                                                suffix='participants.tsv')
    participants_json_fname = make_bids_basename(prefix=bids_root,
                                                 suffix='participants.json')
    coordsystem_fname = make_bids_basename(
        subject=subject_id, session=session_id, acquisition=acquisition,
        suffix='coordsystem.json', prefix=data_path)
    sidecar_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acquisition, suffix='%s.json' % kind, prefix=data_path)
    events_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task,
        acquisition=acquisition, run=run, suffix='events.tsv',
        prefix=data_path)
    channels_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acquisition, suffix='channels.tsv', prefix=data_path)
    if ext not in ['.fif', '.ds', '.vhdr', '.edf', '.bdf', '.set', '.con',
                   '.sqd']:
        bids_raw_folder = bids_fname.split('.')[0]
        bids_fname = op.join(bids_raw_folder, bids_fname)

    # Anonymize
    if anonymize is not None:
        # if info['meas_date'] None, then the dates are not stored
        if raw.info['meas_date'] is None:
            daysback = None
        else:
            if 'daysback' not in anonymize or anonymize['daysback'] is None:
                raise ValueError('`daysback` argument required to anonymize.')
            daysback = anonymize['daysback']
            daysback_min, daysback_max = _get_anonymization_daysback(raw)
            if daysback < daysback_min:
                warn('`daysback` is too small; the measurement date '
                     'is after 1925, which is not recommended by BIDS.'
                     'The minimum `daysback` value for changing the '
                     'measurement date of this data to before this date '
                     'is %i' % daysback_min)
            if ext == '.fif' and daysback > daysback_max:
                raise ValueError('`daysback` exceeds maximum value MNE '
                                 'is able to store in FIF format, must '
                                 'be less than %i' % daysback_max)
        keep_his = anonymize['keep_his'] if 'keep_his' in anonymize else False
        raw.info = anonymize_info(raw.info, daysback=daysback,
                                  keep_his=keep_his)
        if kind == 'meg' and ext != '.fif':
            if verbose:
                warn('Converting to FIF for anonymization')
            bids_fname = bids_fname.replace(ext, '.fif')
        elif kind in ['eeg', 'ieeg']:
            if verbose:
                warn('Converting to BV for anonymization')
            bids_fname = bids_fname.replace(ext, '.vhdr')

    # Read in Raw object and extract metadata from Raw object if needed
    orient = ORIENTATION.get(ext, 'n/a')
    unit = UNITS.get(ext, 'n/a')
    manufacturer = MANUFACTURERS.get(ext, 'n/a')

    # save all meta data
    _participants_tsv(raw, subject_id, participants_tsv_fname, overwrite,
                      verbose)
    _participants_json(participants_json_fname, True, verbose)
    _scans_tsv(raw, op.join(kind, bids_fname), scans_fname, overwrite, verbose)

    # TODO: Implement coordystem.json and electrodes.tsv for EEG
    electrodes_fname = make_bids_basename(
        subject=subject_id, session=session_id, acquisition=acquisition,
        suffix='electrodes.tsv', prefix=data_path)
    if kind == 'meg' and not emptyroom:
        _coordsystem_json(raw, unit, orient,
                          manufacturer, coordsystem_fname, kind,
                          overwrite, verbose)
    elif kind == 'ieeg':
        coord_frame = "mri"  # defaults to MRI coordinates
        unit = "m"  # defaults to meters
    else:
        coord_frame = None
        unit = None

    # We only write electrodes.tsv and accompanying coordsystem.json
    # if we have an available DigMontage
    if raw.info['dig'] is not None:
        if kind == "ieeg":
            coords = _extract_landmarks(raw.info['dig'])

            # Rescale to MNE-Python "head" coord system, which is the
            # "ElektaNeuromag" system (equivalent to "CapTrak" system)
            if set(['RPA', 'NAS', 'LPA']) != set(list(coords.keys())):
                # Now write the data to the elec coords and the coordsystem
                _electrodes_tsv(raw, electrodes_fname,
                                kind, overwrite, verbose)
                _coordsystem_json(raw, unit, orient,
                                  coord_frame, coordsystem_fname, kind,
                                  overwrite, verbose)
        elif kind != "meg":
            warn('Writing of electrodes.tsv is not supported for kind "{}". '
                 'Skipping ...'.format(kind))

    events, event_id = _read_events(events_data, event_id, raw, ext)
    if events is not None and len(events) > 0 and not emptyroom:
        _events_tsv(events, raw, events_fname, event_id, overwrite, verbose)

    dataset_description_fpath = op.join(bids_root, "dataset_description.json")
    if not op.exists(dataset_description_fpath) or overwrite:
        make_dataset_description(bids_root, name=" ", verbose=verbose)

    _sidecar_json(raw, task, manufacturer, sidecar_fname, kind, overwrite,
                  verbose)
    _channels_tsv(raw, channels_fname, overwrite, verbose)

    # set the raw file name to now be the absolute path to ensure the files
    # are placed in the right location
    bids_fname = op.join(data_path, bids_fname)
    if os.path.exists(bids_fname) and not overwrite:
        raise FileExistsError('"%s" already exists. Please set '  # noqa: F821
                              'overwrite to True.' % bids_fname)
    _mkdir_p(os.path.dirname(bids_fname))

    convert = ext not in ALLOWED_EXTENSIONS[kind]

    if kind == 'meg' and convert and not anonymize:
        raise ValueError('Got file extension %s for MEG data, ' +
                         'expected one of %s' %
                         ALLOWED_EXTENSIONS['meg'])

    if not (convert or anonymize) and verbose:
        print('Copying data files to %s' % op.splitext(bids_fname)[0])

    # File saving branching logic
    if convert or (anonymize is not None):
        if kind == 'meg':
            if ext == '.pdf':
                bids_fname = op.join(data_path, op.basename(bids_fname))
            _write_raw_fif(raw, bids_fname)
        else:
            if verbose:
                warn('Converting data files to BrainVision format')
            _write_raw_brainvision(raw, bids_fname, events)
    elif ext == '.fif':
        _write_raw_fif(raw, bids_fname)
    # CTF data is saved and renamed in a directory
    elif ext == '.ds':
        copyfile_ctf(raw_fname, bids_fname)
    # BrainVision is multifile, copy over all of them and fix pointers
    elif ext == '.vhdr':
        copyfile_brainvision(raw_fname, bids_fname)
    # EEGLAB .set might be accompanied by a .fdt - find out and copy it too
    elif ext == '.set':
        copyfile_eeglab(raw_fname, bids_fname)
    elif ext == '.pdf':
        copyfile_bti(raw_orig, op.join(data_path, bids_raw_folder))
    elif ext == '.con' or '.sqd':
        copyfile_kit(raw_fname, bids_fname, subject_id, session_id,
                     task, run, raw._init_kwargs)
    else:
        sh.copyfile(raw_fname, bids_fname)

    return bids_root


def write_anat(bids_root, subject, t1w, session=None, acquisition=None,
               raw=None, trans=None, landmarks=None, deface=False,
               overwrite=False, verbose=False):
    """Put anatomical MRI data into a BIDS format.

    Given a BIDS directory and a T1 weighted MRI scan for a certain subject,
    format the MRI scan to be in BIDS format and put it into the correct
    location in the bids_dir. If a transformation matrix is supplied, a
    sidecar JSON file will be written for the T1 weighted data.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        Path to root of the BIDS folder
    subject : str
        Subject label as in 'sub-<label>', for example: '01'
    t1w : str | pathlib.Path | nibabel image object
        Path to a T1 weighted MRI scan of the subject. Can be in any format
        readable by nibabel. Can also be a nibabel image object of a T1
        weighted MRI scan. Will be written as a .nii.gz file.
    session : str | None
        The session for `t1w`. Corresponds to "ses"
    acquisition: str | None
        The acquisition parameters for `t1w`. Corresponds to "acq"
    raw : instance of Raw | None
        The raw data of `subject` corresponding to `t1w`. If `raw` is None,
        `trans` has to be None as well
    trans : instance of mne.transforms.Transform | str | None
        The transformation matrix from head coordinates to MRI coordinates. Can
        also be a string pointing to a .trans file containing the
        transformation matrix. If None, no sidecar JSON file will be written
        for `t1w`
    deface : bool | dict
        If False, no defacing is performed.
        If True, deface with default parameters.
        `trans` and `raw` must not be `None` if True.
        If dict, accepts the following keys:

        - `inset`: how far back in millimeters to start defacing
          relative to the nasion (default 20)

        - `theta`: is the angle of the defacing shear in degrees relative
          to the normal to the plane passing through the anatomical
          landmarks (default 35).

    landmarks: instance of DigMontage | str
        The DigMontage or filepath to a DigMontage with landmarks that can be
        passed to provide information for defacing. Landmarks can be determined
        from the head model using `mne coreg` GUI, or they can be determined
        from the MRI using `freeview`.
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
        If verbose is True, this will print a snippet of the sidecar files. If
        False, no content will be printed.

    Returns
    -------
    anat_dir : str
        Path to the anatomical scan in the `bids_dir`

    """
    if not has_nibabel():  # pragma: no cover
        raise ImportError('This function requires nibabel.')
    import nibabel as nib

    if deface and (trans is None or raw is None) and landmarks is None:
        raise ValueError('The raw object, trans and raw or the landmarks '
                         'must be provided to deface the T1')

    # Make directory for anatomical data
    anat_dir = op.join(bids_root, 'sub-{}'.format(subject))
    # Session is optional
    if session is not None:
        anat_dir = op.join(anat_dir, 'ses-{}'.format(session))
    anat_dir = op.join(anat_dir, 'anat')
    if not op.exists(anat_dir):
        os.makedirs(anat_dir)

    # Try to read our T1 file and convert to MGH representation
    try:
        t1w = _path_to_str(t1w)
    except ValueError:
        # t1w -> str conversion failed, so maybe the user passed an nibabel
        # object instead of a path.
        if type(t1w) not in nib.all_image_classes:
            raise ValueError('`t1w` must be a path to a T1 weighted MRI data '
                             'file , or a nibabel image object, but it is of '
                             'type "{}"'.format(type(t1w)))
    else:
        # t1w -> str conversion in the try block was successful, so load the
        # file from the specified location. We do this here and not in the try
        # block to keep the try block as short as possible.
        t1w = nib.load(t1w)

    t1w = nib.Nifti1Image(t1w.dataobj, t1w.affine)
    # XYZT_UNITS = NIFT_UNITS_MM (10 in binary or 2 in decimal)
    # seems to be the default for Nifti files
    # https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/xyzt_units.html
    if t1w.header['xyzt_units'] == 0:
        t1w.header['xyzt_units'] = np.array(10, dtype='uint8')

    # Now give the NIfTI file a BIDS name and write it to the BIDS location
    t1w_basename = make_bids_basename(subject=subject, session=session,
                                      acquisition=acquisition, prefix=anat_dir,
                                      suffix='T1w.nii.gz')

    # Check if we have necessary conditions for writing a sidecar JSON
    if trans is not None or landmarks is not None:
        t1_mgh = nib.MGHImage(t1w.dataobj, t1w.affine)

        if landmarks is not None:
            if raw is not None:
                raise ValueError('Please use either `landmarks` or `raw`, ' +
                                 'which digitization to use is ambiguous.')
            if isinstance(landmarks, str):
                landmarks, coord_frame = read_fiducials(landmarks)
                landmarks = np.array([landmark['r'] for landmark in
                                      landmarks], dtype=float)  # unpack
            else:
                coords_dict, coord_frame = _get_fid_coords(landmarks.dig)
                landmarks = np.asarray((coords_dict['lpa'],
                                        coords_dict['nasion'],
                                        coords_dict['rpa']))
            if coord_frame == FIFF.FIFFV_COORD_HEAD:
                if trans is None:
                    raise ValueError('Head space landmarks provided, ' +
                                     '`trans` required')
                mri_landmarks = _meg_landmarks_to_mri_landmarks(
                    landmarks, trans)
                mri_landmarks = _mri_landmarks_to_mri_voxels(
                    mri_landmarks, t1_mgh)
            elif coord_frame in (FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                                 FIFF.FIFFV_COORD_MRI):
                if trans is not None:
                    raise ValueError('`trans` was provided but `landmark` ' +
                                     'data is in mri space. Please use ' +
                                     'only one of these.')
                if coord_frame == FIFF.FIFFV_COORD_MRI:
                    landmarks = _mri_landmarks_to_mri_voxels(landmarks * 1e3,
                                                             t1_mgh)
                mri_landmarks = landmarks
            else:
                raise ValueError('Coordinate frame not recognized, ' +
                                 'found %s' % coord_frame)
        elif trans is not None:
            # get trans and ensure it is from head to MRI
            trans, _ = _get_trans(trans, fro='head', to='mri')

            if not isinstance(raw, BaseRaw):
                raise ValueError('`raw` must be specified if `trans` ' +
                                 'is not None')

            # Prepare to write the sidecar JSON
            # extract MEG landmarks
            coords_dict, coord_fname = _get_fid_coords(raw.info['dig'])
            meg_landmarks = np.asarray((coords_dict['lpa'],
                                        coords_dict['nasion'],
                                        coords_dict['rpa']))
            mri_landmarks = _meg_landmarks_to_mri_landmarks(
                meg_landmarks, trans)
            mri_landmarks = _mri_landmarks_to_mri_voxels(
                mri_landmarks, t1_mgh)

        # Write sidecar.json
        t1w_json = dict()
        t1w_json['AnatomicalLandmarkCoordinates'] = \
            {'LPA': list(mri_landmarks[0, :]),
             'NAS': list(mri_landmarks[1, :]),
             'RPA': list(mri_landmarks[2, :])}
        fname = t1w_basename.replace('.nii.gz', '.json')
        if op.isfile(fname) and not overwrite:
            raise IOError('Wanted to write a file but it already exists and '
                          '`overwrite` is set to False. File: "{}"'
                          .format(fname))
        _write_json(fname, t1w_json, overwrite, verbose)

        if deface:
            t1w = _deface(t1w, mri_landmarks, deface)

    # Save anatomical data
    if op.exists(t1w_basename):
        if overwrite:
            os.remove(t1w_basename)
        else:
            raise IOError('Wanted to write a file but it already exists and '
                          '`overwrite` is set to False. File: "{}"'
                          .format(t1w_basename))

    nib.save(t1w, t1w_basename)

    return anat_dir
