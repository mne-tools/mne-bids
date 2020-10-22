"""Utility and helper functions for MNE-BIDS."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)
import json
import os
import re
from datetime import datetime, date, timedelta, timezone
from os import path as op

import numpy as np
from mne import read_events, events_from_annotations
from mne.channels import make_standard_montage
from mne.io.constants import FIFF
from mne.io.kit.kit import get_kit_info
from mne.io.pick import pick_types
from mne.utils import warn, logger

from mne_bids.tsv_handler import _to_tsv, _tsv_to_str


# This regex matches key-val pairs. Any characters are allowed in the key and
# the value, except these special symbols: - _ . \ /
param_regex = re.compile(r'([^-_\.\\\/]+)-([^-_\.\\\/]+)')


def _ensure_tuple(x):
    """Return a tuple."""
    if x is None:
        return tuple()
    elif isinstance(x, str):
        return (x,)
    else:
        return tuple(x)


def _get_ch_type_mapping(fro='mne', to='bids'):
    """Map between BIDS and MNE nomenclatures for channel types.

    Parameters
    ----------
    fro : str
        Mapping from nomenclature of `fro`. Can be 'mne', 'bids'
    to : str
        Mapping to nomenclature of `to`. Can be 'mne', 'bids'

    Returns
    -------
    mapping : dict
        Dictionary mapping from one nomenclature of channel types to another.
        If a key is not present, a default value will be returned that depends
        on the `fro` and `to` parameters.

    Notes
    -----
    For the mapping from BIDS to MNE, MEG channel types are ignored for now.
    Furthermore, this is not a one-to-one mapping: Incomplete and partially
    one-to-many/many-to-one.

    Bio channels are supported in mne-python and are converted to MISC
    because there is no "Bio" supported channel in BIDS.
    """
    if fro == 'mne' and to == 'bids':
        mapping = dict(eeg='EEG', misc='MISC', stim='TRIG', emg='EMG',
                       ecog='ECOG', seeg='SEEG', eog='EOG', ecg='ECG',
                       resp='RESP', bio='MISC',
                       # MEG channels
                       meggradaxial='MEGGRADAXIAL', megmag='MEGMAG',
                       megrefgradaxial='MEGREFGRADAXIAL',
                       meggradplanar='MEGGRADPLANAR', megrefmag='MEGREFMAG')

    elif fro == 'bids' and to == 'mne':
        mapping = dict(EEG='eeg', MISC='misc', TRIG='stim', EMG='emg',
                       ECOG='ecog', SEEG='seeg', EOG='eog', ECG='ecg',
                       RESP='resp',
                       # No MEG channels for now
                       # Many to one mapping
                       VEOG='eog', HEOG='eog')
    else:
        raise ValueError('Only two types of mappings are currently supported: '
                         'from mne to bids, or from bids to mne. However, '
                         'you specified from "{}" to "{}"'.format(fro, to))

    return mapping


def _handle_datatype(raw):
    """Get datatype."""
    if 'eeg' in raw and ('ecog' in raw or 'seeg' in raw):
        raise ValueError('Both EEG and iEEG channels found in data.'
                         'There is currently no specification on how'
                         'to handle this data. Please proceed manually.')
    elif 'meg' in raw:
        datatype = 'meg'
    elif 'ecog' in raw or 'seeg' in raw:
        datatype = 'ieeg'
    elif 'eeg' in raw:
        datatype = 'eeg'
    else:
        raise ValueError('Neither MEG/EEG/iEEG channels found in data.'
                         'Please use raw.set_channel_types to set the '
                         'channel types in the data.')
    return datatype


def _age_on_date(bday, exp_date):
    """Calculate age from birthday and experiment date.

    Parameters
    ----------
    bday : instance of datetime.datetime
        The birthday of the participant.
    exp_date : instance of datetime.datetime
        The date the experiment was performed on.

    """
    if exp_date < bday:
        raise ValueError("The experimentation date must be after the birth "
                         "date")
    if exp_date.month > bday.month:
        return exp_date.year - bday.year
    elif exp_date.month == bday.month:
        if exp_date.day >= bday.day:
            return exp_date.year - bday.year
    return exp_date.year - bday.year - 1


def _check_types(variables):
    """Make sure all vars are str or None."""
    for var in variables:
        if not isinstance(var, (str, type(None))):
            raise ValueError(f"You supplied a value ({var}) of type "
                             f"{type(var)}, where a string or None was "
                             f"expected.")


def _write_json(fname, dictionary, overwrite=False, verbose=False):
    """Write JSON to a file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError(f'"{fname}" already exists. '
                              'Please set overwrite to True.')

    json_output = json.dumps(dictionary, indent=4)
    with open(fname, 'w') as fid:
        fid.write(json_output)
        fid.write('\n')

    if verbose is True:
        print(os.linesep + f"Writing '{fname}'..." + os.linesep)
        print(json_output)


def _write_tsv(fname, dictionary, overwrite=False, verbose=False):
    """Write an ordered dictionary to a .tsv file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError(f'"{fname}" already exists. '
                              'Please set overwrite to True.')
    _to_tsv(dictionary, fname)

    if verbose:
        print(os.linesep + f"Writing '{fname}'..." + os.linesep)
        print(_tsv_to_str(dictionary))


def _write_text(fname, text, overwrite=False, verbose=True):
    """Write text to a file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError(f'"{fname}" already exists. '
                              'Please set overwrite to True.')
    with open(fname, 'w') as fid:
        fid.write(text)
        fid.write('\n')

    if verbose:
        print(os.linesep + f"Writing '{fname}'..." + os.linesep)
        print(text)


def _check_key_val(key, val):
    """Perform checks on a value to make sure it adheres to the spec."""
    if any(ii in val for ii in ['-', '_', '/']):
        raise ValueError("Unallowed `-`, `_`, or `/` found in key/value pair"
                         f" {key}: {val}")
    return key, val


def _read_events(events_data, event_id, raw, verbose=None):
    """Read in events data.

    Parameters
    ----------
    events_data : str | array | None
        The events file. If a string, a path to the events file. If an array,
        the MNE events array (shape n_events, 3). If None, events will be
        inferred from the stim channel using `find_events`.
    event_id : dict
        The event id dict used to create a 'trial_type' column in events.tsv,
        mapping a description key to an integer valued event code.
    raw : instance of Raw
        The data as MNE-Python Raw object.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    events : array, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.

    """
    # get events from events_data
    if isinstance(events_data, str):
        events = read_events(events_data, verbose=verbose).astype(int)
    elif isinstance(events_data, np.ndarray):
        if events_data.ndim != 2:
            raise ValueError('Events must have two dimensions, '
                             f'found {events_data.ndim}')
        if events_data.shape[1] != 3:
            raise ValueError('Events must have second dimension of length 3, '
                             f'found {events_data.shape[1]}')
        events = events_data
    else:
        events = np.empty(shape=(0, 3), dtype=int)

    # get events from annotations
    events_annot, event_id_annot = events_from_annotations(raw, event_id,
                                                           verbose=verbose)
    if event_id is None:
        events = events_annot
        event_id = event_id_annot
    else:
        events = np.concatenate((events, events_annot), axis=0)
        # Sort rows by onset sample.
        # See https://stackoverflow.com/a/2828121/1944216
        events = events[events[:, 0].argsort()]
        event_id.update(event_id_annot)

    if events.size == 0:
        warn('No events found or provided. Please make sure to'
             ' set channel type using raw.set_channel_types'
             ' or provide events_data.')
        events = None

    return events, event_id


def _get_mrk_meas_date(mrk):
    """Find the measurement date from a KIT marker file."""
    info = get_kit_info(mrk, False)[0]
    meas_date = info.get('meas_date', None)
    if isinstance(meas_date, (tuple, list, np.ndarray)):
        meas_date = meas_date[0]
    if isinstance(meas_date, datetime):
        meas_datetime = meas_date
    elif meas_date is not None:
        meas_datetime = datetime.fromtimestamp(meas_date)
    else:
        meas_datetime = datetime.min
    return meas_datetime


def _infer_eeg_placement_scheme(raw):
    """Based on the channel names, try to infer an EEG placement scheme.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.

    Returns
    -------
    placement_scheme : str
        Description of the EEG placement scheme. Will be "n/a" for unsuccessful
        extraction.

    """
    placement_scheme = 'n/a'
    # Check if the raw data contains eeg data at all
    if 'eeg' not in raw:
        return placement_scheme

    # How many of the channels in raw are based on the extended 10/20 system
    raw.load_data()
    sel = pick_types(raw.info, meg=False, eeg=True)
    ch_names = [raw.ch_names[i] for i in sel]
    channel_names = [ch.lower() for ch in ch_names]
    montage1005 = make_standard_montage('standard_1005')
    montage1005_names = [ch.lower() for ch in montage1005.ch_names]

    if set(channel_names).issubset(set(montage1005_names)):
        placement_scheme = 'based on the extended 10/20 system'

    return placement_scheme


def _extract_landmarks(dig):
    """Extract NAS, LPA, and RPA from raw.info['dig']."""
    coords = dict()
    landmarks = {d['ident']: d for d in dig
                 if d['kind'] == FIFF.FIFFV_POINT_CARDINAL}
    if landmarks:
        if FIFF.FIFFV_POINT_NASION in landmarks:
            coords['NAS'] = landmarks[FIFF.FIFFV_POINT_NASION]['r'].tolist()
        if FIFF.FIFFV_POINT_LPA in landmarks:
            coords['LPA'] = landmarks[FIFF.FIFFV_POINT_LPA]['r'].tolist()
        if FIFF.FIFFV_POINT_RPA in landmarks:
            coords['RPA'] = landmarks[FIFF.FIFFV_POINT_RPA]['r'].tolist()
    return coords


def _update_sidecar(sidecar_fname, key, val):
    """Update a sidecar JSON file with a given key/value pair.

    Parameters
    ----------
    sidecar_fname : str | os.PathLike
        Full name of the data file
    key : str
        The key in the sidecar JSON file. E.g. "PowerLineFrequency"
    val : str
        The corresponding value to change to in the sidecar JSON file.
    """
    with open(sidecar_fname, "r") as fin:
        sidecar_json = json.load(fin)
    sidecar_json[key] = val
    with open(sidecar_fname, "w") as fout:
        json.dump(sidecar_json, fout)


def _scale_coord_to_meters(coord, unit):
    """Scale units to meters (mne-python default)."""
    if unit == 'cm':
        return np.divide(coord, 100.)
    elif unit == 'mm':
        return np.divide(coord, 1000.)
    else:
        return coord


def _check_empty_room_basename(bids_path, on_invalid_er_task='raise'):
    # only check task entity for emptyroom when it is the sidecar/MEG file
    if bids_path.suffix == 'meg':
        if bids_path.task != 'noise':
            msg = (f'task must be "noise" if subject is "emptyroom", but '
                   f'received: {bids_path.task}')
            if on_invalid_er_task == 'raise':
                raise ValueError(msg)
            elif on_invalid_er_task == 'warn':
                logger.critical(msg)
            else:
                pass


def _check_anonymize(anonymize, raw, ext):
    """Check the `anonymize` dict."""
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
                 f'is {daysback_min}')
        if ext == '.fif' and daysback > daysback_max:
            raise ValueError('`daysback` exceeds maximum value MNE '
                             'is able to store in FIF format, must '
                             f'be less than {daysback_max}')
    keep_his = anonymize['keep_his'] if 'keep_his' in anonymize else False
    return daysback, keep_his


def _get_anonymization_daysback(raw):
    """Get the min and max number of daysback necessary to satisfy BIDS specs.

    Parameters
    ----------
    raw : mne.io.Raw
        Subject raw data.

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
        Subject raw data or list of raw data from several subjects.

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
