"""Utility and helper functions for MNE-BIDS."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD-3-Clause
import json
import os
import re
from datetime import datetime, date, timedelta, timezone
from os import path as op

import numpy as np
from mne.channels import make_standard_montage
from mne.io.constants import FIFF
from mne.io.kit.kit import get_kit_info
from mne.io.pick import pick_types
from mne.utils import warn, logger, verbose

from mne_bids.tsv_handler import _to_tsv


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
                       resp='RESP', bio='MISC', dbs='DBS',
                       # MEG channels
                       meggradaxial='MEGGRADAXIAL', megmag='MEGMAG',
                       megrefgradaxial='MEGREFGRADAXIAL',
                       meggradplanar='MEGGRADPLANAR', megrefmag='MEGREFMAG',
                       ias='MEGOTHER', syst='MEGOTHER', exci='MEGOTHER')

    elif fro == 'bids' and to == 'mne':
        mapping = dict(EEG='eeg', MISC='misc', TRIG='stim', EMG='emg',
                       ECOG='ecog', SEEG='seeg', EOG='eog', ECG='ecg',
                       RESP='resp',
                       # No MEG channels for now
                       # Many to one mapping
                       VEOG='eog', HEOG='eog', DBS='dbs')
    else:
        raise ValueError('Only two types of mappings are currently supported: '
                         'from mne to bids, or from bids to mne. However, '
                         'you specified from "{}" to "{}"'.format(fro, to))

    return mapping


def _handle_datatype(raw, datatype):
    """Check if datatype exists in raw object or infer datatype if possible.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object.
    datatype : str | None
        Can be one of either ``'meg'``, ``'eeg'``, or ``'ieeg'``. If ``None``,
        `mne.utils._handle_datatype()` will attempt to infer the datatype from
        the ``raw`` object. In case of multiple data types in the ``raw``
        object, ``datatype`` must not be ``None``.

    Returns
    -------
    datatype : str
        One of either ``'meg'``, ``'eeg'``, or ``'ieeg'``.
    """
    if datatype is not None:
        _check_datatype(raw, datatype)
        # MEG data is not supported by BrainVision or EDF files
        if datatype in ['eeg', 'ieeg'] and 'meg' in raw:
            logger.info(f"{os.linesep}Both {datatype} and 'meg' data found. "
                        f"BrainVision and EDF do not support 'meg' data. "
                        f"The data will therefore be stored as 'meg' data. "
                        f"If you wish to store your {datatype} data in "
                        f"BrainVision or EDF, please remove the 'meg'"
                        f"channels from your recording.{os.linesep}")
            datatype = 'meg'
    else:
        datatypes = list()
        ieeg_types = ['seeg', 'ecog', 'dbs']
        if any(ieeg_type in raw for ieeg_type in ieeg_types):
            datatypes.append('ieeg')
        if 'meg' in raw:
            datatypes.append('meg')
        if 'eeg' in raw:
            datatypes.append('eeg')
        if len(datatypes) == 0:
            raise ValueError('No MEG, EEG or iEEG channels found in data. '
                             'Please use raw.set_channel_types to set the '
                             'channel types in the data.')
        elif len(datatypes) > 1:
            if 'meg' in datatypes and 'ieeg' not in datatypes:
                datatype = 'meg'
            elif 'ieeg' in datatypes and 'meg' not in datatypes:
                datatype = 'ieeg'
            else:
                raise ValueError(f'Multiple data types (``{datatypes}``) were '
                                 'found in the data. Please specify the '
                                 'datatype using '
                                 '`bids_path.update(datatype="<datatype>")` '
                                 'or use raw.set_channel_types to set the '
                                 'correct channel types in the raw object.')
        else:
            datatype = datatypes[0]
    return datatype


def _age_on_date(bday, exp_date):
    """Calculate age from birthday and experiment date.

    Parameters
    ----------
    bday : datetime.datetime
        The birthday of the participant.
    exp_date : datetime.datetime
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


def _write_json(fname, dictionary, overwrite=False):
    """Write JSON to a file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError(f'"{fname}" already exists. '
                              'Please set overwrite to True.')

    json_output = json.dumps(dictionary, indent=4)
    with open(fname, 'w', encoding='utf-8') as fid:
        fid.write(json_output)
        fid.write('\n')

    logger.info(f"Writing '{fname}'...")


@verbose
def _write_tsv(fname, dictionary, overwrite=False, verbose=None):
    """Write an ordered dictionary to a .tsv file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError(f'"{fname}" already exists. '
                              'Please set overwrite to True.')
    _to_tsv(dictionary, fname)

    logger.info(f"Writing '{fname}'...")


def _write_text(fname, text, overwrite=False):
    """Write text to a file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError(f'"{fname}" already exists. '
                              'Please set overwrite to True.')
    with open(fname, 'w', encoding='utf-8-sig') as fid:
        fid.write(text)
        fid.write('\n')

    logger.info(f"Writing '{fname}'...")


def _check_key_val(key, val):
    """Perform checks on a value to make sure it adheres to the spec."""
    if any(ii in val for ii in ['-', '_', '/']):
        raise ValueError("Unallowed `-`, `_`, or `/` found in key/value pair"
                         f" {key}: {val}")
    return key, val


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
    raw : mne.io.Raw
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
    keep_source = anonymize['keep_source'] if 'keep_source' in \
        anonymize else False
    return daysback, keep_his, keep_source


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


@verbose
def get_anonymization_daysback(raws, verbose=None):
    """Get the group min and max number of daysback necessary for BIDS specs.

    .. warning:: It is important that you remember the anonymization
                 number if you would ever like to de-anonymize but
                 that it is not included in the code publication
                 as that would break the anonymization.

    BIDS requires that anonymized dates be before 1925. In order to
    preserve the longitudinal structure and ensure anonymization, the
    user is asked to provide the same `daysback` argument to each call
    of `write_raw_bids`. To determine the minimum number of daysback
    necessary, this function will calculate the minimum number based on
    the most recent measurement date of raw objects.

    Parameters
    ----------
    raw : mne.io.Raw | list of mne.io.Raw
        Subject raw data or list of raw data from several subjects.
    %(verbose)s

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
        raise ValueError('All measurement dates are None, '
                         'pass any `daysback` value to anonymize.')
    daysback_min = max(daysback_min_list)
    daysback_max = min(daysback_max_list)
    if daysback_min > daysback_max:
        raise ValueError('The dataset spans more time than can be '
                         'accomodated by MNE, you may have to '
                         'not follow BIDS recommendations and use'
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


def _check_datatype(raw, datatype):
    """Check if datatype exists in given raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object.
    datatype : str
        Can be one of either ``'meg'``, ``'eeg'``, or ``'ieeg'``.

    Returns
    -------
    None
    """
    supported_types = ('meg', 'eeg', 'ieeg')
    if datatype not in supported_types:
        raise ValueError(
            f'The specified datatype {datatype} is currently not supported. '
            f'It should be one of  either `meg`, `eeg` or `ieeg` (Got '
            f'`{datatype}`. Please specify a valid datatype using '
            f'`bids_path.update(datatype="<datatype>")`.')
    datatype_matches = False
    if datatype == 'eeg' and datatype in raw:
        datatype_matches = True
    elif datatype == 'meg' and datatype in raw:
        datatype_matches = True
    elif datatype == 'ieeg':
        ieeg_types = ('seeg', 'ecog', 'dbs')
        if any(ieeg_type in raw for ieeg_type in ieeg_types):
            datatype_matches = True
    if not datatype_matches:
        raise ValueError(
            f'The specified datatype {datatype} was not found in the raw '
            'object. Please specify the correct datatype using '
            '`bids_path.update(datatype="<datatype>")` or use '
            'raw.set_channel_types to set the correct channel types in '
            'the raw object.')
