"""Utility and helper functions for MNE-BIDS."""
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
import glob
import warnings
import json
import shutil as sh
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
from mne import read_events, find_events, events_from_annotations
from mne.utils import check_version
from mne.channels import read_montage
from mne.io.pick import pick_types
from mne.io.kit.kit import get_kit_info
from mne.io.constants import FIFF

from mne_bids.tsv_handler import _to_tsv, _tsv_to_str


def get_kinds(bids_root):
    """Get list of data types ("kinds") present in a BIDS dataset.

    Parameters
    ----------
    bids_root : str
        Path to the root of the BIDS directory.

    Returns
    -------
    kinds : list of str
        List of the data types present in the BIDS dataset pointed to by
        `bids_root`.

    """
    # Take all possible kinds from "entity" table (Appendix in BIDS spec)
    kind_list = ('anat', 'func', 'dwi', 'fmap', 'beh', 'meg', 'eeg', 'ieeg')
    kinds = list()
    for root, dirs, files in os.walk(bids_root):
        for dir in dirs:
            if dir in kind_list and dir not in kinds:
                kinds.append(dir)

    return kinds


def get_entity_vals(bids_root, entity_key):
    """Get list of values associated with an `entity_key` in a BIDS dataset.

    BIDS file names are organized by key-value pairs called "entities" [1]_.
    With this function, you can get all values for an entity indexed by its
    key.

    Parameters
    ----------
    bids_root : str
        Path to the root of the BIDS directory.
    entity_key : str
        The name of the entity key to search for. Can be one of
        ['sub', 'ses', 'run', 'acq'].

    Returns
    -------
    entity_vals : list of str
        List of the values associated with an `entity_key` in the BIDS dataset
        pointed to by `bids_root`.

    Examples
    --------
    >>> bids_root = '~/bids_datasets/eeg_matchingpennies'
    >>> entity_key = 'sub'
    >>> get_entity_vals(bids_root, entity_key)
    ['05', '06', '07', '08', '09', '10', '11']


    References
    ----------
    .. [1] https://bids-specification.rtfd.io/en/latest/02-common-principles.html#file-name-structure  # noqa: E501

    """
    entities = ('sub', 'ses', 'task', 'run', 'acq')
    if entity_key not in entities:
        raise ValueError('`key` must be one of "{}". Got "{}"'
                         .format(entities, entity_key))

    p = re.compile(r'{}-(.*?)_'.format(entity_key))
    value_list = list()
    for filename in Path(bids_root).rglob('*{}-*_*'.format(entity_key)):
        match = p.search(filename.stem)
        value = match.group(1)
        if value not in value_list:
            value_list.append(value)
    return sorted(value_list)


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
    ch_type_mapping : collections.defaultdict
        Dictionary mapping from one nomenclature of channel types to another.
        If a key is not present, a default value will be returned that depends
        on the `fro` and `to` parameters.

    Notes
    -----
    For the mapping from BIDS to MNE, MEG channel types are ignored for now.
    Furthermore, this is not a one-to-one mapping: Incomplete and partially
    one-to-many/many-to-one.

    """
    if fro == 'mne' and to == 'bids':
        map_chs = dict(eeg='EEG', misc='MISC', stim='TRIG', emg='EMG',
                       ecog='ECOG', seeg='SEEG', eog='EOG', ecg='ECG',
                       # MEG channels
                       meggradaxial='MEGGRADAXIAL', megmag='MEGMAG',
                       megrefgradaxial='MEGREFGRADAXIAL',
                       meggradplanar='MEGGRADPLANAR', megrefmag='MEGREFMAG',
                       )
        default_value = 'OTHER'

    elif fro == 'bids' and to == 'mne':
        map_chs = dict(EEG='eeg', MISC='misc', TRIG='stim', EMG='emg',
                       ECOG='ecog', SEEG='seeg', EOG='eog', ECG='ecg',
                       # No MEG channels for now
                       # Many to one mapping
                       VEOG='eog', HEOG='eog',
                       )
        default_value = 'misc'

    else:
        raise ValueError('Only two types of mappings are currently supported: '
                         'from mne to bids, or from bids to mne. However, '
                         'you specified from "{}" to "{}"'.format(fro, to))

    # Make it a defaultdict to prevent key errors
    ch_type_mapping = defaultdict(lambda: default_value)
    ch_type_mapping.update(map_chs)

    return ch_type_mapping


def print_dir_tree(folder, max_depth=None):
    """Recursively print dir tree starting from `folder` up to `max_depth`."""
    if not op.exists(folder):
        raise ValueError('Directory does not exist: {}'.format(folder))
    msg = '`max_depth` must be a positive integer or None'
    if not isinstance(max_depth, (int, type(None))):
        raise ValueError(msg)
    if max_depth is None:
        max_depth = float('inf')
    if max_depth < 0:
        raise ValueError(msg)

    # Use max_depth same as the -L param in the unix `tree` command
    max_depth += 1

    # Base length of a tree branch, to normalize each tree's start to 0
    baselen = len(folder.split(os.sep)) - 1

    # Recursively walk through all directories
    for root, dirs, files in os.walk(folder):

        # Check how far we have walked
        branchlen = len(root.split(os.sep)) - baselen

        # Only print, if this is up to the depth we asked
        if branchlen <= max_depth:
            if branchlen <= 1:
                print('|{}'.format(op.basename(root) + os.sep))
            else:
                print('|{} {}'.format((branchlen - 1) * '---',
                                      op.basename(root) + os.sep))

            # Only print files if we are NOT yet up to max_depth or beyond
            if branchlen < max_depth:
                for file in files:
                    print('|{} {}'.format(branchlen * '---', file))


def _mkdir_p(path, overwrite=False, verbose=False):
    """Create a directory, making parent directories as needed [1].

    References
    ----------
    .. [1] stackoverflow.com/questions/600268/mkdir-p-functionality-in-python

    """
    if overwrite and op.isdir(path):
        sh.rmtree(path)
        if verbose is True:
            print('Clearing path: %s' % path)

    os.makedirs(path, exist_ok=True)
    if verbose is True:
        print('Creating folder: %s' % path)


def _parse_ext(raw_fname, verbose=False):
    """Split a filename into its name and extension."""
    fname, ext = os.path.splitext(raw_fname)
    # BTi data is the only file format that does not have a file extension
    if ext == '' or 'c,rf' in fname:
        if verbose is True:
            print('Found no extension for raw file, assuming "BTi" format and '
                  'appending extension .pdf')
        ext = '.pdf'
    # If ending on .gz, check whether it is an .nii.gz file
    elif ext == '.gz' and raw_fname.endswith('.nii.gz'):
        ext = '.nii.gz'
        fname = fname[:-4]  # cut off the .nii
    return fname, ext


# This regex matches key-val pairs. Any characters are allowed in the key and
# the value, except these special symbols: - _ . \ /
param_regex = re.compile(r'([^-_\.\\\/]+)-([^-_\.\\\/]+)')


def _parse_bids_filename(fname, verbose):
    """Get dict from BIDS fname."""
    keys = ['sub', 'ses', 'task', 'acq', 'run', 'proc', 'run', 'space',
            'recording', 'kind']
    params = {key: None for key in keys}
    idx_key = 0
    for match in re.finditer(param_regex, op.basename(fname)):
        key, value = match.groups()
        if key not in keys:
            raise KeyError('Unexpected entity ''%s'' found in filename ''%s'''
                           % (key, fname))
        if keys.index(key) < idx_key:
            raise ValueError('Entities in filename not ordered correctly.'
                             ' "%s" should have occured earlier in the '
                             'filename "%s"' % (key, fname))
        idx_key = keys.index(key)
        params[key] = value
    return params


def _handle_kind(raw):
    """Get kind."""
    if 'eeg' in raw and ('ecog' in raw or 'seeg' in raw):
        raise ValueError('Both EEG and iEEG channels found in data.'
                         'There is currently no specification on how'
                         'to handle this data. Please proceed manually.')
    elif 'meg' in raw:
        kind = 'meg'
    elif 'ecog' in raw or 'seeg' in raw:
        kind = 'ieeg'
    elif 'eeg' in raw:
        kind = 'eeg'
    else:
        raise ValueError('Neither MEG/EEG/iEEG channels found in data.'
                         'Please use raw.set_channel_types to set the '
                         'channel types in the data.')
    return kind


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
            raise ValueError("All values must be either None or strings. "
                             "Found type %s." % type(var))


def _write_json(fname, dictionary, overwrite=False, verbose=False):
    """Write JSON to a file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError('"%s" already exists. Please set '  # noqa: F821
                              'overwrite to True.' % fname)

    json_output = json.dumps(dictionary, indent=4)
    with open(fname, 'w') as fid:
        fid.write(json_output)
        fid.write('\n')

    if verbose is True:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(json_output)


def _write_tsv(fname, dictionary, overwrite=False, verbose=False):
    """Write an ordered dictionary to a .tsv file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError('"%s" already exists. Please set '  # noqa: F821
                              'overwrite to True.' % fname)
    _to_tsv(dictionary, fname)

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(_tsv_to_str(dictionary))


def _check_key_val(key, val):
    """Perform checks on a value to make sure it adheres to the spec."""
    if any(ii in val for ii in ['-', '_', '/']):
        raise ValueError("Unallowed `-`, `_`, or `/` found in key/value pair"
                         " %s: %s" % (key, val))
    return key, val


def _read_events(events_data, event_id, raw, ext):
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
    ext : str
        The extension of the original data file.

    Returns
    -------
    events : array, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.

    """
    if isinstance(events_data, str):
        events = read_events(events_data).astype(int)
    elif isinstance(events_data, np.ndarray):
        if events_data.ndim != 2:
            raise ValueError('Events must have two dimensions, '
                             'found %s' % events_data.ndim)
        if events_data.shape[1] != 3:
            raise ValueError('Events must have second dimension of length 3, '
                             'found %s' % events_data.shape[1])
        events = events_data
    elif 'stim' in raw:
        events = find_events(raw, min_duration=0.001, initial_event=True)
    elif ext in ['.vhdr', '.set'] and check_version('mne', '0.18'):
        events, event_id = events_from_annotations(raw, event_id)
    else:
        warnings.warn('No events found or provided. Please make sure to'
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
    if meas_date is not None:
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
    montage1005 = read_montage(kind='standard_1005')
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


def _find_best_candidates(params, candidate_list):
    """Return the best candidate, based on the number of param matches.

    Assign each candidate a score, based on how many entities are shared with
    the ones supplied in the `params` parameter. The candidate with the highest
    score wins. Candidates with entities that conflict with the supplied
    `params` are disqualified.

    Parameters
    ----------
    params : dict
        The entities that the candidate should match.
    candidate_list : list of str
        A list of candidate filenames.

    Returns
    -------
    best_candidates : list of str
        A list of all the candidate filenames that are tied for first place.
        Hopefully, the list will have a length of one.
    """
    params = {key: value for key, value in params.items() if value is not None}

    best_candidates = []
    best_n_matches = 0
    for candidate in candidate_list:
        n_matches = 0
        candidate_disqualified = False
        candidate_params = _parse_bids_filename(candidate, verbose=False)
        for entity, value in params.items():
            if entity in candidate_params:
                if candidate_params[entity] is None:
                    continue
                elif candidate_params[entity] == value:
                    n_matches += 1
                else:
                    # Incompatible entity found, candidate is disqualified
                    candidate_disqualified = True
                    break
        if not candidate_disqualified:
            if n_matches > best_n_matches:
                best_n_matches = n_matches
                best_candidates = [candidate]
            elif n_matches == best_n_matches:
                best_candidates.append(candidate)

    return best_candidates


def _find_matching_sidecar(bids_fname, bids_root, suffix, allow_fail=False):
    """Try to find a sidecar file with a given suffix for a data file.

    Parameters
    ----------
    bids_fname : str
        Full name of the data file
    bids_root : str
        Path to root of the BIDS folder
    suffix : str
        The suffix of the sidecar file to be found. E.g., "_coordsystem.json"
    allow_fail : bool
        If False, will raise RuntimeError if not exactly one matching sidecar
        was found. If True, will return None in that case. Defaults to False

    Returns
    -------
    sidecar_fname : str | None
        Path to the identified sidecar file, or None, if `allow_fail` is True
        and no sidecar_fname was found

    """
    params = _parse_bids_filename(bids_fname, verbose=False)

    # We only use subject and session as identifier, because all other
    # parameters are potentially not binding for metadata sidecar files
    search_str = 'sub-' + params['sub']
    if params['ses'] is not None:
        search_str += '_ses-' + params['ses']

    # Find all potential sidecar files, doing a recursive glob from bids_root
    search_str = op.join(bids_root, '**', search_str + '*' + suffix)
    candidate_list = glob.glob(search_str, recursive=True)
    best_candidates = _find_best_candidates(params, candidate_list)

    if len(best_candidates) == 1:
        # Success
        return best_candidates[0]

    # We failed. Construct a helpful error message.
    # If this was expected, simply return None, otherwise, raise an exception.
    msg = None
    if len(best_candidates) == 0:
        msg = ('Did not find any {} file associated with {}.'
               .format(suffix, bids_fname))
    elif len(best_candidates) > 1:
        # More than one candidates were tied for best match
        msg = ('Expected to find a single {} file associated with '
               '{}, but found {}: "{}".'
               .format(suffix, bids_fname, len(candidate_list),
                       candidate_list))
    msg += '\n\nThe search_str was "{}"'.format(search_str)
    if allow_fail:
        warnings.warn(msg)
        return None
    else:
        raise RuntimeError(msg)
