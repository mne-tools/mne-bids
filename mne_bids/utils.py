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
import warnings
import json
import shutil as sh

import numpy as np
from mne import read_events, find_events, events_from_annotations
from mne.utils import check_version
from mne.channels import read_montage
from mne.io.pick import pick_types

from .tsv_handler import _to_tsv, _tsv_to_str


def print_dir_tree(folder):
    """Recursively print a directory tree starting from `folder`."""
    if not op.exists(folder):
        raise ValueError('Directory does not exist: {}'.format(folder))

    baselen = len(folder.split(os.sep)) - 1  # makes tree always start at 0 len
    for root, dirs, files in os.walk(folder):
        branchlen = len(root.split(os.sep)) - baselen
        if branchlen <= 1:
            print('|%s' % (op.basename(root)))
        else:
            print('|%s %s' % ((branchlen - 1) * '---', op.basename(root)))  # noqa: E501
        for file in files:
            print('|%s %s' % (branchlen * '---', file))


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


def _parse_bids_filename(fname, verbose):
    """Get dict from BIDS fname."""
    keys = ['sub', 'ses', 'task', 'acq', 'run', 'proc', 'run', 'space',
            'recording', 'kind']
    params = {key: None for key in keys}
    entities = fname.split('_')
    idx_key = 0
    for entity in entities:
        assert '-' in entity
        key, value = entity.split('-')
        if key not in keys:
            raise KeyError('Unexpected entity ''%s'' found in filename ''%s'''
                           % (entity, fname))
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
        The event_id dict as provided in write_raw_bids, mapping a
        description key to an integer valued event code.
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
    elif ext in ['.vhdr', '.set'] and check_version('mne', '0.18.dev0'):
        events, event_id = events_from_annotations(raw)
    else:
        warnings.warn('No events found or provided. Please make sure to'
                      ' set channel type using raw.set_channel_types'
                      ' or provide events_data.')
        events = None
    return events, event_id


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
