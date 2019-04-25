"""Check whether a file format is supported by BIDS and then load it."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os
import os.path as op
import glob

import numpy as np
from mne import io

from .tsv_handler import _from_tsv, _drop
from .config import ALLOWED_EXTENSIONS

reader = {'.con': io.read_raw_kit, '.sqd': io.read_raw_kit,
          '.fif': io.read_raw_fif, '.pdf': io.read_raw_bti,
          '.ds': io.read_raw_ctf, '.vhdr': io.read_raw_brainvision,
          '.edf': io.read_raw_edf, '.bdf': io.read_raw_edf,
          '.set': io.read_raw_eeglab}


def _parse_ext(raw_fname, verbose=False):
    """Split a filename into its name and extension."""
    fname, ext = os.path.splitext(raw_fname)
    # BTi data is the only file format that does not have a file extension
    if ext == '' or 'c,rf' in fname:
        if verbose is True:
            print('Found no extension for raw file, assuming "BTi" format and '
                  'appending extension .pdf')
        ext = '.pdf'
    return fname, ext


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


def read_raw_bids(bids_fname, bids_root, return_events=True,
                  verbose=True):
    """Read BIDS compatible data.

    Parameters
    ----------
    bids_fname : str
        Full name of the data file
    bids_root : str
        Path to root of the BIDS folder
    return_events: bool
        Whether to return events or not. Default is True.
    verbose : bool
        The verbosity level

    Returns
    -------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    events : ndarray, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.
    event_id : dict
        Dictionary of events in the raw data mapping from the event value
        to an index.

    """
    from .utils import _parse_bids_filename

    # Full path to data file is needed so that mne-bids knows
    # what is the modality -- meg, eeg, ieeg to read
    bids_basename = '_'.join(bids_fname.split('_')[:-1])
    kind = bids_fname.split('_')[-1].split('.')[0]
    _, ext = _parse_ext(bids_fname)

    params = _parse_bids_filename(bids_basename, verbose)
    kind_dir = op.join(bids_root, 'sub-%s' % params['sub'],
                       'ses-%s' % params['ses'], kind)

    config = None
    if ext in ('.fif', '.ds', '.vhdr', '.edf', '.bdf', '.set'):
        bids_fname = op.join(kind_dir,
                             bids_basename + '_%s%s' % (kind, ext))
    elif ext == '.sqd':
        data_path = op.join(kind_dir, bids_basename + '_%s' % kind)
        bids_fname = op.join(data_path,
                             bids_basename + '_%s%s' % (kind, ext))
    elif ext == '.pdf':
        bids_raw_folder = op.join(kind_dir, bids_basename + '_%s' % kind)
        bids_fname = glob.glob(op.join(bids_raw_folder, 'c,rf*'))[0]
        config = op.join(bids_raw_folder, 'config')
    raw = _read_raw(bids_fname, electrode=None, hsp=None, hpi=None,
                    config=config, montage=None, verbose=None)

    events_fname = op.join(kind_dir, bids_basename + '_events.tsv')

    if not return_events:
        return raw

    if not op.exists(events_fname):
        raise ValueError('return_events=True but _events.tsv file'
                         ' not found')

    events, event_id = None, dict()
    dtypes = [float, float, str, int, int]
    events_dict = _from_tsv(events_fname, dtypes=dtypes)
    events_dict = _drop(events_dict, 'n/a', 'trial_type')

    if 'trial_type' not in events_dict:
        raise ValueError('trial_type column is missing. Cannot'
                         ' return events')

    for idx, ev in enumerate(np.unique(events_dict['trial_type'])):
        event_id[ev] = idx

    events = np.zeros((len(events_dict['onset']), 3), dtype=int)
    events[:, 0] = np.array(events_dict['onset']) * raw.info['sfreq'] + \
        raw.first_samp
    events[:, 2] = np.array([event_id[ev] for ev
                             in events_dict['trial_type']])

    return raw, events, event_id
