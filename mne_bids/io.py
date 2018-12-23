"""Check whether a file format is supported by BIDS and then load it."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os
import glob
import pandas as pd
from mne import io

allowed_extensions_meg = ['.con', '.sqd', '.fif', '.pdf', '.ds']
allowed_extensions_eeg = ['.vhdr',  # BrainVision, accompanied by .vmrk, .eeg
                          '.edf',  # European Data Format
                          '.bdf',  # Biosemi
                          '.set',  # EEGLAB, potentially accompanied by .fdt
                          ]

allowed_extensions_ieeg = ['.vhdr',  # BrainVision, accompanied by .vmrk, .eeg
                           '.edf',  # European Data Format
                           '.set',  # EEGLAB, potentially accompanied by .fdt
                           '.mef',  # MEF: Multiscale Electrophysiology File
                           '.nwb',  # Neurodata without borders
                           ]

ALLOWED_EXTENSIONS = (allowed_extensions_meg +
                      allowed_extensions_eeg +
                      allowed_extensions_ieeg)

reader = {'.con': io.read_raw_kit, '.sqd': io.read_raw_kit,
          '.fif': io.read_raw_fif, '.pdf': io.read_raw_bti,
          '.ds': io.read_raw_ctf, '.vhdr': io.read_raw_brainvision,
          '.edf': io.read_raw_edf, '.bdf': io.read_raw_edf,
          '.set': io.read_raw_eeglab}


def _parse_ext(raw_fname, verbose=False):
    """Split a filename into its name and extension."""
    fname, ext = os.path.splitext(raw_fname)
    # BTi data is the only file format that does not have a file extension
    if ext == '':
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
    elif ext == '.pdf' and os.path.isfile(raw_fname):
        raw = io.read_raw_bti(raw_fname, config_fname=config,
                              head_shape_fname=hsp,
                              preload=False, verbose=verbose)

    elif ext in ['.fif', '.ds', '.vhdr', '.set']:
        raw = reader[ext](raw_fname, allow_maxshield=allow_maxshield)

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


def read_raw_bids(bids_fname, output_path, return_events=True,
                  verbose=True):
    """Read BIDS compatible data."""
    import os.path as op
    import numpy as np
    from .utils import _parse_bids_filename

    bids_basename = '_'.join(bids_fname.split('_')[:-1])
    kind = bids_fname.split('_')[-1].split('.')[0]
    _, ext = _parse_ext(bids_fname)

    params = _parse_bids_filename(bids_basename, verbose)
    meg_dir = op.join(output_path, 'sub-%s' % params['sub'],
                      'ses-%s' % params['ses'], kind)

    events_fname = op.join(meg_dir, bids_basename + '_events.tsv')

    # channels_fname = fname.split('.') + '_channels.tsv'
    events_df = pd.read_csv(events_fname, delimiter='\t')
    events_df = events_df.dropna()

    event_id = dict()
    for idx, ev in enumerate(np.unique(events_df['trial_type'])):
        event_id[ev] = idx

    # electrode = blah
    # hsp = blah
    # hpi = blah
    # config = blah

    # create montage here
    # blah

    if ext == '.fif':
        bids_fname = op.join(meg_dir,
                             bids_basename + '_%s%s' % (kind, ext))
    raw = _read_raw(bids_fname, electrode=None, hsp=None, hpi=None,
                    config=None, montage=None, verbose=None)

    events = np.zeros((events_df.shape[0], 3), dtype=int)
    events[:, 0] = events_df['onset'] * raw.info['sfreq'] + raw.first_samp
    events[:, 2] = np.array([event_id[ev] for ev in events_df['trial_type']])

    if return_events:
        return raw, events, event_id
    return raw
