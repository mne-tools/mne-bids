"""Check whether a file format is supported by BIDS and then load it."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
from mne import io
import os

allowed_extensions_meg = ['.con', '.sqd', '.fif', '.pdf', '.ds']
allowed_extensions_eeg = ['.vhdr',  # BrainVision, accompanied by .vmrk, .eeg
                          '.edf',  # European Data Format
                          '.bdf',  # Biosemi
                          '.set',  # EEGLAB, potentially accompanied by .fdt
                          '.cnt',  # Neuroscan
                          ]

ALLOWED_EXTENSIONS = allowed_extensions_meg + allowed_extensions_eeg


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
              verbose=None):
    """Read a raw file into MNE, making inferences based on extension."""
    fname, ext = _parse_ext(raw_fname)

    # MEG File Types
    # --------------
    # KIT systems
    if ext in ['.con', '.sqd']:
        raw = io.read_raw_kit(raw_fname, elp=electrode, hsp=hsp,
                              mrk=hpi, preload=False)

    # Neuromag or converted-to-fif systems
    elif ext in ['.fif']:
        raw = io.read_raw_fif(raw_fname, preload=False)

    # BTi systems
    elif ext == '.pdf':
        if os.path.isfile(raw_fname):
            raw = io.read_raw_bti(raw_fname, config_fname=config,
                                  head_shape_fname=hsp,
                                  preload=False, verbose=verbose)

    # CTF systems
    elif ext == '.ds':
        raw = io.read_raw_ctf(raw_fname)

    # EEG File Types
    # --------------
    # BrainVision format by Brain Products, links to  a .eeg and a .vmrk file
    elif ext == '.vhdr':
        raw = io.read_raw_brainvision(raw_fname)

    # EDF (european data format) or BDF (biosemi) format
    elif ext == '.edf' or ext == '.bdf':
        raw = io.read_raw_edf(raw_fname)

    # EEGLAB .set format, a .fdt file is potentially linked from the .set
    elif ext == '.set':
        raw = io.read_raw_eeglab(raw_fname)

    # Neuroscan .cnt format
    elif ext == '.cnt':
        raw = io.read_raw_cnt(raw_fname)

    # No supported data found ...
    # ---------------------------
    else:
        raise ValueError("Raw file name extension must be one of %\n"
                         "Got %" % (ALLOWED_EXTENSIONS, ext))
    return raw
