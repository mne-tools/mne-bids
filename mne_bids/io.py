from mne import io
import os


def _parse_ext(raw_fname):
    fname, ext = os.path.splitext(raw_fname)
    # BTi data is the only file format that does have a file extension
    if ext == '':
        ext = '.pdf'
    return fname, ext


def _read_raw(raw_fname, electrode=None, hsp=None, hpi=None, config=None,
              verbose=None):
    """Read a raw file into MNE, making inferences based on extension."""
    fname, ext = _parse_ext(raw_fname)

    # KIT systems
    if ext in ['.con', '.sqd']:
        raw = io.read_raw_kit(raw_fname, elp=electrode, hsp=hsp,
                              mrk=hpi, preload=False)

    # Neuromag or converted-to-fif systems
    elif ext in ['.fif', '.gz']:
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
    return raw
