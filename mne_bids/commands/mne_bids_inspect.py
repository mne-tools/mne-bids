r"""Inspect MEG and EEG raw data, and interactively mark channels as bad.

example usage:
$ mne_bids inspect --subject_id=01 --task=experiment --session=test \
--datatype=meg --suffix=meg --bids_root=bids_root

"""
# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD-3-Clause

from mne.utils import logger

import mne_bids
from mne_bids import BIDSPath, inspect_dataset


def run():
    """Run the mark_channels command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__, usage="usage: %prog options args",
                           prog_prefix='mne_bids',
                           version=mne_bids.__version__)

    parser.add_option('--bids_root', dest='bids_root',
                      help='The path of the folder containing the BIDS '
                           'dataset')
    parser.add_option('--subject_id', dest='subject',
                      help=('Subject name'))
    parser.add_option('--session_id', dest='session',
                      help='Session name')
    parser.add_option('--task', dest='task',
                      help='Task name')
    parser.add_option('--acq', dest='acquisition',
                      help='Acquisition parameter')
    parser.add_option('--run', dest='run',
                      help='Run number')
    parser.add_option('--proc', dest='processing',
                      help='Processing label.')
    parser.add_option('--rec', dest='recording',
                      help='Recording name')
    parser.add_option('--type', dest='datatype',
                      help='Recording data type, e.g. meg, ieeg or eeg')
    parser.add_option('--suffix', dest='suffix',
                      help='The filename suffix, i.e. the last part before '
                           'the extension')
    parser.add_option('--ext', dest='extension',
                      help='The filename extension, including the leading '
                           'period, e.g. .fif')
    parser.add_option('--find_flat', dest='find_flat',
                      help='Whether to auto-detect flat channels and time '
                           'segments')
    parser.add_option('--l_freq', dest='l_freq',
                      help='The high-pass filter cutoff frequency')
    parser.add_option('--h_freq', dest='h_freq',
                      help='The low-pass filter cutoff frequency')
    parser.add_option('--verbose', dest='verbose', action='store_true',
                      help='Whether do generate additional diagnostic output')

    opt, args = parser.parse_args()
    if args:
        parser.print_help()
        parser.error(f'Please do not specify arguments without flags. '
                     f'Got: {args}.\n')

    if opt.bids_root is None:
        parser.print_help()
        parser.error('You must specify bids_root')

    bids_path = BIDSPath(subject=opt.subject, session=opt.session,
                         task=opt.task, acquisition=opt.acquisition,
                         run=opt.run, processing=opt.processing,
                         recording=opt.recording, datatype=opt.datatype,
                         suffix=opt.suffix, extension=opt.extension,
                         root=opt.bids_root)

    find_flat = True if opt.find_flat is None else bool(opt.find_flat)
    l_freq = None if opt.l_freq is None else float(opt.l_freq)
    h_freq = None if opt.h_freq is None else float(opt.h_freq)

    logger.info(f'Inspecting {bids_path.basename} …')
    inspect_dataset(bids_path=bids_path, find_flat=find_flat,
                    l_freq=l_freq, h_freq=h_freq,
                    verbose=opt.verbose)


if __name__ == '__main__':
    run()
