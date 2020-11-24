r"""Inspect MEG and EEG raw data, and interactively mark channels as bad.

example usage:
$ mne_bids inspect --subject_id=01 --task=experiment --session=test \
--datatype=meg --suffix=meg --bids_root=bids_root

"""
# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD (3-clause)

from mne.utils import logger

import mne_bids
from mne_bids import BIDSPath, inspect


def run():
    """Run the mark_bad_channels command."""
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
    if opt.subject is None:
        parser.print_help()
        parser.error('You must specify a subject')
    if opt.task is None:
        parser.print_help()
        parser.error('You must specify a task')

    bids_path = BIDSPath(subject=opt.subject, session=opt.session,
                         task=opt.task, acquisition=opt.acquisition,
                         run=opt.run, processing=opt.processing,
                         recording=opt.recording, datatype=opt.datatype,
                         suffix=opt.suffix, extension=opt.extension,
                         root=opt.bids_root)

    logger.info(f'Inspecting {bids_path.basename} …')
    inspect(bids_path=bids_path, verbose=opt.verbose)


if __name__ == '__main__':  # pragma: no cover
    run()