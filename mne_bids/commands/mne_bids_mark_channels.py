"""Mark channels in an existing BIDS dataset as "bad".

example usage:
$ mne_bids mark_channels --ch_name="MEG 0112" --description="noisy" \
                         --ch_name="MEG 0131" --description="flat" \
                         --subject_id=01 --task=experiment --session=test \
                         --bids_root=bids_root

"""
# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD-3-Clause

from mne.utils import logger

import mne_bids
from mne_bids.config import reader
from mne_bids import BIDSPath, mark_channels


def run():
    """Run the mark_channels command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__, usage="usage: %prog options args",
                           prog_prefix='mne_bids',
                           version=mne_bids.__version__)

    parser.add_option('--ch_name', dest='ch_names', action='append',
                      default=[],
                      help='The names of the bad channels. If multiple '
                           'channels are bad, pass the --ch_name parameter '
                           'multiple times.')
    parser.add_option('--status',
                      default='bad',
                      help='Status of the channels (Either "good", or "bad").')
    parser.add_option('--description', dest='descriptions', action='append',
                      default=[],
                      help='Descriptions as to why the channels are bad. '
                           'Must match the number of bad channels provided. '
                           'Pass multiple times to supply more than one '
                           'value in that case.')
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
    if opt.ch_names is None:
        parser.print_help()
        parser.error('You must specify some --ch_name parameters.')

    status = opt.status
    ch_names = [] if opt.ch_names == [''] else opt.ch_names
    bids_path = BIDSPath(subject=opt.subject, session=opt.session,
                         task=opt.task, acquisition=opt.acquisition,
                         run=opt.run, processing=opt.processing,
                         recording=opt.recording, datatype=opt.datatype,
                         suffix=opt.suffix, extension=opt.extension,
                         root=opt.bids_root)

    bids_paths = bids_path.match()
    # Only keep data we can actually read & write.
    allowed_extensions = list(reader.keys())
    bids_paths = [p for p in bids_paths
                  if p.extension in allowed_extensions]

    if not bids_paths:
        logger.info('No matching files found. Please consider using a less '
                    'restrictive set of entities to broaden the search.')
        return  # XXX should be return with an error code?

    logger.info(f'Marking channels {", ".join(ch_names)} as bad in '
                f'{len(bids_paths)} recording(s) …')
    for bids_path in bids_paths:
        logger.info(f'Processing: {bids_path.basename}')
        mark_channels(bids_path=bids_path, ch_names=ch_names,
                      status=status, descriptions=opt.descriptions,
                      verbose=opt.verbose)


if __name__ == '__main__':
    run()
