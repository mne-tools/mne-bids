"""Mark channels in an existing BIDS dataset as "bad".

example usage:
$ mne_bids mark_bad_channels --ch_name="MEG 0112" --description="noisy" \
                             --ch_name="MEG 0131" --description="flat" \
                             --subject_id=01 --task=experiment --session=test \
                             --bids_root=bids_root

"""
# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD (3-clause)

import mne_bids
from mne_bids import make_bids_basename, mark_bad_channels


def run():
    """Run the mark_bad_channels command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__, usage="usage: %prog options args",
                           prog_prefix='mne_bids',
                           version=mne_bids.__version__)

    parser.add_option('--ch_name', dest='ch_names', action='append',
                      help='The names of the channels, separated by commas')
    parser.add_option('--description', dest='descriptions', action='append',
                      help='Descriptions as to why the channels are bad')
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
    parser.add_option('--kind', dest='kind',
                      help='Recording kind, e.g. meg or eeg')
    parser.add_option('--overwrite', dest='overwrite',
                      help='Retain existing channel status entries or not')

    opt, args = parser.parse_args()
    opt = opt.__dict__  # XXX Is there a cleaner way?
    ch_names = opt.pop('ch_names')
    descriptions = opt.pop('descriptions')
    bids_root = opt.pop('bids_root')
    kind = opt.pop('kind')
    overwrite = opt.pop('overwrite')

    if args:
        parser.print_help()
        parser.error(f'Please do not specify arguments without flags. '
                     f'Got: {args}.\n')

    if bids_root is None:
        parser.print_help()
        parser.error('You must specify bids_root')
    if ch_names is None:
        parser.print_help()
        parser.error('You must specify ch_names')
    if opt['subject'] is None:
        parser.print_help()
        parser.error('You must specify subject_id')

    bids_basename = make_bids_basename(**opt)
    mark_bad_channels(ch_names=ch_names, descriptions=descriptions,
                      bids_basename=bids_basename, bids_root=bids_root,
                      kind=kind, overwrite=overwrite)


if __name__ == '__main__':  # pragma: no cover
    run()
