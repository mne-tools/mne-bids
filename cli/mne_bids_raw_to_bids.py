"""Write raw files to BIDS format.

example usage:  $ mne_bids raw_to_bids --subject_id sub01 --task rest
--raw data.edf --output_path new_path

"""
# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import mne_bids
from mne_bids import write_raw_bids, make_bids_basename
from mne_bids.read import _read_raw


def run():
    """Run the raw_to_bids command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__, usage="usage: %prog options args",
                           prog_prefix='mne_bids',
                           version=mne_bids.__version__)

    parser.add_option('--subject_id', dest='subject_id',
                      help=('The subject name in BIDS compatible format'
                            '(01,02, etc.)'))
    parser.add_option('--task', dest='task',
                      help='Name of the task the data is based on.')
    parser.add_option('--raw', dest='raw_fname',
                      help='The path to the raw MEG file.')
    parser.add_option('--output_path', dest='output_path',
                      help='The path of the BIDS compatible folder.')
    parser.add_option('--session_id', dest='session_id',
                      help='The session name in BIDS compatible format.')
    parser.add_option('--run', dest='run',
                      help='The run number for this dataset.')
    parser.add_option('--acq', dest='acq',
                      help='The acquisition parameter.')
    parser.add_option('--events_data', dest='events_data',
                      help='The events file.')
    parser.add_option('--event_id', dest='event_id',
                      help='The event id dict.', metavar='eid')
    parser.add_option('--hpi', dest='hpi',
                      help='The path to the MEG Marker points')
    parser.add_option('--electrode', dest='electrode',
                      help='The path to head-native Digitizer points')
    parser.add_option('--hsp', dest='hsp',
                      help='The path to headshape points.')
    parser.add_option('--config', dest='config',
                      help='The path to the configuration file')
    parser.add_option('--overwrite', dest='overwrite',
                      help="Boolean. Whether to overwrite existing data")
    parser.add_option('--allow_maxshield', dest='allow_maxshield',
                      help="Boolean. Whether to allow non Maxfiltered data.",
                      action='store_true')

    opt, args = parser.parse_args()

    if len(args) > 0:
        parser.print_help()
        parser.error('Do not specify arguments without flags. Found: "{}".\n'
                     .format(args))

    if not all([opt.subject_id, opt.task, opt.raw_fname, opt.output_path]):
        parser.print_help()
        parser.error('Arguments missing. You need to specify at least the'
                     'following: --subject_id, --task, --raw, --output_path.')

    bids_basename = make_bids_basename(
        subject=opt.subject_id, session=opt.session_id, run=opt.run,
        acquisition=opt.acq, task=opt.task)
    raw = _read_raw(opt.raw_fname, hpi=opt.hpi, electrode=opt.electrode,
                    hsp=opt.hsp, config=opt.config,
                    allow_maxshield=opt.allow_maxshield)
    write_raw_bids(raw, bids_basename, opt.output_path, event_id=opt.event_id,
                   events_data=opt.events_data, overwrite=opt.overwrite,
                   verbose=True)


if __name__ == '__main__':  # pragma: no cover
    run()
