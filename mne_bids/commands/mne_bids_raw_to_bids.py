"""Write raw files to BIDS format.

example usage:  $ mne_bids raw_to_bids --subject_id sub01 --task rest
--raw data.edf --bids_root new_path

"""
# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
import mne_bids
from mne_bids import write_raw_bids, BIDSPath
from mne_bids.read import _read_raw


def run():
    """Run the raw_to_bids command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__, usage="usage: %prog options args",
                           prog_prefix='mne_bids',
                           version=mne_bids.__version__)

    parser.add_option('--subject_id', dest='subject_id',
                      help=('subject name in BIDS compatible format '
                            '(01, 02, etc.)'))
    parser.add_option('--task', dest='task',
                      help='name of the task the data is based on')
    parser.add_option('--raw', dest='raw_fname',
                      help='path to the raw MEG file')
    parser.add_option('--bids_root', dest='bids_root',
                      help='The path of the BIDS compatible folder.')
    parser.add_option('--session_id', dest='session_id',
                      help='session name in BIDS compatible format')
    parser.add_option('--run', dest='run',
                      help='run number for this dataset')
    parser.add_option('--acq', dest='acq',
                      help='acquisition parameter for this dataset')
    parser.add_option('--events_data', dest='events_data',
                      help='events file (events.tsv)')
    parser.add_option('--event_id', dest='event_id',
                      help='event id dict', metavar='eid')
    parser.add_option('--hpi', dest='hpi',
                      help='path to the MEG marker points')
    parser.add_option('--electrode', dest='electrode',
                      help='path to head-native digitizer points')
    parser.add_option('--hsp', dest='hsp',
                      help='path to headshape points')
    parser.add_option('--config', dest='config',
                      help='path to the configuration file')
    parser.add_option('--overwrite', dest='overwrite',
                      help="whether to overwrite existing data (BOOLEAN)")
    parser.add_option('--line_freq', dest='line_freq',
                      help="The frequency of the line noise in Hz "
                           "(e.g. 50 or 60). If unknown, pass None")

    opt, args = parser.parse_args()

    if len(args) > 0:
        parser.print_help()
        parser.error('Do not specify arguments without flags. Found: "{}".\n'
                     .format(args))

    if not all([opt.subject_id, opt.task, opt.raw_fname, opt.bids_root]):
        parser.print_help()
        parser.error('Arguments missing. You need to specify at least the'
                     'following: --subject_id, --task, --raw, --bids_root.')

    bids_path = BIDSPath(
        subject=opt.subject_id, session=opt.session_id, run=opt.run,
        acquisition=opt.acq, task=opt.task, root=opt.bids_root)

    allow_maxshield = False
    if opt.raw_fname.endswith('.fif'):
        allow_maxshield = True

    raw = _read_raw(opt.raw_fname, hpi=opt.hpi, electrode=opt.electrode,
                    hsp=opt.hsp, config_path=opt.config,
                    allow_maxshield=allow_maxshield)
    if opt.line_freq is not None:
        line_freq = None if opt.line_freq == "None" else opt.line_freq
        raw.info['line_freq'] = line_freq
    write_raw_bids(raw, bids_path, event_id=opt.event_id,
                   events_data=opt.events_data, overwrite=opt.overwrite,
                   verbose=True)


if __name__ == '__main__':
    run()
