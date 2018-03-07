#!/usr/bin/env python
# Authors: Teon Brooks  <teon.brooks@gmail.com>

"""Command line interface for meg_bids.

example usage:  $ mne_bids raw_to_bids --subject sub01 --task rest --out out_path

"""

import sys
from mne_bids import raw_to_bids


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option('--subject', dest='subject_id',
                      help='The subject name in BIDS compatible format (01, 02, etc.)', metavar='sub')
    parser.add_option('--session', dest='session_id',
                      help='The session name in BIDS compatible format.', metavar='ses')
    parser.add_option('--run', dest='run',
                      help='The run number in BIDS compatible format.', metavar='r')
    parser.add_option('--task', dest='task',
                      help='The task name.', metavar='t')
    parser.add_option('--raw', dest='raw_fname',
                      help='The path to the raw MEG file.', metavar='raw')
    parser.add_option('--out', dest='out_path',
                      help='The path of the BIDS compatible folder.', metavar='o')
    parser.add_option('--events', dest='events_fname',
                      help='The path to the events file.', metavar='e')
    parser.add_option('--hpi', dest='hpi',
                      help='The path to the MEG Marker points', metavar='filename')
    parser.add_option('--electrode', dest='electrode',
                      help='The path to head-native Digitizer points', metavar='elec')
    parser.add_option('--hsp', dest='hsp',
                      help='The path to headshape points', metavar='hsp')
    parser.add_option('--config', dest='config',
                      help='The path to the configuration file', metavar='cfg')
    parser.add_option('--overwrite', dest='overwrite',
                      help='Boolean. If the file already exists, whether to overwrite it.',
                      metavar='ow')

    opt, args = parser.parse_args()

    raw = raw_to_bids(opt.subject_id, opt.session_id, opt.run, opt.task,
                      opt.raw_fname, opt.output_path, opt.events_fname,
                      opt.event_id, opt.hpi, opt.electrode, opt.hsp,
                      opt.config, opt.overwrite, verbose=True)

    sys.exit(0)

is_main = (__name__ == '__main__')
if is_main:
    run()
