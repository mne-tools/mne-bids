"""Count events in BIDS dataset.

example usage:  $ mne_bids count_events --bids_root bids_root_path

"""
# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause
from pathlib import Path

import mne_bids
from mne_bids.stats import count_events


def run():
    """Run the raw_to_bids command."""
    import pandas as pd
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__, usage="usage: %prog options args",
                           prog_prefix='mne_bids',
                           version=mne_bids.__version__)

    parser.add_option('--bids_root', dest='bids_root',
                      help='The path of the BIDS compatible folder.')

    parser.add_option('--datatype', dest='datatype', default='auto',
                      help='The datatype to consider.')

    parser.add_option('--describe', dest='describe', action="store_true",
                      help=('If set print the descriptive statistics '
                            '(min, max, etc.).'))

    parser.add_option('--output', dest='output', default=None,
                      help='Path to the CSV file where to store the results.')

    parser.add_option('--overwrite', dest='overwrite', action='store_true',
                      help='If set, overwrite an existing output file.')

    parser.add_option('--silent', dest='silent', action='store_true',
                      help='Whether to print the event counts on the screen.')

    opt, args = parser.parse_args()

    if len(args) > 0:
        parser.print_help()
        parser.error('Do not specify arguments without flags. Found: "{}".\n'
                     .format(args))

    if not all([opt.bids_root]):
        parser.print_help()
        parser.error('Arguments missing. You need to specify the '
                     '--bids_root parameter.')

    if opt.output and Path(opt.output).exists() and not opt.overwrite:
        parser.error('Output file exists. To overwrite, pass --overwrite')

    counts = count_events(opt.bids_root, datatype=opt.datatype)

    if opt.describe:
        counts = counts.describe()

    if not opt.silent:
        with pd.option_context(
            'display.max_rows', 1000,
            'display.max_columns', 80,
            'display.width', 1000
        ):
            print(counts)

    if opt.output:
        counts.to_csv(opt.output)
        print(f'\nOutput stored in {opt.output}')


if __name__ == '__main__':
    run()
