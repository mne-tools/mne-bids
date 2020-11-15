"""Count events in BIDS dataset.

example usage:  $ mne_bids count_events --bids_root bids_root_path

"""
# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)
import mne_bids
from mne_bids.stats import count_events


def run():
    """Run the raw_to_bids command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__, usage="usage: %prog options args",
                           prog_prefix='mne_bids',
                           version=mne_bids.__version__)

    parser.add_option('--bids_root', dest='bids_root',
                      help='The path of the BIDS compatible folder.')

    opt, args = parser.parse_args()

    if len(args) > 0:
        parser.print_help()
        parser.error('Do not specify arguments without flags. Found: "{}".\n'
                     .format(args))

    if not all([opt.bids_root]):
        parser.print_help()
        parser.error('Arguments missing. You need to specify the '
                     '--bids_root parameter.')

    counts = count_events(opt.bids_root)
    print(counts)


if __name__ == '__main__':  # pragma: no cover
    run()
