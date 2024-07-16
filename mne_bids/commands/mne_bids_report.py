"""Summarize basic properties of the dataset and write it to stdout.

example usage:  $ mne_bids report --bids_root bids_root_path

"""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import mne_bids
from mne_bids import make_report


def run():
    """Run the make_report command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(
        __file__,
        usage="usage: %prog options args",
        prog_prefix="mne_bids",
        version=mne_bids.__version__,
    )

    parser.add_option(
        "--bids_root", dest="bids_root", help="The path of the BIDS compatible folder."
    )

    opt, args = parser.parse_args()

    if len(args) > 0:
        parser.print_help()
        parser.error(f'Do not specify arguments without flags. Found: "{args}".\n')

    if not all([opt.bids_root]):
        parser.print_help()
        parser.error(
            "Arguments missing. You need to specify the --bids_root parameter."
        )

    report = make_report(opt.bids_root)
    print("-" * 36 + " REPORT " + "-" * 36)
    print(report)
    assert "  " not in report


if __name__ == "__main__":
    run()
