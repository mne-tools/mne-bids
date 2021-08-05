"""Write Elekta/Neuromag/MEGIN crosstalk data to BIDS.

example usage:
$ mne_bids crosstalk_to_bids --subject_id=01 --session=test
--bids_root=bids_root --file=ct_sparse.fif

"""
# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD-3-Clause

from mne.utils import logger

import mne_bids
from mne_bids import BIDSPath, write_meg_crosstalk


def run():
    """Run the crosstalk_to_bids command."""
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
    parser.add_option('--file', dest='fname',
                      help='The path of the crosstalk file')
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

    bids_path = BIDSPath(subject=opt.subject, session=opt.session,
                         root=opt.bids_root)

    logger.info(f'Writing crosstalk file {bids_path.basename} …')
    write_meg_crosstalk(fname=opt.fname, bids_path=bids_path,
                        verbose=opt.verbose)


if __name__ == '__main__':
    run()
