"""Rename files (making a copy) and update their internal pointers.

example usage: $ mne_bids cp --input myfile.vhdr --output sub-01_task-test.vhdr
"""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
import mne_bids
from mne_bids.copyfiles import (copyfile_brainvision, copyfile_eeglab,
                                copyfile_ctf)


def run():
    """Run the cp command."""
    from mne.commands.utils import get_optparser

    accepted_formats_msg = ('(accepted formats: BrainVision .vhdr, '
                            'EEGLAB .set, CTF .ds)')

    parser = get_optparser(__file__, usage="usage: %prog -i INPUT -o OUTPUT",
                           prog_prefix='mne_bids',
                           version=mne_bids.__version__)

    parser.add_option('-i', '--input', dest='input',
                      help=('path to the input file. {}'
                            .format(accepted_formats_msg)), metavar='INPUT')

    parser.add_option('-o', '--output', dest='output',
                      help=('path to the output file (MUST be same format '
                            'as input file)'), metavar='OUTPUT')

    parser.add_option('-v', '--verbose', dest="verbose",
                      help='set logging level to verbose', action="store_true")

    opt, args = parser.parse_args()
    opt_dict = vars(opt)

    # Check the usage and raise error if invalid
    if len(args) > 0:
        parser.print_help()
        parser.error('Do not specify arguments without flags. Found: "{}".\n'
                     'Did you forget to provide -i and -o?'
                     .format(args))

    if not opt_dict.get('input') or not opt_dict.get('output'):
        parser.print_help()
        parser.error('Incorrect number of arguments. Supply one input and one '
                     'output file. You supplied: "{}"'.format(opt))

    # Attempt to do the copying. Errors will be raised by the copyfile
    # functions if there are issues with the file formats
    if opt.input.endswith('.vhdr'):
        copyfile_brainvision(opt.input, opt.output, opt.verbose)
    elif opt.input.endswith('.set'):
        copyfile_eeglab(opt.input, opt.output)
    elif opt.input.endswith('.ds'):
        copyfile_ctf(opt.input, opt.output)
    else:
        parser.error('{} You supplied: "{}"'.format(accepted_formats_msg, opt))


if __name__ == '__main__':
    run()
