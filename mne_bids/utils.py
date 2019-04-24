"""Utility and helper functions for MNE-BIDS."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)
import os
import os.path as op


def print_dir_tree(folder):
    """Recursively print a directory tree starting from `folder`."""
    if not op.exists(folder):
        raise ValueError('Directory does not exist: {}'.format(folder))

    baselen = len(folder.split(os.sep)) - 1  # makes tree always start at 0 len
    for root, dirs, files in os.walk(folder):
        branchlen = len(root.split(os.sep)) - baselen
        if branchlen <= 1:
            print('|%s' % (op.basename(root)))
        else:
            print('|%s %s' % ((branchlen - 1) * '---', op.basename(root)))
        for file in files:
            print('|%s %s' % (branchlen * '---', file))
