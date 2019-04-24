"""Initialize IO functions."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
from mne_bids.io.base import read_raw_bids, write_raw_bids

__all__ = ['read_raw_bids', 'write_raw_bids']
