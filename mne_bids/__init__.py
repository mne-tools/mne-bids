"""MNE software for making MEG BIDS compatible datasets easily."""

__version__ = '0.1.dev0'


from .meg_bids import raw_to_bids  # noqa
from .utils import make_bids_folders, make_bids_filename  # noqa
from .config import BIDS_VERSION  # noqa
