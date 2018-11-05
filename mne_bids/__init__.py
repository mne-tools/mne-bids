"""MNE software for making MEG BIDS compatible datasets easily."""

__version__ = '0.2.dev0'


from .mne_bids import write_raw_bids  # noqa
from .utils import make_bids_folders, make_bids_basename  # noqa
from .config import BIDS_VERSION  # noqa
