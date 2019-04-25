"""MNE software for easily interacting with BIDS compatible datasets."""

__version__ = '0.2.dev0'


from .mne_bids import write_raw_bids  # noqa
from .read import read_raw_bids  # noqa
from .utils import make_bids_folders, make_bids_basename  # noqa
from .config import BIDS_VERSION  # noqa
