"""Testing downloading and fetching of data."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op

# This is here to handle mne-python <0.20
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne  # noqa: F401

from mne_bids.datasets import fetch_faces_data


def test_fetch_faces_data():
    """Dry test fetch_faces_data (Will not download anything)."""
    data_path = fetch_faces_data(subject_ids=[])
    assert op.exists(data_path)
