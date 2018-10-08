"""Testing downloading and fetching of data."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op

from mne.io import read_raw_brainvision

from mne_bids.datasets import fetch_faces_data, fetch_brainvision_testing_data


def test_fetch_faces_data():
    """Dry test fetch_faces_data (Will not download anything)."""
    data_path = fetch_faces_data(subject_ids=[])
    assert op.exists(data_path)


def test_fetch_brainvision_testing_data():
    """Test downloading of BrainVision testing data (~500kB)."""
    data_path = fetch_brainvision_testing_data()
    raw = read_raw_brainvision(op.join(data_path, 'test.vhdr'))
    assert raw
