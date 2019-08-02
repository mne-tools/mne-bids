"""Testing downloading and fetching of data."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op

import pytest

from mne.io import read_raw_brainvision

from mne_bids.datasets import (fetch_matchingpennies, fetch_faces_data,
                               fetch_brainvision_testing_data)


def test_fetch_matchingpennies():
    """Dry test fetch matchingpennies."""
    with pytest.raises(ValueError, match=''):
        data_path = fetch_matchingpennies(subjects=1)

    # Write some mock data so we don't download too much in the test
    data_path = op.join(op.expanduser('~'), 'mne_data', 'mne_bids_examples')
    for ff in ['CHANGES', 'README', 'participants.tsv', 'participants.json',
               'LICENSE', 'dataset_description.json',
               'task-matchingpennies_eeg.json',
               'task-matchingpennies_events.json']:
        with open(op.join(data_path, 'eeg_matchingpennies', ff), 'w') as fout:
            fout.write('test file. Re-run fetch_matchingpennies with '
                       'overwrite=True')

    # Overwrite is False, so it should only download ".bidsignore"
    fetch_matchingpennies(data_path, download_dataset_data=True, subjects=[])
    assert op.exists(data_path)


def test_fetch_faces_data():
    """Dry test fetch_faces_data (Will not download anything)."""
    data_path = fetch_faces_data(subject_ids=[])
    assert op.exists(data_path)


def test_fetch_brainvision_testing_data():
    """Test downloading of BrainVision testing data (~500kB)."""
    data_path = fetch_brainvision_testing_data(overwrite=True)
    raw = read_raw_brainvision(op.join(data_path, 'test.vhdr'))
    assert raw
