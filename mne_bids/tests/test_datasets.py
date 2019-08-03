"""Testing downloading and fetching of data."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os
import os.path as op
import time

import pytest

from mne.io import read_raw_brainvision

from mne_bids.datasets import (fetch_matchingpennies, fetch_faces_data,
                               fetch_brainvision_testing_data)


def test_fetch_matchingpennies(tmpdir):
    """Dry test fetch matchingpennies."""
    tmpdir = str(tmpdir)
    mp_path = op.join(tmpdir, 'eeg_matchingpennies')

    # We write some mock data so we don't download too much in the test
    for ff in [
        'CHANGES', 'README', 'participants.tsv', 'participants.json',
        'LICENSE', 'dataset_description.json', 'task-matchingpennies_eeg.json',
        'task-matchingpennies_events.json',
        'stimuli{}left_hand.png'.format(os.sep),
        'stimuli{}right_hand.png'.format(os.sep),
            ]:
        fpath = op.join(mp_path, ff)
        os.makedirs(op.dirname(fpath), exist_ok=True)
        open(fpath, 'w').close()

    # Overwrite is False, so it should only download ".bidsignore"
    assert not op.exists(op.join(mp_path, '.bidsignore'))
    fetch_matchingpennies(tmpdir, dataset_data=True, subjects=[])
    assert op.exists(op.join(mp_path, '.bidsignore'))

    # Test we raise an error due to wrong subject, no downloading
    with pytest.raises(ValueError, match='Specify `subjects` as a list'):
        fetch_matchingpennies(tmpdir, subjects=1)

    fetch_matchingpennies(tmpdir, sourcedata=True, dry=True)


def test_fetch_faces_data(tmpdir):
    """Dry test fetch_faces_data (Will not download anything)."""
    tmpdir = str(tmpdir)
    data_path = fetch_faces_data(tmpdir, subject_ids=[])
    assert op.exists(data_path)


def test_fetch_brainvision_testing_data():
    """Test downloading of BrainVision testing data (~500kB)."""
    start = time.time()
    data_path = fetch_brainvision_testing_data(overwrite=True)
    tdownload = time.time() - start
    raw = read_raw_brainvision(op.join(data_path, 'test.vhdr'))
    assert raw

    # This should return without downloading: The files are all there
    start = time.time()
    data_path = fetch_brainvision_testing_data(overwrite=False)
    assert time.time() - start < tdownload
