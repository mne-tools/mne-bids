"""Test the downloading of datasets."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os

import pytest

from mne.utils import _TempDir

from mne_bids.datasets import download_matchingpennies_subj


def test_download_matchingpennies_subj():
    """Test downloading subject data from OSF to the working dir."""
    # Make a temporary directory for testing
    tmp_dir = _TempDir()

    # Download a single file
    url_dict = {'subj_id': 5,
                'channels.tsv': 'https://osf.io/dq5v2/'}
    download_matchingpennies_subj(url_dict, tmp_dir)

    # Assert that the file is there
    fpath = os.path.join(tmp_dir, 'sub-05_task-matchingpennies_channels.tsv')
    assert os.path.exists(fpath)

    # Assert it is the expected file
    with open(fpath, 'r') as f:
        line = f.readline()
        assert 'status' in line.split('\t')
        line = f.readline()
        assert 'FC5' in line.split('\t')

    # Assert we get an error for invalid subj_id
    with pytest.raises(ValueError):
        d = {'channels.tsv': 'https://osf.io/wzyh2/'}
        download_matchingpennies_subj(d)
