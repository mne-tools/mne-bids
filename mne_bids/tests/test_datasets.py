"""Test the downloading of datasets."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os

from nose.tools import assert_raises

from mne_bids import datasets


def test_download_matchingpennies_subj():
    """Test downloading subject data from OSF to the working dir."""
    # Make a temporary directory for testing
    tmp_dir = '.' + os.sep + 'tmp_dir_matchingpennies'
    try:
        assert not os.path.exists(tmp_dir)
    except AssertionError:
        os.rmdir(tmp_dir)
    os.mkdir(tmp_dir)

    # Download a single file
    url_dict = {'subj_id': 5,
                'channels.tsv': 'https://osf.io/wzyh2/'}
    datasets.download_matchingpennies_subj(url_dict, tmp_dir)

    # Assert that the file is there
    fpath = os.path.join('.', tmp_dir,
                         'sub-05_task-matchingpennies_channels.tsv')
    assert os.path.exists(fpath)

    # Assert it is the expected file
    with open(fpath, 'r') as f:
        line = f.readline()
        assert 'status' in line.split('\t')
        line = f.readline()
        assert 'FC5' in line.split('\t')

    # Assert we get an error for invalid subj_id
    assert_raises(ValueError,
                  datasets.download_matchingpennies_subj,
                  {'channels.tsv': 'https://osf.io/wzyh2/'})

    # It worked, now delete the file and the tmp_dir again
    os.remove(fpath)
    os.rmdir(tmp_dir)
    assert not os.path.exists(tmp_dir)
