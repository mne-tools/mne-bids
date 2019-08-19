"""Helper functions to fetch data to work with."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os
import os.path as op
import shutil
import tarfile
import urllib.request

from mne.utils import _fetch_file


def fetch_faces_data(data_path=None, repo='ds000117', subject_ids=[1]):
    """Dataset fetcher for OpenfMRI dataset ds000117.

    Parameters
    ----------
    data_path : str | None
        Path to the folder where data is stored. Defaults to
        '~/mne_data/mne_bids_examples'
    repo : str
        The folder name. Defaults to 'ds000117'.
    subject_ids : list of int
        The subjects to fetch. Defaults to [1], downloading subject 1.

    Returns
    -------
    data_path : str
        Path to the parent directory where the `repo` folder containing the
        data is stored.

    """
    if data_path is None:
        home = os.path.expanduser('~')
        data_path = os.path.join(home, 'mne_data', 'mne_bids_examples')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    target_dir = op.join(data_path, repo)
    os.makedirs(target_dir, exist_ok=True)
    for subject_id in subject_ids:  # pragma: no cover

        # If we have data for that subject already, skip
        sub_str = 'sub{:03}'.format(subject_id)
        if op.exists(op.join(target_dir, sub_str)):
            continue

        # Else, download it
        src_url = ('http://openfmri.s3.amazonaws.com/tarballs/'
                   'ds117_R0.1.1_{}_raw.tgz'.format(sub_str))
        tar_fname = op.join(data_path, repo, sub_str + '.tgz')
        if not op.exists(tar_fname):
            _fetch_file(url=src_url, file_name=tar_fname,
                        print_destination=True)

        # Unpack the downloaded archive to the correct location
        tf = tarfile.open(tar_fname)
        print('Extracting files. This may take a while ...')
        tf.extractall(path=data_path)  # will extract the 'ds117' directory
        shutil.move(op.join(data_path, 'ds117', sub_str), target_dir)

        # Clean up to get ready to download next subject
        shutil.rmtree(op.join(data_path, 'ds117'))
        os.remove(tar_fname)

    return data_path


def fetch_brainvision_testing_data(data_path=None):
    """Download the MNE-Python testing data for the BrainVision format.

    Parameters
    ----------
    data_path : str | None
        Path to the folder where data is stored. Defaults to
        '~/mne_data/mne_bids_examples/testing_data_BrainVision'

    Returns
    -------
    data_path : str
        Path to the folder containing the data.

    """
    if not data_path:
        home = os.path.expanduser('~')
        data_path = os.path.join(home, 'mne_data', 'mne_bids_examples',
                                 'testing_data_BrainVision')
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    base_url = 'https://github.com/mne-tools/mne-python/'
    base_url += 'raw/master/mne/io/brainvision/tests/data/test'
    file_endings = ['.vhdr', '.vmrk', '.eeg', ]
    for f_ending in file_endings:
        url = base_url + f_ending
        response = urllib.request.urlopen(url)

        fname = os.path.join(data_path, 'test' + f_ending)
        with open(fname, 'wb') as fout:
            fout.write(response.read())

    return data_path
