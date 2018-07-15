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

from mne.utils import _fetch_file


def fetch_faces_data(data_path, repo, subject_ids):
    """Dataset fetcher for OpenfMRI dataset ds000117.

    Parameters
    ----------
    data_path : str
        Path to the folder where data is stored
    repo : str
        The folder name (typically 'ds000117')
    subject_ids : list of int
        The subjects to fetch

    """
    for subject_id in subject_ids:
        src_url = ('http://openfmri.s3.amazonaws.com/tarballs/'
                   'ds117_R0.1.1_sub%03d_raw.tgz' % subject_id)
        tar_fname = op.join(data_path, repo + '.tgz')
        target_dir = op.join(data_path, repo)
        if not op.exists(target_dir):
            if not op.exists(tar_fname):
                _fetch_file(url=src_url, file_name=tar_fname,
                            print_destination=True, resume=True, timeout=10.)
            tf = tarfile.open(tar_fname)
            print('Extracting files. This may take a while ...')
            tf.extractall(path=data_path)
            shutil.move(op.join(data_path, 'ds117'), target_dir)
            os.remove(tar_fname)


def download_matchingpennies_subj(url_dict=None, directory='.'):
    """Download and save subject data from the matching pennies dataset[1].

    Parameters
    ----------
    url_dict : dict | None
        Dictionary with keys corresponding to the type of BIDS file
        and its extension. Dict values corresponding to valid urls
        pointing to the respective files on the Open Science
        Framework. MUST contain a key value pair
        "subj_id: int" (see Examples). If None, will download data for sub-05.
    subj_id : int | None
        The subject identifier the url_dict corresponds to. If None,
        will be set to 99.
    directory : str
        Path to the directory where to save the data. Defaults
        to the working directory.

    References
    ----------
    .. [1] Appelhoff, S., Sauer, D., & Gill, S. S. (2018, July 15).
       Matching Pennies: A Brain Computer Interface Implementation.
       Retrieved from osf.io/cj2dr

    Examples
    --------
    >>> # Links manually extracted for sub-05 from osf.io/cj2dr
    >>> url_dict = {'subj_id': 5,
                    'channels.tsv': 'https://osf.io/wzyh2/',
                    'eeg.eeg': 'https://osf.io/mj7q8/',
                    'eeg.vhdr': 'https://osf.io/j2dgs/',
                    'eeg.vmrk': 'https://osf.io/w8rhu/',
                    'events.tsv': 'https://osf.io/kzvwn/'}
    >>> download_matchingpennies_subj(url_dict, directory='.')
    Downloading ... this may take a while.
    >>> os.path.exists('sub-05_task-matchingpennies_channels.tsv')
    True

    """
    # Check input
    if url_dict is None:
        url_dict = {'subj_id': 5,
                    'channels.tsv': 'https://osf.io/wzyh2/',
                    'eeg.eeg': 'https://osf.io/mj7q8/',
                    'eeg.vhdr': 'https://osf.io/j2dgs/',
                    'eeg.vmrk': 'https://osf.io/w8rhu/',
                    'events.tsv': 'https://osf.io/kzvwn/'}

    # Get the present subject
    subj_id = url_dict.pop('subj_id', 'n/a')
    if subj_id not in [5, 6, 7, 8, 9, 10, 11]:
        raise ValueError('Specify a valid subj_id in url_dict.')

    # Download all files for the present subject
    print('Downloading ... this may take a while.\n')
    for f, url in url_dict.items():
        url += 'download'
        fname = 'sub-{:02}_task-matchingpennies_{}'.format(subj_id, f)
        fpath = os.path.join(directory, fname)
        if not op.exists(fpath):
            _fetch_file(url=url, file_name=fpath,
                        print_destination=True, resume=True, timeout=10.)
