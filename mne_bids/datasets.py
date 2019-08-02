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
import urllib

from mne.utils import _fetch_file

from mne_bids import make_bids_basename, make_bids_folders


def _download_data(data, overwrite, verbose):
    """Iterate over `data`, a dict with fpath, url and download."""
    for fpath, url in data.items():
        if op.exists(fpath) and not overwrite:
            continue
        _fetch_file(url=url, file_name=fpath, print_destination=verbose,
                    resume=True, timeout=10., verbose=verbose)


def fetch_matchingpennies(data_path=None, download_dataset_data=True,
                          subjects=None, overwrite=False, verbose=True):
    """Fetch the eeg_matchingpennies dataset [1]_.

    Parameters
    ----------
    data_path : str
        Path to a directory into which to place the "eeg_matchingpennies"
        directory. Defaults to "~/mne_data/mne_bids_examples"
    dataset_data : bool
        If True, download the dataset metadata. Defaults to True.
    subjects : list of str | None
        The subject identifiers to download data from. Subjects identifiers
        that are invalid will be ignored. Defaults to downloading from all
        subjects. Specify an empty list to not download any subject data.
    overwrite : bool
        Whether or not to overwrite data. Defaults to False.
    verbose : bool
        Whether or not to print output. Defaults to True.

    Returns
    -------
    data_path : str
        The path to the eeg_matchingpennies dataset.

    Notes
    -----
    Download of the "/sourcedata" directory containing the original .xdf files
    is not supported.

    References
    ----------
    .. [1] Appelhoff, S., Sauer, D., & Gill, S. S. (2018, July 22). Matching
           Pennies: A Brain Computer Interface Implementation Dataset.
           https://doi.org/10.17605/OSF.IO/CJ2DR

    """
    if data_path is None:
        data_path = op.join(op.expanduser('~'), 'mne_data',
                            'mne_bids_examples')
    data_path = op.join(data_path, 'eeg_matchingpennies')
    os.makedirs(data_path, exist_ok=True)

    if not isinstance(subjects, (list, tuple, type(None))):
        raise ValueError('Specify `subjects` as a list of str, or None.')
    if subjects is None:
        subjects = [str(ii) for ii in range(5, 12)]

    task = 'matchingpennies'
    base_url = 'https://osf.io/{}/download'

    # Download subject data
    file_suffixes = ('channels.tsv', 'eeg.eeg', 'eeg.vhdr', 'eeg.vmrk',
                     'events.tsv')
    # Mapping subjects to the identifier url suffixes
    file_key_map = {'05': ('wdb42', '3at5h', '3m8et', '7gq4s', '9q8r2'),
                    '06': ('256sk', 'p52dn', 'jk649', 'wdjk9', '5br27'),
                    '07': ('qvze6', 'z792x', '2an4r', 'u7v2g', 'uyhtd'),
                    '08': ('4safg', 'dg9b4', 'w6kn2', 'mrkag', 'u76fs'),
                    '09': ('nqjfm', '6m5ez', 'btv7d', 'daz4f', 'ue7ah'),
                    '10': ('5cfmh', 'ya8kr', 'he3c2', 'bw6fp', 'r5ydt'),
                    '11': ('6p8vr', 'ywnpg', 'p7xk2', '8u5fm', 'rjzhy'),
                    }

    for subject in subjects:
        file_keys = file_key_map.get(subject, None)

        # If wrong subject, do not attempt to download
        if file_keys is None:
            continue

        bids_sub_dir = make_bids_folders(subject, kind='eeg',
                                         output_path=data_path,
                                         make_dir=True, verbose=verbose,
                                         overwrite=overwrite)

        # Compile download data
        data = dict()
        for suffix, key in zip(file_suffixes, file_keys):
            fname = make_bids_basename(subject=subject, task=task,
                                       suffix=suffix)
            fpath = op.join(bids_sub_dir, fname)
            data[fpath] = base_url.format(key)

        # Download
        _download_data(data, overwrite, verbose)

    # If requested, download general data
    if download_dataset_data:
        file_key_map = {
            '.bidsignore': '6thgf',
            'CHANGES': 'ckmbf',
            'dataset_description.json': 'tsy4c',
            'LICENSE': 'mkhd4',
            'participants.tsv': '6mceu',
            'participants.json': 'ku2dn',
            'README': 'k8hjf',
            'task-matchingpennies_eeg.json': 'qf5d8',
            'task-matchingpennies_events.json': '3qztv',
        }
        # Compile data
        data = dict()
        for name, key in file_key_map:
            fpath = op.join(data_path, fname)
            data[fpath] = base_url.format(key)

        # Download
        _download_data(data, overwrite, verbose)

    return data_path


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
    if not data_path:
        data_path = os.path.join(os.path.expanduser('~'), 'mne_data',
                                 'mne_bids_examples')
    os.makedirs(data_path, exist_ok=True)

    for subject_id in subject_ids:  # pragma: no cover
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

    return data_path


def fetch_brainvision_testing_data(data_path=None, overwrite=False,
                                   verbose=True):
    """Download the MNE-Python testing data for the BrainVision format.

    Parameters
    ----------
    data_path : str | None
        Path to the folder where the "testing_data_BrainVision" folder will
        be stored. Defaults to '~/mne_data/mne_bids_examples'
    overwrite : bool
        Whether or not to overwrite data. Defaults to False.

    Returns
    -------
    data_path : str
        Path to the folder containing the data.

    """
    if not data_path:
        data_path = os.path.join(os.path.expanduser('~'), 'mne_data',
                                 'mne_bids_examples')
    data_path = op.join(data_path, 'testing_data_BrainVision')
    os.makedirs(data_path, exist_ok=True)

    base_url = 'https://github.com/mne-tools/mne-python/'
    base_url += 'raw/master/mne/io/brainvision/tests/data/test{}'
    file_endings = ['.vhdr', '.vmrk', '.eeg', ]

    # Compile data
    data = dict()
    for f_ending in file_endings:
        fname = os.path.join(data_path, 'test{}'.format(f_ending))
        data[fname] = base_url.format(f_ending)

    # Download
    for fpath, url in data.items():
        if op.exists(fpath) and not overwrite:
            continue
        response = urllib.request.urlopen(url)
        with open(fpath, 'wb') as fout:
            fout.write(response.read())

    return data_path
