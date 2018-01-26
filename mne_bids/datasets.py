# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
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
