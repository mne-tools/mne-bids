#! /usr/bin/env python
"""Setup MNE-BIDS."""
import os
from setuptools import setup, find_packages

# get the version
version = None
with open(os.path.join('mne_bids', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = ('MNE-BIDS: Organizing MEG, EEG, and iEEG data according to the BIDS '
         'specification and facilitating their analysis with MNE-Python')

DISTNAME = 'mne-bids'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
URL = 'https://mne.tools/mne-bids/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mne-tools/mne-bids.git'
VERSION = version

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          long_description_content_type='text/markdown',
          python_requires='~=3.6',
          install_requires=[
              'mne >=0.21',
              'numpy >=1.14',
              'scipy >=0.18.1',
          ],
          extras_require={
              'full': [
                  'nibabel >=2.2',
                  'pybv >=0.4',
                  'matplotlib',
                  'pandas >=0.23.4'
              ]
          },
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
          ],
          platforms='any',
          packages=find_packages(),
          entry_points={'console_scripts': [
              'mne_bids = mne_bids.commands.run:main',
          ]},
          project_urls={
              'Documentation': 'https://mne.tools/mne-bids',
              'Bug Reports': 'https://github.com/mne-tools/mne-bids/issues',
              'Source': 'https://github.com/mne-tools/mne-bids',
          },
          )
