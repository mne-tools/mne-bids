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


descr = """Experimental code for BIDS using MNE."""

DISTNAME = 'mne-bids'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
URL = 'https://mne-tools.github.io/mne-bids/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/mne-tools/mne-bids'
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
          long_description=open('README.rst').read(),
          long_description_content_type='text/x-rst',
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
          ],
          platforms='any',
          packages=find_packages(),
          scripts=['bin/mne_bids']
          )
