#! /usr/bin/env python
from setuptools import setup

descr = """Experimental code for BIDS using MNE."""

DISTNAME = 'mne-bids'
DESCRIPTION = descr
MAINTAINER = 'Alexandre Gramfort'
MAINTAINER_EMAIL = 'alexandre.gramfort@telecom-paristech.fr'
URL = 'http://martinos.org/mne'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/mne-tools/mne-bids'
VERSION = '0.1.dev0'

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
          packages=[
              'mne_bids'
          ],
          scripts=['bin/mne_bids']
)
