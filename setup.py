"""Setup MNE-BIDS."""
import os
from setuptools import setup

# get the version
version = None
with open(os.path.join('mne_bids', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version.')

if __name__ == "__main__":
    setup(version=version)
