"""Testing utilities for file io."""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import pytest

from mne.utils import _TempDir

from mne_bids.read import _parse_ext, _read_raw


def test_parse_ext():
    """Test the file extension extraction."""
    f = 'sub-05_task-matchingpennies.vhdr'
    fname, ext = _parse_ext(f)
    assert fname == 'sub-05_task-matchingpennies'
    assert ext == '.vhdr'

    # Test for case where no extension: assume BTI format
    f = 'sub-01_task-rest'
    fname, ext = _parse_ext(f)
    assert fname == f
    assert ext == '.pdf'


def test_read_raw():
    """Test the raw reading."""
    # Use a file ending that does not exist
    f = 'file.bogus'
    with pytest.raises(ValueError, match='file name extension must be one of'):
        _read_raw(f)


def test_not_implemented():
    """Test the not yet implemented data formats raise an adequate error."""
    for not_implemented_ext in ['.mef', '.nwb']:
        data_path = _TempDir()
        raw_fname = op.join(data_path, 'test' + not_implemented_ext)
        with open(raw_fname, 'w'):
            pass
        with pytest.raises(ValueError, match=('there is no IO support for '
                                              'this file format yet')):
            _read_raw(raw_fname)
