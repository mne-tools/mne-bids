"""Testing copyfile functions."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import pytest

from scipy.io import savemat
import mne
from mne.datasets import testing
from mne.utils import _TempDir

from mne_bids.copyfiles import (_get_brainvision_encoding,
                                _get_brainvision_paths,
                                copyfile_brainvision, copyfile_eeglab)

base_path = op.join(op.dirname(mne.__file__), 'io')


def test_get_brainvision_encoding():
    """Test getting the file-encoding from a BrainVision header."""
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')

    with pytest.raises(UnicodeDecodeError):
        with open(raw_fname, 'r', encoding='ascii') as f:
            f.readlines()

    enc = _get_brainvision_encoding(raw_fname, verbose=True)
    with open(raw_fname, 'r', encoding=enc) as f:
        f.readlines()


def test_get_brainvision_paths():
    """Test getting the file links from a BrainVision header."""
    test_dir = _TempDir()
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')

    with pytest.raises(ValueError):
        _get_brainvision_paths(op.join(data_path, 'test.eeg'))

    # Write some temporary test files
    with open(op.join(test_dir, 'test1.vhdr'), 'w') as f:
        f.write('DataFile=testing.eeg')

    with open(op.join(test_dir, 'test2.vhdr'), 'w') as f:
        f.write('MarkerFile=testing.vmrk')

    with pytest.raises(ValueError):
        _get_brainvision_paths(op.join(test_dir, 'test1.vhdr'))

    with pytest.raises(ValueError):
        _get_brainvision_paths(op.join(test_dir, 'test2.vhdr'))

    # This should work
    eeg_file_path, vmrk_file_path = _get_brainvision_paths(raw_fname)
    head, tail = op.split(eeg_file_path)
    assert tail == 'test.eeg'
    head, tail = op.split(vmrk_file_path)
    assert tail == 'test.vmrk'


def test_copyfile_brainvision():
    """Test the copying of BrainVision vhdr, vmrk and eeg files."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')
    new_name = op.join(output_path, 'tested_conversion.vhdr')

    # IO error testing
    with pytest.raises(ValueError):
        copyfile_brainvision(raw_fname, new_name + '.eeg')

    # Try to copy the file
    copyfile_brainvision(raw_fname, new_name)

    # Have all been copied?
    head, tail = op.split(new_name)
    assert op.exists(op.join(head, 'tested_conversion.vhdr'))
    assert op.exists(op.join(head, 'tested_conversion.vmrk'))
    assert op.exists(op.join(head, 'tested_conversion.eeg'))

    # Try to read with MNE - if this works, the links are correct
    raw = mne.io.read_raw_brainvision(new_name)
    assert raw.filenames[0] == (op.join(head, 'tested_conversion.eeg'))


def test_copyfile_eeglab():
    """Test the copying of EEGlab set and fdt files."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw.set')
    new_name = op.join(output_path, 'tested_conversion.set')

    # IO error testing
    with pytest.raises(ValueError):
        copyfile_eeglab(raw_fname, new_name + '.wrong')

    # Bad .set file testing
    with pytest.raises(ValueError):
        tmp = _TempDir()
        fake_set = op.join(tmp, 'fake.set')
        savemat(fake_set, {'arr': [1, 2, 3]}, appendmat=False)
        copyfile_eeglab(fake_set, new_name)

    # Test copying and reading a combined set+fdt
    copyfile_eeglab(raw_fname, new_name)
    raw = mne.io.read_raw_eeglab(new_name)
    assert isinstance(raw, mne.io.BaseRaw)
