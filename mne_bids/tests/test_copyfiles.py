"""Testing copyfile functions."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import pytest

from scipy.io import savemat

# This is here to handle mne-python <0.20
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne

from mne.datasets import testing
from mne.utils import _TempDir
from mne_bids.utils import _handle_kind, _parse_ext

from mne_bids.copyfiles import (_get_brainvision_encoding,
                                _get_brainvision_paths,
                                copyfile_brainvision,
                                copyfile_eeglab,
                                copyfile_kit)

from mne_bids.write import make_bids_basename

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
    bids_root = _TempDir()
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')
    new_name = op.join(bids_root, 'tested_conversion.vhdr')

    # IO error testing
    with pytest.raises(ValueError, match='Need to move data with same'):
        copyfile_brainvision(raw_fname, new_name + '.eeg')

    # Try to copy the file
    copyfile_brainvision(raw_fname, new_name, verbose=True)

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
    bids_root = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw.set')
    new_name = op.join(bids_root, 'tested_conversion.set')

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


def test_copyfile_kit():
    """Test copying and renaming KIT files to a new location."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'kit', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.sqd')
    hpi_fname = op.join(data_path, 'test_mrk.sqd')
    electrode_fname = op.join(data_path, 'test.elp')
    headshape_fname = op.join(data_path, 'test.hsp')
    subject_id = '01'
    session_id = '01'
    run = '01'
    acq = '01'
    task = 'testing'

    bids_basename = make_bids_basename(
        subject=subject_id, session=session_id, run=run, acquisition=acq,
        task=task)

    kit_bids_basename = bids_basename.replace('_acq-01', '')

    raw = mne.io.read_raw_kit(
        raw_fname, mrk=hpi_fname, elp=electrode_fname,
        hsp=headshape_fname)
    _, ext = _parse_ext(raw_fname, verbose=True)
    kind = _handle_kind(raw)
    bids_fname = bids_basename + '_%s%s' % (kind, ext)
    bids_fname = op.join(output_path, bids_fname)

    copyfile_kit(raw_fname, bids_fname, subject_id, session_id,
                 task, run, raw._init_kwargs)
    assert op.exists(bids_fname)
    _, ext = _parse_ext(hpi_fname, verbose=True)
    if ext == '.sqd':
        assert op.exists(op.join(
            output_path, kit_bids_basename + '_markers.sqd'))
    elif ext == '.mrk':
        assert op.exists(op.join(
            output_path, kit_bids_basename + '_markers.mrk'))

    if op.exists(electrode_fname):
        task, run, key = None, None, 'ELP'
        elp_ext = '.pos'
        elp_fname = make_bids_basename(
            subject=subject_id, session=session_id, task=task, run=run,
            acquisition=key, suffix='headshape%s' % elp_ext,
            prefix=output_path)
        assert op.exists(elp_fname)

    if op.exists(headshape_fname):
        task, run, key = None, None, 'HSP'
        hsp_ext = '.pos'
        hsp_fname = make_bids_basename(
            subject=subject_id, session=session_id, task=task, run=run,
            acquisition=key, suffix='headshape%s' % hsp_ext,
            prefix=output_path)
        assert op.exists(hsp_fname)
