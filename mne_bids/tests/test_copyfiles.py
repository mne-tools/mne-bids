"""Testing copyfile functions."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op
import datetime
from distutils.version import LooseVersion

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
from mne_bids import BIDSPath
from mne_bids.utils import _handle_datatype
from mne_bids.path import _parse_ext

from mne_bids.copyfiles import (_get_brainvision_encoding,
                                _get_brainvision_paths,
                                copyfile_brainvision,
                                copyfile_edf,
                                copyfile_eeglab,
                                copyfile_kit)


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


@pytest.mark.filterwarnings('ignore:.*Exception ignored.*:'
                            'pytest.PytestUnraisableExceptionWarning')
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

    # Test with anonymization
    raw = mne.io.read_raw_brainvision(raw_fname)
    prev_date = raw.info['meas_date']
    anonymize = {'daysback': 32459}
    copyfile_brainvision(raw_fname, new_name, anonymize, verbose=True)
    raw = mne.io.read_raw_brainvision(new_name)
    new_date = raw.info['meas_date']
    assert new_date == (prev_date - datetime.timedelta(days=32459))


def test_copyfile_edf():
    """Test the anonymization of EDF/BDF files"""
    bids_root = _TempDir()
    data_path = op.join(base_path, 'edf', 'tests', 'data')

    # Test regular copying
    for ext in ['.edf', '.bdf']:
        raw_fname = op.join(data_path, 'test' + ext)
        new_name = op.join(bids_root, 'test_copy' + ext)
        copyfile_edf(raw_fname, new_name)

    # IO error testing
    with pytest.raises(ValueError, match='Need to move data with same'):
        raw_fname = op.join(data_path, 'test.edf')
        new_name = op.join(bids_root, 'test_copy.bdf')
        copyfile_edf(raw_fname, new_name)

    # Add some subject info to an EDF to test anonymization
    testfile = op.join(bids_root, 'test_copy.edf')
    raw_date = mne.io.read_raw_edf(testfile).info['meas_date']
    date = datetime.datetime.strftime(raw_date, "%d-%b-%Y").upper()
    test_id_info = '023 F 02-AUG-1951 Jane'
    test_rec_info = 'Startdate {0} ID-123 John BioSemi_ActiveTwo'.format(date)
    with open(testfile, 'r+b') as f:
        f.seek(8)
        f.write(bytes(test_id_info.ljust(80), 'ascii'))
        f.write(bytes(test_rec_info.ljust(80), 'ascii'))

    # Test date anonymization
    def _edf_get_real_date(fpath):
        with open(fpath, 'rb') as f:
            f.seek(88)
            rec_info = f.read(80).decode('ascii').rstrip()
        startdate = rec_info.split(' ')[1]
        return datetime.datetime.strptime(startdate, "%d-%b-%Y")

    bids_root2 = _TempDir()
    infile = op.join(bids_root, 'test_copy.edf')
    outfile = op.join(bids_root2, 'test_copy_anon.edf')
    anonymize = {'daysback': 33459, 'keep_his': False}
    copyfile_edf(infile, outfile, anonymize)
    prev_date = _edf_get_real_date(infile)
    new_date = _edf_get_real_date(outfile)
    assert new_date == (prev_date - datetime.timedelta(days=33459))

    # Test full ID info anonymization
    anon_startdate = datetime.datetime.strftime(new_date, "%d-%b-%Y").upper()
    with open(outfile, 'rb') as f:
        f.seek(8)
        id_info = f.read(80).decode('ascii').rstrip()
        rec_info = f.read(80).decode('ascii').rstrip()
    rec_info_tmp = "Startdate {0} X mne-bids_anonymize X"
    assert id_info == "0 X X X"
    assert rec_info == rec_info_tmp.format(anon_startdate)

    # Test partial ID info anonymization
    outfile2 = op.join(bids_root2, 'test_copy_anon_partial.edf')
    anonymize = {'daysback': 33459, 'keep_his': True}
    copyfile_edf(infile, outfile2, anonymize)
    with open(outfile2, 'rb') as f:
        f.seek(8)
        id_info = f.read(80).decode('ascii').rstrip()
        rec_info = f.read(80).decode('ascii').rstrip()
    rec = 'Startdate {0} ID-123 John BioSemi_ActiveTwo'.format(anon_startdate)
    assert id_info == "023 F X X"
    assert rec_info == rec


@pytest.mark.parametrize('fname',
                         ('test_raw.set', 'test_raw_chanloc.set'))
def test_copyfile_eeglab(fname):
    """Test the copying of EEGlab set and fdt files."""
    if (fname == 'test_raw_chanloc.set' and
            LooseVersion(testing.get_version()) < LooseVersion('0.112')):
        return

    bids_root = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, fname)
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
    if fname == 'test_raw_chanloc.set':
        with pytest.warns(RuntimeWarning,
                          match="The data contains 'boundary' events"):
            raw = mne.io.read_raw_eeglab(new_name)
            assert 'Fp1' in raw.ch_names
    else:
        raw = mne.io.read_raw_eeglab(new_name)
        assert 'EEG 001' in raw.ch_names
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

    raw = mne.io.read_raw_kit(
        raw_fname, mrk=hpi_fname, elp=electrode_fname,
        hsp=headshape_fname)
    _, ext = _parse_ext(raw_fname, verbose=True)
    datatype = _handle_datatype(raw)

    bids_path = BIDSPath(
        subject=subject_id, session=session_id, run=run, acquisition=acq,
        task=task)
    kit_bids_path = bids_path.copy().update(acquisition=None,
                                            datatype=datatype,
                                            root=output_path)
    bids_fname = str(bids_path.copy().update(datatype=datatype,
                                             suffix=datatype,
                                             extension=ext,
                                             root=output_path))

    copyfile_kit(raw_fname, bids_fname, subject_id, session_id,
                 task, run, raw._init_kwargs)
    assert op.exists(bids_fname)
    _, ext = _parse_ext(hpi_fname, verbose=True)
    if ext == '.sqd':
        kit_bids_path.update(suffix='markers', extension='.sqd')
        assert op.exists(kit_bids_path)
    elif ext == '.mrk':
        kit_bids_path.update(suffix='markers', extension='.mrk')
        assert op.exists(kit_bids_path)

    if op.exists(electrode_fname):
        task, run, key = None, None, 'ELP'
        elp_ext = '.pos'
        elp_fname = BIDSPath(
            subject=subject_id, session=session_id, task=task, run=run,
            acquisition=key, suffix='headshape', extension=elp_ext,
            datatype='meg', root=output_path)
        assert op.exists(elp_fname)

    if op.exists(headshape_fname):
        task, run, key = None, None, 'HSP'
        hsp_ext = '.pos'
        hsp_fname = BIDSPath(
            subject=subject_id, session=session_id, task=task, run=run,
            acquisition=key, suffix='headshape', extension=hsp_ext,
            datatype='meg', root=output_path)
        assert op.exists(hsp_fname)
