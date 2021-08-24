# -*- coding: utf-8 -*-
"""Test the MNE BIDS converter.

For each supported file format, implement a test.
"""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD-3-Clause
import os.path as op
import pytest

# This is here to handle mne-python <0.20
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne
from mne.datasets import testing
from mne.utils import requires_version
from mne.io.constants import FIFF

from mne_bids import (write_raw_bids, read_raw_bids, BIDSPath)
from mne_bids.tsv_handler import _from_tsv


base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
subject_id2 = '02'
session_id = '01'
run = '01'
acq = '01'
run2 = '02'
task = 'testing'

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)

warning_str = dict(
    channel_unit_changed='ignore:The unit for chann*.:RuntimeWarning:mne',
)


def _wrap_read_raw(read_raw):
    def fn(fname, *args, **kwargs):
        raw = read_raw(fname, *args, **kwargs)
        raw.info['line_freq'] = 60
        return raw
    return fn


_read_raw_ctf = _wrap_read_raw(mne.io.read_raw_ctf)
_read_raw_edf = _wrap_read_raw(mne.io.read_raw_edf)
_read_raw_persyst = _wrap_read_raw(mne.io.read_raw_persyst)
_read_raw_nihon = _wrap_read_raw(mne.io.read_raw_nihon)

# parametrized directory, filename and reader for EEG/iEEG data formats
test_convert_data = [
    ('EDF', 'test_reduced.edf', _read_raw_edf),
    ('CTF', 'testdata_ctf.ds', _read_raw_ctf),
]

# parametrization for converting datasets to BrainVision
test_convertbrainvision_data = [
    ('EDF', 'test_reduced.edf', _read_raw_edf),
    ('Persyst', 'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay', _read_raw_persyst),  # noqa
    ('NihonKohden', 'MB0400FU.EEG', _read_raw_nihon)
]


@pytest.mark.parametrize(
    'dir_name, fname, reader', test_convertbrainvision_data)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_convert_brainvision(dir_name, fname, reader, _bids_validate, tmpdir):
    """Test conversion of EEG/iEEG manufacturer format to BrainVision.

    BrainVision should correctly store data from pybv>=0.5 that
    has different non-voltage units.
    """
    bids_root = tmpdir.mkdir('bids1')
    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    # the BIDS path for test datasets to get written to
    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg')

    raw = reader(raw_fname)
    kwargs = dict(raw=raw, bids_path=bids_path, overwrite=True)

    # alter some channels manually; in NK and Persyst, this will cause
    # channel to not have units
    raw.set_channel_types({raw.info['ch_names'][0]: 'stim'})

    if dir_name == 'NihonKohden':
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "short" format'):
            bids_output_path = write_raw_bids(**kwargs)
    else:
        with pytest.warns(RuntimeWarning,
                          match='Encountered data in "double" format'):
            bids_output_path = write_raw_bids(**kwargs)

    # channel units should stay the same
    raw2 = read_raw_bids(bids_output_path)
    assert all([ch1['unit'] == ch2['unit'] for ch1, ch2 in
                zip(raw.info['chs'], raw2.info['chs'])])
    assert raw2.info['chs'][0]['unit'] == FIFF.FIFF_UNIT_NONE

    # load in the channels tsv and the channel unit should be not set
    channels_fname = bids_output_path.update(
        suffix='channels', extension='.tsv')
    channels_tsv = _from_tsv(channels_fname)
    assert channels_tsv['units'][0] == 'n/a'

    # write_raw_bids should have converted the dataset to desired format
    raw = read_raw_bids(bids_output_path)
    assert raw.filenames[0].endswith('.eeg')
    assert bids_output_path.extension == '.vhdr'


@requires_version('mne', '0.22')
@pytest.mark.parametrize(
    'dir_name, fname, reader', test_convertbrainvision_data)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_error_write_meg_as_eeg(dir_name, fname, reader, tmpdir):
    """Test error writing as BrainVision EEG data for MEG."""
    bids_root = tmpdir.mkdir('bids1')
    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg',
                                         extension='.vhdr')
    raw = reader(raw_fname)
    kwargs = dict(raw=raw, bids_path=bids_path.update(datatype='meg'))

    # if we accidentally add MEG channels, then an error will occur
    raw.set_channel_types({raw.info['ch_names'][0]: 'mag'})
    with pytest.raises(ValueError, match='Got file extension .*'
                                         'for MEG data'):
        write_raw_bids(**kwargs)


@pytest.mark.parametrize('dir_name, fname, reader', test_convert_data)
@pytest.mark.filterwarnings(warning_str['channel_unit_changed'])
def test_convert_raw_errors(dir_name, fname, reader, tmp_path):
    """Test errors when converting raw file formats."""
    bids_root = tmp_path.mkdir('bids1')

    data_path = op.join(testing.data_path(), dir_name)
    raw_fname = op.join(data_path, fname)

    # the BIDS path for test datasets to get written to
    bids_path = _bids_path.copy().update(root=bids_root, datatype='eeg')

    # test conversion to BrainVision/FIF
    raw = reader(raw_fname)
    kwargs = dict(raw=raw, bids_path=bids_path, overwrite=True)

    # only accepted keywords will work for the 'format' parameter
    with pytest.raises(ValueError, match='The input "format" .* is '
                                         'not an accepted input format for '
                                         '`write_raw_bids`'):
        kwargs['format'] = 'blah'
        write_raw_bids(**kwargs)

    # write should fail when trying to convert to wrong data format for
    # the datatype inside the file (e.g. EEG -> 'FIF' or MEG -> 'BrainVision')
    with pytest.raises(ValueError, match='The input "format" .* is not an '
                                         'accepted input format for '
                                         '.* datatype.'):
        if dir_name == 'CTF':
            new_format = 'BrainVision'
        else:
            new_format = 'FIF'
        kwargs['format'] = new_format
        write_raw_bids(**kwargs)
