from pathlib import Path

import pytest

import mne
from mne.datasets import testing
from mne.utils import requires_nibabel, requires_version
from mne_bids.viz import plot_anat_landmarks
from mne_bids import BIDSPath, write_anat


@requires_nibabel()
@requires_version('nilearn', '0.6')
def test_plot_anat_landmarks(tmpdir):
    """Test writing anatomical data with pathlib.Paths."""
    data_path = Path(testing.data_path())
    raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    trans_fname = str(raw_fname).replace('_raw.fif', '-trans.fif')
    raw = mne.io.read_raw_fif(raw_fname)
    trans = mne.read_trans(trans_fname)

    bids_root = Path(tmpdir)
    t1w_mgh_fname = Path(data_path) / 'subjects' / 'sample' / 'mri' / 'T1.mgz'
    bids_path = BIDSPath(subject='01', session='mri', root=bids_root)
    bids_path = write_anat(t1w_mgh_fname, bids_path=bids_path, overwrite=True)

    with pytest.warns(RuntimeWarning, match='No landmarks available'):
        fig = plot_anat_landmarks(bids_path, show=False)
        assert fig is None

    bids_path = write_anat(t1w_mgh_fname, bids_path=bids_path, raw=raw,
                           trans=trans, deface=True, verbose=True,
                           overwrite=True)
    fig = plot_anat_landmarks(bids_path, show=False)
    assert len(fig.axes) == 12  # 3 subplots + 3 x 3 MRI slices
