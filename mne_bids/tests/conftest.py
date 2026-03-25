"""Configure tests."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import re
import shutil

import numpy as np
import pytest
from mne import read_trans
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.transforms import apply_trans
from mne.utils import run_subprocess
from packaging.version import Version

test_path = testing.data_path(download=False)


@pytest.fixture(scope="session")
def _bids_validate():
    """Fixture to run BIDS validator."""

    def _validate(bids_root):
        cmd = _get_validator_cmd(validator_args=[bids_root])
        if cmd is None:
            raise FileNotFoundError(
                "⛔️ A BIDS validator runtime is required to run validation tests. "
                "Ensure Deno is available or install bids-validator-deno from PyPI"
            )
        run_subprocess(cmd, shell=_use_shell())

    return _validate


@pytest.fixture(scope="session")
def _validator_version():
    """Return bids-validator version or None if unknown/uninstalled."""
    cmd = _get_validator_cmd(validator_args=["--version"])
    if cmd is None:
        return None

    out = run_subprocess(cmd, shell=_use_shell())[0]
    match = re.search(r"\d+\.\d+\.\d+", out)  # MAJOR.MINOR.PATCH
    if match is None:
        return None
    return Version(match.group(0))


def _get_validator_cmd(validator_args: list[str] | None = None):
    """Return the command used to invoke the BIDS validator."""
    if validator_args is None:
        validator_args = []

    deno_path = shutil.which("deno")
    # Fallback for devs who don't have deno but do have the Python CLI
    validator_cli_path = shutil.which("bids-validator-deno")

    requested_version = os.getenv("BIDS_VALIDATOR_VERSION", "stable")
    dev_validator_url = (
        "https://github.com/bids-standard/bids-validator/raw/deno-build/"
        "bids-validator.js"
    )

    if deno_path is not None:
        if requested_version == "dev":
            package_spec = dev_validator_url
        elif requested_version == "stable":
            package_spec = "jsr:@bids/validator"
        else:
            package_spec = f"jsr:@bids/validator@{requested_version}"
        return [deno_path, "-A", package_spec, *validator_args]

    if validator_cli_path is not None:
        return [validator_cli_path, *validator_args]

    return None


def _use_shell():
    """Whether to run subprocess with shell injection."""
    # See: https://stackoverflow.com/q/28891053/5201771
    # On Windows, shell must be True
    return platform.system() == "Windows"


# Deal with:
# Auto-close()ing of figures upon backend switching is deprecated since 3.8 and will
# be removed in 3.10.  To suppress this warning, explicitly call plt.close('all')
# first.
@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _load_t1():
    import nibabel as nib

    t1_path = test_path / "subjects" / "sample" / "mri" / "T1.mgz"
    t1 = nib.load(t1_path)
    return t1


def _get_head_to_vox_trans(t1):
    from numpy.linalg import inv

    vox_to_ras = t1.header.get_vox2ras_tkr()
    ras_to_vox_trans = inv(vox_to_ras)
    return ras_to_vox_trans


def _get_head_fids():
    raw_path = test_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
    raw_info = read_raw_fif(raw_path).info
    head_fids = [dig["r"] for dig in raw_info["dig"] if dig["kind"] == 1]
    head_fids = np.array(head_fids)
    return head_fids


@pytest.fixture(scope="module")
def mri_landmarks():
    trans_name = 'sample_audvis_trunc-trans.fif'
    trans_path = test_path / 'MEG' / 'sample' / trans_name
    trans = read_trans(trans_path)
    head_fids = _get_head_fids()
    t1 = _load_t1()
    head_to_mri_trans = _get_head_to_vox_trans(t1)
    mri_fids = np.zeros(shape=head_fids.shape)
    for hfi, hfid in enumerate(head_fids):
        t_fid = apply_trans(trans, hfid, move=True)
        mri_fids[hfi] = apply_trans(head_to_mri_trans, t_fid * 1e3, move=True)
    return mri_fids


@pytest.fixture(scope="module")
def t1_image():
    t1_im = _load_t1()
    return t1_im
