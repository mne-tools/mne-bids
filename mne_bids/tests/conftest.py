"""Configure tests."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import platform

import pytest
from mne.utils import run_subprocess


@pytest.fixture(scope="session")
def _bids_validate():
    """Fixture to run BIDS validator."""
    # See: https://stackoverflow.com/q/28891053/5201771
    # On Windows, shell must be True
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    def _validate(bids_root):
        cmd = ["bids-validator", bids_root]
        run_subprocess(cmd, shell=shell)

    return _validate


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
