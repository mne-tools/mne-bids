"""Configure tests."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform

import pytest
from mne.utils import run_subprocess


# WINDOWS issues:
# the bids-validator development version does not work properly on Windows as
# of 2019-06-25 --> https://github.com/bids-standard/bids-validator/issues/790
# As a workaround, we try to get the path to the executable from an environment
# variable VALIDATOR_EXECUTABLE ... if this is not possible we assume to be
# using the stable bids-validator and make a direct call of bids-validator
# also: for windows, shell = True is needed to call npm, bids-validator etc.
# see: https://stackoverflow.com/q/28891053/5201771
@pytest.fixture(scope="session")
def _bids_validate():
    """Fixture to run BIDS validator."""
    vadlidator_args = ["--config.error=41"]
    exe = os.getenv("VALIDATOR_EXECUTABLE", "bids-validator")

    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    bids_validator_exe = [exe, *vadlidator_args]

    def _validate(bids_root):
        cmd = [*bids_validator_exe, bids_root]
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
