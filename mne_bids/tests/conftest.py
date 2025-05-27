"""Configure tests."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import platform
import shutil

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

    # If neither bids-validator nor npx are available, we cannot validate BIDS
    # datasets, so we raise an exception.
    # If both are available, we prefer bids-validator, but we can use npx as a fallback.

    has_validator = shutil.which("bids-validator") is not None
    has_npx = shutil.which("npx") is not None

    if not has_validator and not has_npx:
        raise FileNotFoundError(
            "⛔️ bids-validator or npx is required to run BIDS validation tests. "
            "Please install the BIDS validator or ensure npx is available."
        )
    elif not has_validator:

        def _validate(bids_root):
            cmd = ["npx", "bids-validator@latest", bids_root]
            run_subprocess(cmd, shell=shell)
    else:

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
