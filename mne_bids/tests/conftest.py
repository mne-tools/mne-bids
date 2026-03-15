"""Configure tests."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import platform
import re
import shutil

import pytest
from mne.utils import run_subprocess
from packaging.version import Version


@pytest.fixture(scope="session")
def _bids_validate():
    """Fixture to run BIDS validator."""
    shell = _use_shell()

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


@pytest.fixture(scope="session")
def _validator_version():
    """Return bids-validator version or None if unknown/uninstalled."""
    shell = _use_shell()
    validator_path = shutil.which("bids-validator")
    npx_path = shutil.which("npx")

    if validator_path is not None:
        cmd = [validator_path, "--version"]
    elif npx_path is not None:
        cmd = [npx_path, "bids-validator@latest", "--version"]
    else:
        return None

    out = run_subprocess(cmd, shell=shell)[0]
    match = re.search(r"\d+\.\d+\.\d+", out)  # MAJOR.MINOR.PATCH
    if match is None:
        return None
    return Version(match.group(0))


@pytest.fixture(scope="session")
def _using_legacy_validator(_validator_version):
    """Is `bids-validator --version` < 2."""
    return _validator_version is not None and _validator_version.major < 2


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
