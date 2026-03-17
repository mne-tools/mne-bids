"""Configure tests."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import re
import shutil

import pytest
from mne.utils import run_subprocess
from packaging.version import Version


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
    match = re.search(r"\d+\.\d+\.\d+", out)
    if match is None:
        return None
    return Version(match.group(0))


def _get_validator_cmd(validator_args: list[str] | None = None):
    """Return the command used to invoke the BIDS validator."""
    if validator_args is None:
        validator_args = []

    deno_path = shutil.which("deno")
    # Fallback for devs who don't have deno but do have the PyPI CLI
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
