"""Configure all tests."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause
import os
from datetime import datetime

import mne
import mne.datasets.utils
import pytest


def pytest_configure(config):
    """Configure pytest options."""
    # Fixtures
    config.addinivalue_line("usefixtures", "monkeypatch_mne")
    # inspect_dataset() calls plt.show(block=True) on any backend but agg, so a test
    # that reaches it before something else has switched backends hangs forever
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use(os.environ["MPLBACKEND"], force=True)


@pytest.fixture(autouse=True)
def close_all():
    """Close all figures after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="session")
def monkeypatch_mne():
    """Monkeypatch MNE to ensure we have download=False everywhere in tests."""
    mne.datasets.utils._MODULES_TO_ENSURE_DOWNLOAD_IS_FALSE_IN_TESTS = (
        "mne",
        "mne_bids",
    )


class WindowsDatetime(datetime):
    """Datetime obj that will raise on tz inference of pre-epoch naive timestamp."""

    def astimezone(self, tz=None):
        """Convert to specified timezone."""
        if self.year < 1970 and self.tzinfo is None:
            raise OSError("simulated Windows pre-epoch failure")
        return super().astimezone(tz)


@pytest.fixture(scope="session")
def windows_datetime():
    """Return WindowsDatime."""
    return WindowsDatetime
