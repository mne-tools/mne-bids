"""Configure all tests."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import mne
import pytest


def pytest_configure(config):
    """Configure pytest options."""
    # Fixtures
    config.addinivalue_line("usefixtures", "monkeypatch_mne")


@pytest.fixture(autouse=True)
def close_all():
    """Close all figures after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="session")
def monkeypatch_mne():
    """Monkeypatch MNE to ensure we have download=False everywhere in tests."""
    # ``mne.datasets`` lazily exposes its submodules starting with MNE 1.7.
    # Accessing ``mne.datasets.utils`` without importing it first raises an
    # ``AttributeError`` under the lazy-loader implementation, so we import
    # the submodule explicitly before touching its attributes.
    import importlib

    importlib.import_module("mne.datasets.utils")
    mne.datasets.utils._MODULES_TO_ENSURE_DOWNLOAD_IS_FALSE_IN_TESTS = (
        "mne",
        "mne_bids",
    )
