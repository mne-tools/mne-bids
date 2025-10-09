"""Testing utilities for the MNE BIDS verbosity."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import inspect
import pkgutil

import mne_bids


def _iter_modules():
    # include top-level package module
    yield importlib.import_module("mne_bids")
    for module_info in pkgutil.walk_packages(
        mne_bids.__path__, mne_bids.__name__ + "."
    ):
        yield importlib.import_module(module_info.name)


def _has_verbose_param(obj):
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    return "verbose" in signature.parameters


def _is_verbose_decorated(obj):
    source = getattr(obj, "__source__", "")
    return "_use_log_level_" in source


def _iter_functions_and_methods(module):
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ != module.__name__:
            continue
        yield f"{module.__name__}.{name}", obj
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__:
            continue
        for method_name, method in inspect.getmembers(cls, inspect.isfunction):
            if method.__module__ != module.__name__:
                continue
            yield f"{module.__name__}.{cls.__name__}.{method_name}", method


def test_functions_with_verbose_are_decorated():
    """Test that all functions with a verbose parameter are decorated."""
    missing = []
    for module in _iter_modules():
        for qualname, obj in _iter_functions_and_methods(module):
            if _has_verbose_param(obj) and not _is_verbose_decorated(obj):
                missing.append(qualname)
    assert not missing, f"Missing @verbose decorator: {missing}"
