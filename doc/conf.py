"""Configure details for documentation with sphinx."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from datetime import date

from intersphinx_registry import get_intersphinx_mapping
from sphinx.config import is_serializable

import mne_bids

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "mne_bids")))
sys.path.append(os.path.abspath(os.path.join(curdir, "sphinxext")))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "2.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
    "sphinx_copybutton",
    "gen_cli",  # custom extension, see ./sphinxext/gen_cli.py
    "gh_substitutions",  # custom extension, see ./sphinxext/gh_substitutions.py
]

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# configure numpydoc
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_aliases = {
    "BIDSPath": ":class:`BIDSPath <mne_bids.BIDSPath>`",
    "path-like": ":term:`path-like <mne:path-like>`",
    "array-like": ":term:`array_like <numpy:array_like>`",
    "int": ":class:`int <python:int>`",
    "bool": ":class:`bool <python:bool>`",
    "float": ":class:`float <python:float>`",
    "list": ":class:`list <python:list>`",
    "tuple": ":class:`tuple <python:tuple>`",
    "NibabelImageObject": "nibabel.spatialimages.SpatialImage",
}
numpydoc_xref_ignore = {
    # words
    "instance",
    "instances",
    "of",
}


# generate autosummary even if no references
autosummary_generate = True
autodoc_default_options = {"inherited-members": None}
default_role = "autolink"  # XXX silently allows bad syntax, someone should fix

# configure linkcheck
# https://sphinx-doc.org/en/master/usage/configuration.html?#options-for-the-linkcheck-builder
linkcheck_retries = 3
linkcheck_rate_limit_timeout = 15.0
linkcheck_ignore = [
    r"https://www.researchgate.net/profile/.*",
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "MNE-BIDS"
today = date.today().isoformat()
copyright = f"2017, The MNE-BIDS developers. Last updated on {today}"  # noqa: A001

author = "The MNE-BIDS developers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = mne_bids.__version__
# The full version, including alpha/beta/rc tags.
release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["auto_examples/index.rst", "_build", "Thumbs.db", ".DS_Store"]

nitpick_ignore_regex = [
    # TODO can be removed when min. Sphinx version is 8.2
    ("py:class", r".*pathlib\._local\.Path"),
]

# HTML options (e.g., theme)
html_show_sourcelink = False
html_copy_source = False

html_theme = "pydata_sphinx_theme"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["style.css"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
switcher_version_match = "dev" if "dev" in release else version
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/mne-tools/mne-bids",
            icon="fab fa-github-square",
        ),
        dict(
            name="Discourse",
            url="https://mne.discourse.group/tags/mne-bids",
            icon="fab fa-discourse",
        ),
    ],
    "icon_links_label": "Quick Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "analytics": dict(google_analytics_id="G-C8SH9E98QC"),
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/mne-tools/mne-bids/main/doc/_static/versions.json",  # noqa: E501
        "version_match": switcher_version_match,
    },
}

html_context = {
    "default_mode": "auto",
    "doc_path": "doc",
}

html_sidebars = {}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = get_intersphinx_mapping(
    packages={
        "matplotlib",
        "mne",
        "nibabel",
        "nilearn",
        "numpy",
        "pandas",
        "python",
        "scipy",
    }
)
intersphinx_mapping["mne-gui-addons"] = ("https://mne.tools/mne-gui-addons", None)
intersphinx_timeout = 5

sphinx_gallery_conf = {
    "doc_module": "mne_bids",
    "reference_url": {
        "mne_bids": None,
    },
    "backreferences_dir": "generated",
    "examples_dirs": "../examples",
    "within_subsection_order": "mne_bids.utils._example_sorter",
    "gallery_dirs": "auto_examples",
    "filename_pattern": "^((?!sgskip).)*$",
}

assert is_serializable(sphinx_gallery_conf)
