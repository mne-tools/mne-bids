"""Configure details for documentation with sphinx."""
import os
import sys
from datetime import date

import sphinx_gallery  # noqa: F401
from sphinx_gallery.sorting import ExampleTitleSortKey
import sphinx_bootstrap_theme

import mne_bids


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, '..', 'mne_bids')))
sys.path.append(os.path.abspath(os.path.join(curdir, 'sphinxext')))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
    'sphinx_copybutton',
    'gen_cli',  # custom extension, see ./sphinxext/gen_cli.py
    'gh_substitutions',  # custom extension, see ./sphinxext/gh_substitutions.py
]

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# configure numpydoc
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_aliases = {
    'NibabelImageObject': 'nibabel.spatialimages.SpatialImage',
}
numpydoc_xref_ignore = {
    # words
    'of',
}


# generate autosummary even if no references
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}
default_role = 'autolink'  # XXX silently allows bad syntax, someone should fix

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'mne_bids'
td = date.today()
copyright = u'2017-%s, MNE Developers. Last updated on %s' % (td.year,
                                                              td.isoformat())

author = u'MNE Developers'

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
exclude_patterns = ['auto_examples/index.rst', '_build', 'Thumbs.db',
                    '.DS_Store']

# HTML options (e.g., theme)
# see: https://sphinx-bootstrap-theme.readthedocs.io/en/latest/README.html
# Clean up sidebar: Do not show "Source" link
html_show_sourcelink = False

html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = ['style.css']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'navbar_title': 'MNE-BIDS',
    'bootswatch_theme': "flatly",
    'navbar_sidebarrel': False,  # no "previous / next" navigation
    'navbar_pagenav': False,  # no "Page" navigation in sidebar
    'bootstrap_version': "3",
    'navbar_links': [
        ("News", "whats_new"),
        ("Install", "install"),
        ("Use", "use"),
        ("API", "api"),
        ("CLI", "generated/cli"),
        ("Contribute!", "contribute")
    ]}

html_sidebars = {'**': ['localtoc.html']}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'mne': ('https://mne.tools/dev', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
    'nilearn': ('https://nilearn.github.io', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/dev', None),
    'nibabel': ('https://nipy.org/nibabel', None),
}
intersphinx_timeout = 5

# Resolve binder filepath_prefix. From the docs:
# "A prefix to append to the filepath in the Binder links. You should use this
# if you will store your built documentation in a sub-folder of a repository,
# instead of in the root."
# we will store dev docs in a `dev` subdirectory and all other docs in a
# directory "v" + version_str. E.g., "v0.3"
if 'dev' in version:
    filepath_prefix = 'dev'
else:
    filepath_prefix = 'v{}'.format(version)

sphinx_gallery_conf = {
    'doc_module': 'mne_bids',
    'reference_url': {
        'mne_bids': None,
    },
    'backreferences_dir': 'generated',
    'examples_dirs': '../examples',
    'within_subsection_order': ExampleTitleSortKey,
    'gallery_dirs': 'auto_examples',
    'filename_pattern': '^((?!sgskip).)*$',
    'binder': {
        # Required keys
        'org': 'mne-tools',
        'repo': 'mne-bids',
        'branch': 'gh-pages',  # noqa: E501 Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
        'binderhub_url': 'https://mybinder.org',  # noqa: E501 Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
        'filepath_prefix': filepath_prefix,  # noqa: E501 A prefix to prepend to any filepaths in Binder links.
        'dependencies': [
            '../test_requirements.txt',
            './requirements.txt',
        ],
    }
}
