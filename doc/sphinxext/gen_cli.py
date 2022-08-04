"""Custom sphinx extension to generate docs for the command line interface.

Inspired by MNE-Python's `gen_commands.py`
see: github.com/mne-tools/mne-python/blob/main/doc/sphinxext/gen_commands.py
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
import os
import glob
from os import path as op
import subprocess
import sys

from mne.utils import run_subprocess, _replace_md5


def setup(app):
    """Set up the app."""
    app.connect('builder-inited', generate_cli_rst)


# Header markings go:
# 1. =/= : Page title
# 2. =   : Command name
# 3. -/- : Command description
# 4. -   : Command sections (Examples, Notes)

header = """\
:orphan:

.. _python_cli:

=====================================
MNE-BIDS Command Line Interface (CLI)
=====================================

Here we list the MNE-BIDS tools that you can use from the command line.

"""

command_rst = """

.. _gen_%s:

%s
%s

.. rst-class:: callout

%s

"""


def generate_cli_rst(app=None):
    """Generate the command line interface docs."""
    from sphinx_gallery import sphinx_compatibility
    out_dir = op.abspath(op.join(op.dirname(__file__), '..', 'generated'))
    if not op.isdir(out_dir):
        os.mkdir(out_dir)
    out_fname = op.join(out_dir, 'cli.rst.new')

    cli_path = op.abspath(
        op.join(os.path.dirname(__file__), '..', '..', 'mne_bids', 'commands'))
    fnames = sorted([
        op.basename(fname)
        for fname in glob.glob(op.join(cli_path, 'mne_bids*.py'))])
    iterator = sphinx_compatibility.status_iterator(
        fnames, 'generating MNE-BIDS cli help ... ', length=len(fnames))
    with open(out_fname, 'w', encoding='utf-8') as f:
        f.write(header)
        for fname in iterator:
            cmd_name = fname[:-3]
            run_name = op.join(cli_path, fname)
            output, _ = run_subprocess([sys.executable, run_name, '--help'],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, verbose=False)
            output = output.splitlines()

            # Swap usage and title lines
            output[0], output[2] = output[2], output[0]

            # Add header marking
            for idx in (1, 0):
                output.insert(idx, '-' * len(output[0]))

            # Add code styling for the "Usage: " line
            for li, line in enumerate(output):
                if line.startswith('Usage: mne_bids '):
                    output[li] = 'Usage: ``%s``' % line[7:]
                    break

            # Turn "Options:" into field list
            if 'Options:' in output:
                ii = output.index('Options:')
                output[ii] = 'Options'
                output.insert(ii + 1, '-------')
                output.insert(ii + 2, '')
                output.insert(ii + 3, '.. rst-class:: field-list cmd-list')
                output.insert(ii + 4, '')
            output = '\n'.join(output)
            f.write(command_rst % (cmd_name,
                                   cmd_name.replace('mne_bids_', 'mne_bids '),
                                   '=' * len(cmd_name),
                                   output))
    _replace_md5(out_fname)
    print('[Done]')


# This is useful for testing/iterating to see what the result looks like
if __name__ == '__main__':
    generate_cli_rst()
