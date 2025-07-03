"""Custom sphinx extension to generate docs for the command line interface.

Inspired by MNE-Python's `gen_commands.py`
see: github.com/mne-tools/mne-python/blob/main/doc/sphinxext/gen_commands.py
"""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import sys
from pathlib import Path

import sphinx.util
from mne.utils import hashfunc, run_subprocess


def setup(app):
    """Set up the app."""
    app.connect("builder-inited", generate_cli_rst)


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

.. _gen_{0}:

{1}
{2}

.. rst-class:: callout

{3}

"""


def generate_cli_rst(app=None):
    """Generate the command line interface docs."""
    out_dir = Path(__file__).resolve().parent.parent / "generated"
    out_dir.mkdir(exist_ok=True)
    out_fname = out_dir / "cli.rst.new"

    cli_path = Path(__file__).resolve().parent.parent.parent / "mne_bids" / "commands"
    fnames = sorted([fname.name for fname in cli_path.glob("mne_bids*.py")])
    iterator = sphinx.util.display.status_iterator(
        fnames, "generating MNE-BIDS cli help ... ", length=len(fnames)
    )
    with out_fname.open("w", encoding="utf-8") as f:
        f.write(header)
        for fname in iterator:
            cmd_name = fname[:-3]
            run_name = cli_path / fname
            output, _ = run_subprocess(
                [sys.executable, str(run_name), "--help"], verbose=False
            )
            output = output.splitlines()

            # Swap usage and title lines
            output[0], output[2] = output[2], output[0]

            # Add header marking
            for idx in (1, 0):
                output.insert(idx, "-" * len(output[0]))

            # Add code styling for the "Usage: " line
            for li, line in enumerate(output):
                if line.startswith("Usage: mne_bids "):
                    output[li] = f"Usage: ``{line[7:]}``"
                    break

            # Turn "Options:" into field list
            if "Options:" in output:
                ii = output.index("Options:")
                output[ii] = "Options"
                output.insert(ii + 1, "-------")
                output.insert(ii + 2, "")
                output.insert(ii + 3, ".. rst-class:: field-list cmd-list")
                output.insert(ii + 4, "")
            output = "\n".join(output)
            f.write(
                command_rst.format(
                    cmd_name,
                    cmd_name.replace("mne_bids_", "mne_bids "),
                    "=" * len(cmd_name),
                    output,
                )
            )
    _replace_md5(out_fname)
    print("[Done]")


def _replace_md5(fname):
    """Replace a file based on MD5sum."""
    # adapted from sphinx-gallery
    assert fname.name.endswith(".new")
    fname_old = fname.with_suffix("")
    if fname_old.is_file() and hashfunc(fname) == hashfunc(fname_old):
        fname.unlink()
    else:
        shutil.move(str(fname), str(fname_old))


# This is useful for testing/iterating to see what the result looks like
if __name__ == "__main__":
    generate_cli_rst()
