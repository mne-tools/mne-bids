"""Command Line Interface for MNE-BIDS."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os.path as op
import subprocess
import sys

import mne_bids

mne_bin_dir = op.abspath(op.dirname(mne_bids.__file__))
valid_commands = sorted(glob.glob(op.join(mne_bin_dir, "commands", "mne_bids_*.py")))
valid_commands = [c.split(op.sep)[-1][9:-3] for c in valid_commands]


def print_help():
    """Print the help."""
    print("Usage : mne_bids command options\n")
    print("Accepted commands :\n")
    for c in valid_commands:
        print(f"\t- {c}")
    print(
        "\nExample : mne_bids raw_to_bids --subject_id sub01 --task rest",
        "--raw_file data.edf --bids_root new_path",
    )
    sys.exit(0)


def main():
    """Run main command."""
    if len(sys.argv) == 1:
        print_help()
    elif "help" in sys.argv[1] or "-h" in sys.argv[1]:
        print_help()
    elif sys.argv[1] == "--version":
        print(f"MNE-BIDS {mne_bids.__version__}")
    elif sys.argv[1] not in valid_commands:
        print(f'Invalid command: "{sys.argv[1]}"\n')
        print_help()
        sys.exit(0)
    else:
        cmd = sys.argv[1]
        cmd_path = op.join(mne_bin_dir, "commands", f"mne_bids_{cmd}.py")
        sys.exit(subprocess.call([sys.executable, cmd_path] + sys.argv[2:]))
