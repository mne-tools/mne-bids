"""Utility functions to copy raw data files.

When writing BIDS datasets, we often move and/or rename raw data files. several
original data formats have properties that restrict such operations. That is,
moving/renaming raw data files naively might lead to broken files, for example
due to internal pointers that are not being updated.

"""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)
import os
import os.path as op
import re
import shutil as sh

from scipy.io import loadmat, savemat

from .read import _parse_ext


def _copytree(src, dst, **kwargs):
    """See: https://github.com/jupyterlab/jupyterlab/pull/5150."""
    try:
        sh.copytree(src, dst, **kwargs)
    except sh.Error as error:
        # `copytree` throws an error if copying to + from NFS even though
        # the copy is successful (see https://bugs.python.org/issue24564)
        if '[Errno 22]' not in str(error) or not op.exists(dst):
            raise


def _get_brainvision_encoding(vhdr_file, verbose=False):
    """Get the encoding of .vhdr and .vmrk files.

    Parameters
    ----------
    vhdr_file : str
        path to the header file
    verbose : Bool
        determine whether results should be logged.
        (default False)

    Returns
    -------
    enc : str
        encoding of the .vhdr file to pass it on to open() function
        either 'UTF-8' (default) or whatever encoding scheme is specified
        in the header

    """
    with open(vhdr_file, 'rb') as ef:
        enc = ef.read()
        if enc.find(b'Codepage=') != -1:
            enc = enc[enc.find(b'Codepage=')+9:]
            enc = enc.split()[0]
            enc = enc.decode()
            src = '(read from header)'
        else:
            enc = 'UTF-8'
            src = '(default)'
        if verbose is True:
            print('file encoding: %s %s' % (enc, src))
    return enc


def _get_brainvision_paths(vhdr_path):
    """Get the .eeg and .vmrk file paths from a BrainVision header file.

    Parameters
    ----------
    vhdr_path : str
        path to the header file

    Returns
    -------
    paths : tuple
        paths to the .eeg file at index 0 and the .vmrk file
        at index 1 of the returned tuple

    """
    fname, ext = _parse_ext(vhdr_path)
    if ext != '.vhdr':
        raise ValueError('Expecting file ending in ".vhdr",'
                         ' but got {}'.format(ext))

    # Header file seems fine
    # extract encoding from brainvision header file, or default to utf-8
    enc = _get_brainvision_encoding(vhdr_path)

    # ..and read it
    with open(vhdr_path, 'r', encoding=enc) as f:
        lines = f.readlines()

    # Try to find data file .eeg
    eeg_file_match = re.search(r'DataFile=(.*\.eeg)', ' '.join(lines))
    if not eeg_file_match:
        raise ValueError('Could not find a .eeg file link in'
                         ' {}'.format(vhdr_path))
    else:
        eeg_file = eeg_file_match.groups()[0]

    # Try to find marker file .vmrk
    vmrk_file_match = re.search(r'MarkerFile=(.*\.vmrk)', ' '.join(lines))
    if not vmrk_file_match:
        raise ValueError('Could not find a .vmrk file link in'
                         ' {}'.format(vhdr_path))
    else:
        vmrk_file = vmrk_file_match.groups()[0]

    # Make sure we are dealing with file names as is customary, not paths
    # Paths are problematic when copying the files to another system. Instead,
    # always use the file name and keep the file triplet in the same directory
    assert os.sep not in eeg_file
    assert os.sep not in vmrk_file

    # Assert the paths exist
    head, tail = op.split(vhdr_path)
    eeg_file_path = op.join(head, eeg_file)
    vmrk_file_path = op.join(head, vmrk_file)
    assert op.exists(eeg_file_path)
    assert op.exists(vmrk_file_path)

    # Return the paths
    return (eeg_file_path, vmrk_file_path)


def copyfile_ctf(src, dest):
    """Copy and rename CTF files to a new location.

    Parameters
    ----------
    src : str
        path to the source raw .ds folder
    dest : str
        path to the destination of the new bids folder.

    """
    _copytree(src, dest)
    # list of file types to rename
    file_types = ('.acq', '.eeg', '.hc', '.hist', '.infods', '.bak',
                  '.meg4', '.newds', '.res4')
    # Rename files in dest with the name of the dest directory
    fnames = [f for f in os.listdir(dest) if f.endswith(file_types)]
    bids_folder_name = op.splitext(op.split(dest)[-1])[0]
    for fname in fnames:
        ext = op.splitext(fname)[-1]
        os.rename(op.join(dest, fname),
                  op.join(dest, bids_folder_name + ext))


def copyfile_brainvision(vhdr_src, vhdr_dest):
    """Copy a BrainVision file triplet to a new location and repair links.

    Parameters
    ----------
    vhdr_src, vhdr_dest: str
        The src path of the .vhdr file to be copied and the destination
        path. The .eeg and .vmrk files associated with the .vhdr file
        will be given names as in vhdr_dest with adjusted extensions.
        Internal file pointers will be fixed.

    """
    # Get extenstion of the brainvision file
    fname_src, ext_src = _parse_ext(vhdr_src)
    fname_dest, ext_dest = _parse_ext(vhdr_dest)
    if ext_src != ext_dest:
        raise ValueError('Need to move data with same extension'
                         ' but got "{}", "{}"'.format(ext_src, ext_dest))

    eeg_file_path, vmrk_file_path = _get_brainvision_paths(vhdr_src)

    # extract encoding from brainvision header file, or default to utf-8
    enc = _get_brainvision_encoding(vhdr_src, verbose=True)

    # Copy data .eeg ... no links to repair
    sh.copyfile(eeg_file_path, fname_dest + '.eeg')

    # Write new header and marker files, fixing the file pointer links
    # For that, we need to replace an old "basename" with a new one
    # assuming that all .eeg, .vhdr, .vmrk share one basename
    __, basename_src = op.split(fname_src)
    assert basename_src + '.eeg' == op.split(eeg_file_path)[-1]
    assert basename_src + '.vmrk' == op.split(vmrk_file_path)[-1]
    __, basename_dest = op.split(fname_dest)
    search_lines = ['DataFile=' + basename_src + '.eeg',
                    'MarkerFile=' + basename_src + '.vmrk']

    with open(vhdr_src, 'r', encoding=enc) as fin:
        with open(vhdr_dest, 'w', encoding=enc) as fout:
            for line in fin.readlines():
                if line.strip() in search_lines:
                    line = line.replace(basename_src, basename_dest)
                fout.write(line)

    with open(vmrk_file_path, 'r', encoding=enc) as fin:
        with open(fname_dest + '.vmrk', 'w', encoding=enc) as fout:
            for line in fin.readlines():
                if line.strip() in search_lines:
                    line = line.replace(basename_src, basename_dest)
                fout.write(line)


def copyfile_eeglab(src, dest):
    """Copy a EEGLAB files to a new location and adjust pointer to '.fdt' file.

    Some EEGLAB .set files come with a .fdt binary file that contains the data.
    When moving a .set file, we need to check for an associated .fdt file and
    move it to an appropriate location as well as update an internal pointer
    within the .set file.

    Notes
    -----
    Work in progress. This function will abort upon the encounter of a .fdt
    file.

    """
    # Get extenstion of the EEGLAB file
    fname_src, ext_src = _parse_ext(src)
    fname_dest, ext_dest = _parse_ext(dest)
    if ext_src != ext_dest:
        raise ValueError('Need to move data with same extension'
                         ' but got {}, {}'.format(ext_src, ext_dest))

    # Extract matlab struct "EEG" from EEGLAB file
    mat = loadmat(src, squeeze_me=False, chars_as_strings=False,
                  mat_dtype=False, struct_as_record=True)
    if 'EEG' not in mat:
        raise ValueError('Could not find "EEG" field in {}'.format(src))
    eeg = mat['EEG']

    # If the data field is a string, it points to a .fdt file in src dir
    data = eeg[0][0]['data']
    if all([item in data[0, -4:] for item in '.fdt']):
        head, tail = op.split(src)
        fdt_pointer = ''.join(data.tolist()[0])
        fdt_path = op.join(head, fdt_pointer)
        fdt_name, fdt_ext = _parse_ext(fdt_path)
        if fdt_ext != '.fdt':
            raise IOError('Expected extension {} for linked data but found'
                          ' {}'.format('.fdt', fdt_ext))

        # Copy the fdt file and give it a new name
        sh.copyfile(fdt_path, fname_dest + '.fdt')

        # Now adjust the pointer in the set file
        head, tail = op.split(fname_dest + '.fdt')
        mat['EEG'][0][0]['data'] = tail
        savemat(dest, mat, appendmat=False)

    # If no .fdt file, simply copy the .set file, no modifications necessary
    else:
        sh.copyfile(src, dest)


def copyfile_bti(raw, dest):
    """Copy BTi data."""
    pdf_fname = 'c,rfDC'
    if raw.info['highpass'] is not None:
        pdf_fname = 'c,rf%0.1fHz' % raw.info['highpass']
    sh.copyfile(raw._init_kwargs['pdf_fname'],
                op.join(dest, pdf_fname))
    sh.copyfile(raw._init_kwargs['config_fname'],
                op.join(dest, 'config'))
    sh.copyfile(raw._init_kwargs['head_shape_fname'],
                op.join(dest, 'hs_file'))
