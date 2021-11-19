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
# License: BSD-3-Clause
import os
import os.path as op
import re
import shutil as sh
from pathlib import Path

from scipy.io import loadmat, savemat

import mne
from mne.io import (read_raw_brainvision, read_raw_edf, read_raw_bdf,
                    anonymize_info)
from mne.utils import logger, verbose, warn

from mne_bids.path import BIDSPath, _parse_ext, _mkdir_p
from mne_bids.utils import _get_mrk_meas_date, _check_anonymize


def _copytree(src, dst, **kwargs):
    """See: https://github.com/jupyterlab/jupyterlab/pull/5150."""
    try:
        sh.copytree(src, dst, **kwargs)
    except sh.Error as error:
        # `copytree` throws an error if copying to + from NFS even though
        # the copy is successful (see https://bugs.python.org/issue24564)
        if '[Errno 22]' not in str(error) or not op.exists(dst):
            raise


def _get_brainvision_encoding(vhdr_file):
    """Get the encoding of .vhdr and .vmrk files.

    Parameters
    ----------
    vhdr_file : str
        Path to the header file.

    Returns
    -------
    enc : str
        Encoding of the .vhdr file to pass it on to open() function
        either 'UTF-8' (default) or whatever encoding scheme is specified
        in the header.

    """
    with open(vhdr_file, 'rb') as ef:
        enc = ef.read()
        if enc.find(b'Codepage=') != -1:
            enc = enc[enc.find(b'Codepage=') + 9:]
            enc = enc.split()[0]
            enc = enc.decode()
            src = '(read from header)'
        else:
            enc = 'UTF-8'
            src = '(default)'
        logger.debug(f'Detected file encoding: {enc} {src}.')
    return enc


def _get_brainvision_paths(vhdr_path):
    """Get the .eeg and .vmrk file paths from a BrainVision header file.

    Parameters
    ----------
    vhdr_path : str
        Path to the header file.

    Returns
    -------
    paths : tuple
        Paths to the .eeg file at index 0 and the .vmrk file at index 1 of
        the returned tuple.

    """
    fname, ext = _parse_ext(vhdr_path)
    if ext != '.vhdr':
        raise ValueError(f'Expecting file ending in ".vhdr",'
                         f' but got {ext}')

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
                         f' {vhdr_path}')
    else:
        eeg_file = eeg_file_match.groups()[0]

    # Try to find marker file .vmrk
    vmrk_file_match = re.search(r'MarkerFile=(.*\.vmrk)', ' '.join(lines))
    if not vmrk_file_match:
        raise ValueError('Could not find a .vmrk file link in'
                         f' {vhdr_path}')
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
    src : path-like
        Path to the source raw .ds folder.
    dest : path-like
        Path to the destination of the new bids folder.

    See Also
    --------
    copyfile_brainvision
    copyfile_bti
    copyfile_edf
    copyfile_eeglab
    copyfile_kit

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


def copyfile_kit(src, dest, subject_id, session_id,
                 task, run, _init_kwargs):
    """Copy and rename KIT files to a new location.

    Parameters
    ----------
    src : path-like
        Path to the source raw .con or .sqd folder.
    dest : path-like
        Path to the destination of the new bids folder.
    subject_id : str | None
        The subject ID. Corresponds to "sub".
    session_id : str | None
        The session identifier. Corresponds to "ses".
    task : str | None
        The task identifier. Corresponds to "task".
    run : int | None
        The run number. Corresponds to "run".
    _init_kwargs : dict
        Extract information of marker and headpoints

    See Also
    --------
    copyfile_brainvision
    copyfile_bti
    copyfile_ctf
    copyfile_edf
    copyfile_eeglab

    """
    # create parent directories in case it does not exist yet
    _mkdir_p(op.dirname(dest))

    # KIT data requires the marker file to be copied over too
    sh.copyfile(src, dest)
    data_path = op.split(dest)[0]
    datatype = 'meg'

    if 'mrk' in _init_kwargs and _init_kwargs['mrk'] is not None:
        hpi = _init_kwargs['mrk']
        acq_map = dict()
        if isinstance(hpi, list):
            if _get_mrk_meas_date(hpi[0]) > _get_mrk_meas_date(hpi[1]):
                raise ValueError('Markers provided in incorrect order.')
            _, marker_ext = _parse_ext(hpi[0])
            acq_map = dict(zip(['pre', 'post'], hpi))
        else:
            _, marker_ext = _parse_ext(hpi)
            acq_map[None] = hpi
        for key, value in acq_map.items():
            marker_path = BIDSPath(
                subject=subject_id, session=session_id, task=task, run=run,
                acquisition=key, suffix='markers', extension=marker_ext,
                datatype=datatype)
            sh.copyfile(value, op.join(data_path, marker_path.basename))

    for acq in ['elp', 'hsp']:
        if acq in _init_kwargs and _init_kwargs[acq] is not None:
            position_file = _init_kwargs[acq]
            task, run, acq = None, None, acq.upper()
            position_ext = '.pos'
            position_path = BIDSPath(
                subject=subject_id, session=session_id, task=task, run=run,
                acquisition=acq, suffix='headshape', extension=position_ext,
                datatype=datatype)
            sh.copyfile(position_file,
                        op.join(data_path, position_path.basename))


def _replace_file(fname, pattern, replace):
    """Overwrite file, replacing end of lines matching pattern with replace."""
    new_content = []
    for line in open(fname, 'r'):
        match = re.match(pattern, line)
        if match:
            line = match.group()[:-len(replace)] + replace + '\n'
        new_content.append(line)

    with open(fname, 'w', encoding='utf-8') as fout:
        fout.writelines(new_content)


def _anonymize_brainvision(vhdr_file, date):
    """Anonymize vmrk and vhdr files in place using `date` datetime object."""
    _, vmrk_file = _get_brainvision_paths(vhdr_file)

    # Go through VMRK
    pattern = re.compile(r'^Mk\d+=New Segment,.*,\d+,\d+,\d+,\d{20}$')
    replace = date.strftime('%Y%m%d%H%M%S%f')
    _replace_file(vmrk_file, pattern, replace)

    # Go through VHDR
    pattern = re.compile(r'^Impedance \[kOhm\] at \d\d:\d\d:\d\d :$')
    replace = f'at {date.strftime("%H:%M:%S")} :'
    _replace_file(vhdr_file, pattern, replace)


@verbose
def copyfile_brainvision(vhdr_src, vhdr_dest, anonymize=None, verbose=None):
    """Copy a BrainVision file triplet to a new location and repair links.

    The BrainVision file format consists of three files: .vhdr, .eeg, and .vmrk
    The .eeg and .vmrk files associated with the .vhdr file will be given names
    as in `vhdr_dest` with adjusted extensions. Internal file pointers will be
    fixed.

    Parameters
    ----------
    vhdr_src : path-like
        The source path of the .vhdr file to be copied.
    vhdr_dest : path-like
        The destination path of the .vhdr file.
    anonymize : dict | None
        If None (default), no anonymization is performed.
        If dict, data will be anonymized depending on the keys provided with
        the dict: `daysback` is a required key, `keep_his` is an optional key.

        `daysback` : int
            Number of days by which to move back the recording date in time.
            In studies with multiple subjects the relative recording date
            differences between subjects can be kept by using the same number
            of `daysback` for all subject anonymizations. `daysback` should be
            great enough to shift the date prior to 1925 to conform with BIDS
            anonymization rules.

        `keep_his` : bool
            By default (False), all subject information next to the recording
            date will be overwritten as well. If True, keep subject information
            apart from the recording date.

    %(verbose)s

    See Also
    --------
    mne.io.anonymize_info
    copyfile_bti
    copyfile_ctf
    copyfile_edf
    copyfile_eeglab
    copyfile_kit

    """
    # Get extension of the brainvision file
    fname_src, ext_src = _parse_ext(vhdr_src)
    fname_dest, ext_dest = _parse_ext(vhdr_dest)
    if ext_src != ext_dest:
        raise ValueError(f'Need to move data with same extension, '
                         f' but got "{ext_src}" and "{ext_dest}"')

    eeg_file_path, vmrk_file_path = _get_brainvision_paths(vhdr_src)

    # extract encoding from brainvision header file, or default to utf-8
    enc = _get_brainvision_encoding(vhdr_src)

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

    if anonymize is not None:
        raw = read_raw_brainvision(vhdr_src, preload=False, verbose=0)
        daysback, keep_his, _ = _check_anonymize(anonymize, raw, '.vhdr')
        raw.info = anonymize_info(raw.info, daysback=daysback,
                                  keep_his=keep_his)
        _anonymize_brainvision(fname_dest + '.vhdr',
                               date=raw.info['meas_date'])

    for ext in ['.eeg', '.vhdr', '.vmrk']:
        _, fname = os.path.split(fname_dest + ext)
        dirname = op.dirname(op.realpath(vhdr_dest))
        logger.info(f'Created "{fname}" in "{dirname}".')
    if anonymize:
        logger.info('Anonymized all dates in VHDR and VMRK.')


def copyfile_edf(src, dest, anonymize=None):
    """Copy an EDF, EDF+, or BDF file to a new location, optionally anonymize.

    .. warning:: EDF/EDF+/BDF files contain two fields for recording dates:
                 A generic "startdate" field that supports only 2-digit years,
                 and a "Startdate" field as part of the "local recording
                 identification", which supports 4-digit years.
                 If you want to anonymize your file, MNE-BIDS will set the
                 "startdate" field to 85 (i.e., 1985), the earliest possible
                 date for that field. However, the "Startdate" field in the
                 file's "local recording identification" and the date in the
                 session's corresponding ``scans.tsv`` will be set correctly
                 according to the argument provided to the ``anonymize``
                 parameter. Note that it is possible that not all EDF/EDF+/BDF
                 reading software parses the accurate recording date, and
                 that for some reading software, the wrong year (1985) may
                 be parsed.

    Parameters
    ----------
    src : path-like
        The source path of the .edf or .bdf file to be copied.
    dest : path-like
        The destination path of the .edf or .bdf file.
    anonymize : dict | None
        If None (default), no anonymization is performed.
        If dict, data will be anonymized depending on the keys provided with
        the dict: `daysback` is a required key, `keep_his` is an optional key.

        `daysback` : int
            Number of days by which to move back the recording date in time.
            In studies with multiple subjects the relative recording date
            differences between subjects can be kept by using the same number
            of `daysback` for all subject anonymizations. `daysback` should be
            great enough to shift the date prior to 1925 to conform with BIDS
            anonymization rules. Due to limitations of the EDF/BDF format, the
            year of the anonymized date will always be set to 1985 in the
            'startdate' field of the file. The correctly-shifted year will be
            written to the 'local recording identification' region of the
            file header, which may not be parsed by all EDF/EDF+/BDF reader
            software.

        `keep_his` : bool
            By default (False), all subject information next to the recording
            date will be overwritten as well. If True, keep subject information
            apart from the recording date. Participant names and birthdates
            will always be anonymized if present, regardless of this setting.

    See Also
    --------
    mne.io.anonymize_info
    copyfile_brainvision
    copyfile_bti
    copyfile_ctf
    copyfile_eeglab
    copyfile_kit

    """
    # Ensure source & destination extensions are the same
    fname_src, ext_src = _parse_ext(src)
    fname_dest, ext_dest = _parse_ext(dest)

    if ext_src.lower() != ext_dest.lower():
        raise ValueError(f'Need to move data with same extension, '
                         f' but got "{ext_src}" and "{ext_dest}"')

    if ext_dest in ['.EDF', '.BDF']:
        warn('Upper-case extension for EDF/BDF files is not supported '
             'in BIDS. Converting destination extension to lower-case.')
        ext_dest = ext_dest.lower()
        dest = Path(dest).with_suffix(ext_dest)

    # Copy data prior to any anonymization
    sh.copyfile(src, dest)

    # Anonymize EDF/BDF data, if requested
    if anonymize is not None:
        if ext_src in ['.bdf', '.BDF']:
            raw = read_raw_bdf(dest, preload=False, verbose=0)
        elif ext_src in ['.edf', '.EDF']:
            raw = read_raw_edf(dest, preload=False, verbose=0)
        else:
            raise ValueError('Unsupported file type ({0})'.format(ext_src))

        # Get subject info, recording info, and recording date
        with open(dest, 'rb') as f:
            f.seek(8)  # id_info field starts 8 bytes in
            id_info = f.read(80).decode('ascii').rstrip()
            rec_info = f.read(80).decode('ascii').rstrip()

        # Parse metadata from file
        if len(id_info) == 0 or len(id_info.split(' ')) != 4:
            id_info = "X X X X"
        if len(rec_info) == 0 or len(rec_info.split(' ')) != 5:
            rec_info = "Startdate X X X X"
        pid, sex, birthdate, name = id_info.split(' ')
        start_date, admin_code, tech, equip = rec_info.split(' ')[1:5]

        # Try to anonymize the recording date
        daysback, keep_his, _ = _check_anonymize(anonymize, raw, '.edf')
        anonymize_info(raw.info, daysback=daysback, keep_his=keep_his)
        start_date = '01-JAN-1985'
        meas_date = '01.01.85'

        # Anonymize ID info and write to file
        if keep_his:
            # Always remove participant birthdate and name to be safe
            id_info = [pid, sex, "X", "X"]
            rec_info = ["Startdate", start_date, admin_code, tech, equip]
        else:
            id_info = ["0", "X", "X", "X"]
            rec_info = ["Startdate", start_date, "X",
                        "mne-bids_anonymize", "X"]
        with open(dest, 'r+b') as f:
            f.seek(8)  # id_info field starts 8 bytes in
            f.write(bytes(" ".join(id_info).ljust(80), 'ascii'))
            f.write(bytes(" ".join(rec_info).ljust(80), 'ascii'))
            f.write(bytes(meas_date, 'ascii'))


def copyfile_eeglab(src, dest):
    """Copy a EEGLAB files to a new location and adjust pointer to '.fdt' file.

    Some EEGLAB .set files come with a .fdt binary file that contains the data.
    When moving a .set file, we need to check for an associated .fdt file and
    move it to an appropriate location as well as update an internal pointer
    within the .set file.

    Parameters
    ----------
    src : path-like
        Path to the source raw .set file.
    dest : path-like
        Path to the destination of the new .set file.

    See Also
    --------
    copyfile_brainvision
    copyfile_bti
    copyfile_ctf
    copyfile_edf
    copyfile_kit

    """
    if not mne.utils.check_version('scipy', '1.5.0'):  # pragma: no cover
        raise ImportError('SciPy >=1.5.0 is required handling EEGLAB data.')

    # Get extension of the EEGLAB file
    _, ext_src = _parse_ext(src)
    fname_dest, ext_dest = _parse_ext(dest)
    if ext_src != ext_dest:
        raise ValueError(f'Need to move data with same extension'
                         f' but got {ext_src}, {ext_dest}')

    # Load the EEG struct
    uint16_codec = None
    eeg = loadmat(file_name=src, simplify_cells=True,
                  appendmat=False, uint16_codec=uint16_codec)
    oldstyle = False
    if 'EEG' in eeg:
        eeg = eeg['EEG']
        oldstyle = True

    if isinstance(eeg['data'], str):
        # If the data field is a string, it points to a .fdt file in src dir
        fdt_fname = eeg['data']
        assert fdt_fname.endswith('.fdt')
        head, tail = op.split(src)
        fdt_path = op.join(head, fdt_fname)

        # Copy the .fdt file and give it a new name
        sh.copyfile(fdt_path, fname_dest + '.fdt')

        # Now adjust the pointer in the .set file
        head, tail = op.split(fname_dest + '.fdt')
        eeg['data'] = tail

        # Save the EEG dictionary as a Matlab struct again
        mdict = dict(EEG=eeg) if oldstyle else eeg
        savemat(file_name=dest, mdict=mdict, appendmat=False)
    else:
        # If no .fdt file, simply copy the .set file, no modifications
        # necessary
        sh.copyfile(src, dest)


def copyfile_bti(raw, dest):
    """Copy BTi data.

    Parameters
    ----------
    raw : mne.io.Raw
        An MNE-Python raw object of BTi data.
    dest : path-like
        Destination to copy the BTi data to.

    See Also
    --------
    copyfile_brainvision
    copyfile_ctf
    copyfile_edf
    copyfile_eeglab
    copyfile_kit

    """
    pdf_fname = 'c,rfDC'
    if raw.info['highpass'] is not None:
        pdf_fname = 'c,rf%0.1fHz' % raw.info['highpass']
    sh.copyfile(raw._init_kwargs['pdf_fname'],
                op.join(dest, pdf_fname))
    sh.copyfile(raw._init_kwargs['config_fname'],
                op.join(dest, 'config'))
    sh.copyfile(raw._init_kwargs['head_shape_fname'],
                op.join(dest, 'hs_file'))
