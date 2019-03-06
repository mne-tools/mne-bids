.. mne_bids documentation master file, created by
   sphinx-quickstart on Wed Sep  6 04:42:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MNE-BIDS
========

This is a library for converting existing files into BIDS compatible structure.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_ and Python 3.

Besides numpy, scipy and matplotlib (which are included in the standard Anaconda installation), 
you will need to install the most recent version of MNE using the `pip` command line tool::

    $ pip install -U mne

Then install mne_bids::

    $ pip install -U mne-bids

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

    $ python -c 'import mne_bids'

and it should not give any error messages.

If you want to use the latest development version, use the following command::

    $ pip install https://api.github.com/repos/mne-tools/mne-bids/zipball/master

Quickstart
==========

Currently, we support writing of BIDS datasets for MEG and EEG. Support for
iEEG is experimental at the moment.

.. code:: python

    >>> from mne import io
    >>> from mne_bids import write_raw_bids
    >>> raw = io.read_raw_fif('my_old_file.fif')
    >>> write_raw_bids(raw, 'sub-01_ses-01_run-05', output_path='./bids_dataset')

Reading of BIDS data will also be supported in the next version.

Bug reports
===========

Use the `github issue tracker <https://github.com/mne-tools/mne-bids/issues>`_ to report bugs.

:ref:`whats_new`
================
