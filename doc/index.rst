.. mne_bids documentation master file, created by
   sphinx-quickstart on Wed Sep  6 04:42:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MNE-BIDS
========

This is a library for converting existing files into BIDS compatible structure.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``mne_bids``, you first need to install its dependencies::

    $ pip install pandas mne

Then install mne_bids::

    $ pip install git+https://github.com/mne-tools/mne-bids.git#egg=mne-bids

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

    $ python -c 'import mne_bids'

and it should not give any error messages.

Quickstart
==========

Currently, we support writing of BIDS datasets for MEG and EEG. Support for
iEEG is experimental at the moment.

.. code:: python

    >>> from mne_bids import read_raw, write_raw_bids
    >>> raw = read_raw('my_old_file.fif')
    >>> write_raw_bids(raw, 'sub-01_ses-01_run-05_meg.fif',
                       output_path='./bids_dataset')

Reading of BIDS data will also be supported in the next version.

Bug reports
===========

Use the `github issue tracker <https://github.com/mne-tools/mne-bids/issues>`_ to report bugs.

:ref:`whats_new`
================
