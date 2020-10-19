:orphan:

Using MNE-BIDS
==============

.. contents:: Contents
   :local:
   :depth: 1

Python API reference
--------------------
See the :doc:`API documentation <api>`.

Command line interface reference
--------------------------------
See the :doc:`CLI documentation <generated/cli>`.

Quickstart
----------
MNE-BIDS fully supports writing of BIDS datasets for MEG and EEG. Support for
iEEG is experimental at the moment.

Python
~~~~~~

.. code:: python

    >>> import mne
    >>> from mne_bids import BIDSPath, write_raw_bids
    >>> raw = mne.io.read_raw_fif('my_old_file.fif')
    >>> bids_path = BIDSPath(subject='01', session='01, run='05',
                             datatype='meg', bids_root='./bids_dataset')
    >>> write_raw_bids(raw, bids_path=bids_path)

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Simply type ``mne_bids`` in your command line and press enter to see a list of
accepted commands. Then type ``mne_bids <command> --help``, where ``<command>``
is one of the accepted commands, to get more information about it.

Example:

.. code-block:: bash

  $ mne_bids raw_to_bids --subject_id sub01 --task rest --raw data.edf --bids_root new_path


.. _bidspath-intro:

Mastering BIDSPath
------------------
To be able to effectively use MNE-BIDS, you need to understand how to work with
the ``BIDSPath`` object. Follow `this example <auto_examples/bidspath.html>`_
to learn everything you need!


.. include:: auto_examples/index.rst
   :start-after: :orphan: 
