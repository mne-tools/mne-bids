:orphan:

Installation
============

.. contents:: Contents
   :local:
   :depth: 2

Dependencies
------------

* ``numpy`` (>=1.14)
* ``scipy`` (>=0.18.1)
* ``mne`` (>=0.21)
* ``nibabel`` (>=2.2, optional)
* ``pybv`` (>=0.4, optional)


We recommend the `Anaconda <https://www.anaconda.com/download/>`_ Python
distribution. We require that you use Python 3.6 or higher.
You may choose to install ``mne-bids``
`via pip <#Installation via pip>`_ or
`via conda <#Installation via conda>`_.

Installation via pip
--------------------

Besides ``numpy`` and ``scipy`` (which are included in the standard Anaconda
installation), you will need to install the most recent version of ``MNE``
using the ``pip`` tool:

.. code-block:: bash

   $ pip install -U mne


Then install ``mne-bids``\ :

.. code-block:: bash

   $ pip install -U mne-bids


These ``pip`` commands also work if you want to upgrade if a newer version of
``mne-bids`` is available. If you do not have administrator privileges on the
computer, use the ``--user`` flag with ``pip``.

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   $ python -c 'import mne_bids'

For full functionality of ``mne-bids``, you will also need to ``pip install``
the following packages:

- ``nibabel``, for interacting with MRI data
- ``pybv``, to convert EEG data to BrainVision if input format is not valid according to EEG BIDS specifications

If you want to use the latest development version of ``mne-bids``, use the
following command:

.. code-block:: bash

   $ pip install https://api.github.com/repos/mne-tools/mne-bids/zipball/master

Installation via conda
----------------------

If you have followed the
`MNE-Python installation instructions <https://mne.tools/stable/install_mne_python.html#installing-mne-python-and-its-dependencies>`_,
all that's left to do is to install ``mne-bids`` without its dependencies, as
they've already been installed during the ``MNE`` installation process.

Activate the correct ``conda`` environment and install ``mne-bids``:

.. code-block:: bash

   $ conda activate mne
   $ conda install --channel conda-forge --no-deps mne-bids

This approach ensures that the installation of ``mne-bids`` doesn't alter any
other packages in your existing ``conda`` environment.

Alternatively, you may wish to take advantage of the fact that the
``mne-bids`` package on ``conda-forge`` in fact depends on ``mne``,
meaning that a "full" installation of ``mne-bids`` (i.e., including its
dependencies) will provide you with a working copy of of both ``mne`` and
``mne-bids`` at once:

.. code-block:: bash

   $ conda create --name mne --channel conda-forge mne-bids

After activating the environment, you should be ready to use ``mne-bids``:

.. code-block:: bash

   $ conda activate mne
   $ python -c 'import mne_bids'
