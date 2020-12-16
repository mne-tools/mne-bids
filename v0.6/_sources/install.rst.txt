:orphan:

Installation
============

Dependencies
------------

* ``mne`` (>=0.21)
* ``numpy`` (>=1.14)
* ``scipy`` (>=0.18.1, or >=1.5.0 for certain operations with EEGLAB data)
* ``nibabel`` (>=2.2, optional, for processing MRI data)
* ``pybv`` (>=0.4, optional, for writing BrainVision data)
* ``pandas`` (>=0.23.4, optional, for generating event statistics)
* ``matplotlib`` (optional, for using the interactive data inspector)


We recommend the `Anaconda <https://www.anaconda.com/download/>`_ Python
distribution. We require that you use Python 3.6 or higher.
You may choose to install ``mne-bids``
`via pip <#Installation via pip>`_ or
`via conda <#Installation via conda>`_.

Installation via pip
--------------------

To install MNE-BIDS including all dependencies required to use all features,
simply run:

.. code-block:: bash

   pip install --user -U mne-bids[full]

This ``pip`` command will also work if you want to upgrade if a newer version
of ``mne-bids`` is available.

If you don't require advanced features like interactive visual data inspection,
you may also install a basic version of MNE-BIDS via

.. code-block:: bash

   pip install --user -U mne-bids

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/mne-tools/mne-bids/zipball/master

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   python -c 'import mne_bids'

MNE-BIDS works best with the latest stable release of MNE-Python. To ensure
MNE-Python is up-to-date, run:

.. code-block:: bash

   pip install --user -U mne

Installation via conda
----------------------

If you have followed the
`MNE-Python installation instructions <https://mne.tools/stable/install/mne_python.html#installing-mne-python>`_,
all that's left to do is to install ``mne-bids`` without its dependencies, as
they've already been installed during the ``MNE`` installation process.

Activate the correct ``conda`` environment and install ``mne-bids``:

.. code-block:: bash

   conda activate mne
   conda install --channel conda-forge --no-deps mne-bids

This approach ensures that the installation of ``mne-bids`` doesn't alter any
other packages in your existing ``conda`` environment.

Alternatively, you may wish to take advantage of the fact that the
``mne-bids`` package on ``conda-forge`` in fact depends on ``mne``,
meaning that a "full" installation of ``mne-bids`` (i.e., including its
dependencies) will provide you with a working copy of of both ``mne`` and
``mne-bids`` at once:

.. code-block:: bash

   conda create --name mne --channel conda-forge mne-bids

After activating the environment, you should be ready to use ``mne-bids``:

.. code-block:: bash

   conda activate mne
   python -c 'import mne_bids'
