

.. image:: https://badges.gitter.im/mne-tools/mne-bids.svg
   :target: https://gitter.im/mne-tools/mne-bids?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
   :alt: Gitter

.. image:: https://travis-ci.org/mne-tools/mne-bids.svg?branch=master
   :target: https://travis-ci.org/mne-tools/mne-bids
   :alt: Travis

.. image:: https://ci.appveyor.com/api/projects/status/d4u70pht341cwqxb/branch/master?svg=true
   :target: https://ci.appveyor.com/project/mne-tools/mne-bids/branch/master
   :alt: Appveyor


.. image:: https://codecov.io/gh/mne-tools/mne-bids/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mne-tools/mne-bids
   :alt: codecov


.. image:: https://circleci.com/gh/mne-tools/mne-bids.svg?style=svg
   :target: https://circleci.com/gh/mne-tools/mne-bids
   :alt: CircleCi


.. image:: https://pepy.tech/badge/mne-bids
   :target: https://pepy.tech/project/mne-bids
   :alt: Downloads

.. image:: https://img.shields.io/pypi/v/mne-bids.svg
   :target: https://pypi.org/project/mne-bids/
   :alt: Latest PyPI release

.. image:: https://img.shields.io/conda/vn/conda-forge/mne-bids.svg
   :target: https://anaconda.org/conda-forge/mne-bids/
   :alt: Latest conda-forge release

.. image:: https://joss.theoj.org/papers/5b9024503f7bea324d5e738a12b0a108/status.svg
  :target: https://joss.theoj.org/papers/5b9024503f7bea324d5e738a12b0a108
  :alt: JOSS publication

MNE-BIDS
========

This is a repository for creating
`BIDS <https://bids.neuroimaging.io/>`_\ -compatible datasets with
`MNE-Python <https://mne.tools/stable/index.html>`_.

BIDS (Brain Imaging Data Structure) is a standard to organize data
according to a set of rules that describe:

- how to name your files
- where to place your files within a directory structure
- what additional metadata to store, and how to store it in sidecar json and tsv files

The complete set of rules is written down in the
`BIDS specification <https://bids-specification.readthedocs.io/en/stable/>`_.
A BIDS-compatible dataset conforms to these rules and passes the
`BIDS-validator <https://github.com/bids-standard/bids-validator>`_.

MNE-Python is a software package for analyzing neurophysiology data.

**MNE-BIDS links BIDS and MNE with the goal to make your analyses faster to code,
more robust to errors, and easily shareable with colleagues.**

Documentation
-------------

The documentation can be found under the following links:

- for the `stable release <https://mne.tools/mne-bids/>`_
- for the `latest (development) version <https://mne.tools/mne-bids/dev/index.html>`_

Dependencies
------------

* numpy (>=1.14)
* scipy (>=0.18.1)
* mne (>=0.19.1)
* nibabel (>=2.2, optional)
* pybv (optional)

Installation
------------

We recommend the `Anaconda <https://www.anaconda.com/download/>`_ Python
distribution. We require that you use Python 3.5 or higher.
You may choose to install ``mne-bids``
`via pip <#Installation via pip>`_ or
`via conda <#Installation via conda>`_.

Installation via pip
####################

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
######################

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


Quickstart
----------

Currently, we support writing of BIDS datasets for MEG and EEG. Support for
iEEG is experimental at the moment.

.. code:: python

    >>> from mne import io
    >>> from mne_bids import write_raw_bids
    >>> raw = io.read_raw_fif('my_old_file.fif')
    >>> write_raw_bids(raw, 'sub-01_ses-01_run-05', bids_root='./bids_dataset')

Command Line Interface
----------------------

In addition to ``import mne_bids``, you can use the command line interface.
Simply type ``mne_bids`` in your command line and press enter, to see the
accepted commands. Then type ``mne_bids <command> --help``, where ``<command>``
is one of the accepted commands, to get more information about that
``<command>``.

Example:

.. code-block:: bash

  $ mne_bids raw_to_bids --subject_id sub01 --task rest --raw data.edf --bids_root new_path

Bug reports
-----------

Use the `GitHub issue tracker <https://github.com/mne-tools/mne-bids/issues>`_
to report bugs.

Contributing
------------

Please see our `contributing guide <https://github.com/mne-tools/mne-bids/blob/master/CONTRIBUTING.md>`_.

Cite
----

If you use ``mne-bids`` in your work, please cite:

.. code-block:: Text

    Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C.,
    Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C.,
    Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing
    electrophysiological data into the BIDS format and facilitating their analysis.
    Journal of Open Source Software 4: (1896).

and one of the following papers, depending on which modality you used:

`MEG <http://doi.org/10.1038/sdata.2018.110>`_
##############################################

.. code-block:: Text

   Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A.,
   Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J.,
   Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data
   structure extended to magnetoencephalography. Scientific Data, 5, 180110.
   http://doi.org/10.1038/sdata.2018.110


`EEG <https://doi.org/10.1038/s41597-019-0104-8>`_
##################################################

.. code-block:: Text

   Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G.,
   Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an extension
   to the brain imaging data structure for electroencephalography. Scientific
   Data, 6, 103. https://doi.org/10.1038/s41597-019-0104-8


`iEEG <https://doi.org/10.1038/s41597-019-0105-7>`_
###################################################

.. code-block:: Text

   Holdgraf, C., Appelhoff, S., Bickel, S., Bouchard, K., D'Ambrosio, S.,
   David, O., … Hermes, D. (2019). iEEG-BIDS, extending the Brain Imaging Data
   Structure specification to human intracranial electrophysiology. Scientific
   Data, 6, 102. https://doi.org/10.1038/s41597-019-0105-7
