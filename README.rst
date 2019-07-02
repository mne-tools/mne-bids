

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


MNE-BIDS
========

This is a repository for creating
`BIDS <http://bids.neuroimaging.io/>`_\ -compatible datasets with
`MNE <https://mne-tools.github.io/stable/index.html>`_.

Installation
------------

We recommend the `Anaconda <https://www.anaconda.com/download/>`_ Python
distribution. We require that you use Python 3.
Besides ``numpy`` and ``scipy`` (which are included in the standard Anaconda
installation), you will need to install the most recent version of ``MNE ``
using using the ``pip`` tool:

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

For full functionality of ``mne-bids``, you will also need to install the
following packages:

- nibabel
- nilearn
- matplotlib

If you want to use the latest development version, use the following command:

.. code-block:: bash

   $ pip install https://api.github.com/repos/mne-tools/mne-bids/zipball/master

Quickstart
----------

Currently, we support writing of BIDS datasets for MEG and EEG. Support for
iEEG is experimental at the moment.

.. code:: python

    >>> from mne import io
    >>> from mne_bids import write_raw_bids
    >>> raw = io.read_raw_fif('my_old_file.fif')
    >>> write_raw_bids(raw, 'sub-01_ses-01_run-05', output_path='./bids_dataset')

Command Line Interface
----------------------

In addition to ``import mne_bids``\ , you can use the command line interface.

Example:

.. code-block:: bash

   $ mne_bids raw_to_bids --subject_id sub01 --task rest --raw data.edf --output_path new_path

Bug reports
-----------

Use the `github issue tracker <https://github.com/mne-tools/mne-bids/issues>`_
to report bugs.

Cite
----

If you use ``mne-bids`` in your work, please cite one of the following papers,
depending on which modality you used:

MEG
###

.. code-block:: Text

   Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A.,
   Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J.,
   Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data
   structure extended to magnetoencephalography. Scientific Data, 5, 180110.
   http://doi.org/10.1038/sdata.2018.110


EEG
###

.. code-block:: Text

   Pernet, C. R., Appelhoff, S., Flandin, G., Phillips, C., Delorme, A., &
   Oostenveld, R. (2018, December 6). BIDS-EEG: an extension to the Brain
   Imaging Data Structure  (BIDS) Specification for electroencephalography.
   https://doi.org/10.31234/osf.io/63a4y


iEEG
####

.. code-block:: Text

   Holdgraf, C., Appelhoff, S., Bickel, S., Bouchard, K., D'Ambrosio, S.,
   David, O., … Hermes, D. (2018, December 13). BIDS-iEEG: an extension to the
   brain imaging data structure  (BIDS) specification for human intracranial
   electrophysiology. https://doi.org/10.31234/osf.io/r7vc2
