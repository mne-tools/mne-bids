[![Gitter](https://badges.gitter.im/mne-tools/mne-bids.svg)](https://gitter.im/mne-tools/mne-bids?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Travis](https://api.travis-ci.org/mne-tools/mne-bids.svg?branch=master "Travis")](https://travis-ci.org/mne-tools/mne-bids)
[![codecov](https://codecov.io/gh/mne-tools/mne-bids/branch/master/graph/badge.svg)](https://codecov.io/gh/mne-tools/mne-bids)
[![CircleCi](https://circleci.com/gh/mne-tools/mne-bids.svg?style=svg)](https://circleci.com/gh/mne-tools/mne-bids)

MNE-BIDS
========

This is a repository for creating BIDS-compatible datasets with MNE.

Installation
------------

We recommend the [Anaconda Python distribution](https://www.continuum.io/downloads).
To install ``mne_bids``, you first need to install its dependencies:

    $ pip install pandas mne

Then install mne_bids::

    $ pip install git+https://github.com/mne-tools/mne-bids.git#egg=mne-bids

If you do not have administrator privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do:

    $ python -c 'import mne_bids'

and it should not give any error messages.

Command Line Interface
----------------------

In addition to import mne_bids, you can use the command line interface.


Example :

```bash
$ mne_bids raw_to_bids --subject_id sub01 --task rest --raw_file data.edf --output_path new_path
```

Contributions
-------------

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both API docs and an example script),
it can be incorporated into the master branch.

To run the tests using `pytest`, you need to have the cloned mne-python repository
with a local pip-install instead of the mne-python package from pypi. Update your
installation as follows:

    $ git clone https://github.com/mne-tools/mne-python
    $ cd mne-python
    $ pip uninstall mne  # uninstall pypi mne
    $ pip install -e .  # use the cloned repo for a local install of mne

Note that, for testing purposes, it is necessary to install the
[BIDS validator](https://github.com/INCF/bids-validator). The outputs of
MNE-BIDS are run through the BIDS validator to check if the conversion worked.

Cite
----

If you use `mne-bids` in your work, please cite:

    Niso, G., Gorgolewski, K.J., Bock, E., Brooks, T.L., Flandin, G., Gramfort, A.,
    Henson, R.N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J., 
    Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data 
    structure extended to magnetoencephalography. Scientific Data, 5, 180110. 
    http://doi.org/10.1038/sdata.2018.110

