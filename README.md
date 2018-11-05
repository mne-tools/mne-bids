[![Gitter](https://badges.gitter.im/mne-tools/mne-bids.svg)](https://gitter.im/mne-tools/mne-bids?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Travis](https://api.travis-ci.org/mne-tools/mne-bids.svg?branch=master "Travis")](https://travis-ci.org/mne-tools/mne-bids)
[![Appveyor](https://ci.appveyor.com/api/projects/status/g6jqpv31sp7q103s/branch/master?svg=true "Appveyor")](https://ci.appveyor.com/project/mne-tools/mne-bids/branch/master)
[![codecov](https://codecov.io/gh/mne-tools/mne-bids/branch/master/graph/badge.svg)](https://codecov.io/gh/mne-tools/mne-bids)
[![CircleCi](https://circleci.com/gh/mne-tools/mne-bids.svg?style=svg)](https://circleci.com/gh/mne-tools/mne-bids)

MNE-BIDS
========

This is a repository for creating BIDS-compatible datasets with MNE.

Installation
------------

We recommend the
[Anaconda Python distribution](https://www.continuum.io/downloads).
Next to `numpy`, `scipy`, and `matplotlib` that are included in the standard
anaconda distribution, you will need to install the following dependencies
to be able to use `mne_bids`:

    $ pip install pandas
    $ pip install -U https://api.github.com/repos/mne-tools/mne-python/zipball/master

Then install `mne_bids`:

    $ pip install -U mne-bids

If you do not have administrator privileges on the computer, use the `--user`
flag with `pip`. To upgrade, use the `--upgrade` flag provided by `pip`.

To check if everything worked fine, you can do:

    $ python -c 'import mne_bids'

and it should not give any error messages.

Command Line Interface
----------------------

In addition to `import mne_bids`, you can use the command line interface.


Example :

```bash
$ mne_bids raw_to_bids --subject_id sub01 --task rest --raw_file data.edf --output_path new_path
```

Cite
----

If you use `mne-bids` in your work, please cite:

    Niso, G., Gorgolewski, K.J., Bock, E., Brooks, T.L., Flandin, G., Gramfort, A.,
    Henson, R.N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J.,
    Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data
    structure extended to magnetoencephalography. Scientific Data, 5, 180110.
    http://doi.org/10.1038/sdata.2018.110
