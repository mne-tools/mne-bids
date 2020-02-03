Contributions
-------------

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both API docs and an example script),
it can be incorporated into the master branch.

To help developing `mne-bids`, you will need a few adjustments to your
installation as shown below.

##### Running tests

To run the tests using `pytest`, you need to have the git cloned mne-python
repository with a local pip-install instead of the mne-python package from
pypi. Update your installation as follows.

    $ git clone https://github.com/mne-tools/mne-python --depth 1
    $ cd mne-python
    $ pip uninstall mne  # uninstall pypi mne
    $ pip install -e .  # use the cloned repo for a local install of mne

The same needs to be done with mne-bids. However, here you need to make sure that you work on your [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) of the mne-bids repository.

    $ git clone https://github.com/<your-GitHub-username>/mne-bids
    $ cd mne-bids
    $ pip install -e .

Then, install the following python packages for testing purposes:

    $ pip install flake8 pytest pytest-cov

Afterwards, also install all optional dependencies of `mne-bids` and their
dependencies:

    $ pip install scikit-learn nibabel nilearn pybv

Finally, it is necessary to install the
[BIDS validator](https://github.com/bids-standard/bids-validator). The outputs
of MNE-BIDS are run through the BIDS validator to check if the conversion
worked.
You will need the `command line version` of the validator, to be installed via
[Node.js](https://nodejs.org/en/).

- For installing the *stable* version of the bids-validator, please follow the
instructions as detailed in the README of the bids-validator repository.
- For installing the *development* version, see [here](https://github.com/bids-standard/bids-validator/blob/master/CONTRIBUTING.md#using-the-development-version-of-bids-validator)

##### Building the documentation

The documentation can be built using sphinx. For that, please additionally
install the following:

    $ pip install matplotlib nilearn sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme pillow

To build the documentation locally, one can run:

    $ cd doc/
    $ make html

or

    $ make html-noplot
    
if you don't want to run the examples to build the documentation. This will result in a faster build but produce no plots in the examples.
