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

Then, install the following python packages:

    $ pip install flake8 pytest pytest-cov coverage

Finally, it is necessary to install the
[BIDS validator](https://github.com/bids-standard/bids-validator). The outputs
of MNE-BIDS are run through the BIDS validator to check if the conversion
worked.

##### Building the documentation

The documentation can be built using sphinx. For that, please additionally
install the following:

    $ pip install sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme pillow
