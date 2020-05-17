# Contributions

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both API docs and an example script),
it can be incorporated into the master branch.

To help developing `mne-bids`, you will need a few adjustments to your
installation as shown below.

## Running tests

### (Optional) Install development version of MNE-Python
If you want to run the tests with a development version of MNE-Python,
you can install it by running

    $ pip install -U https://github.com/mne-tools/mne-python/archive/master.zip

### Install development version of MNE-BIDS 
First, you should [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the `mne-bids` repository. Then, clone the fork and install it in
"editable" mode.

    $ git clone https://github.com/<your-GitHub-username>/mne-bids
    $ pip install -e ./mne-bids


### Install Python packages required to run tests
Install the following packages for testing purposes, plus all optonal MNE-BIDS
dependencies to ensure you will be able to run all tests.

    $ pip install flake8 pytest pytest-cov scikit-learn nibabel nilearn pybv

### Install the BIDS validator
Finally, it is necessary to install the
[BIDS validator](https://github.com/bids-standard/bids-validator). The outputs
of MNE-BIDS are run through the BIDS validator to check if the conversion
worked properly and produced datasets that conforms to the BIDS specifications.

You will need the `command line version` of the validator.

#### Global (system-wide) installation
- First, install [Node.js](https://nodejs.org/en/).
- For installing the **stable** version of `bids-validator`, please follow the
instructions as detailed in the README of the bids-validator repository.
- For installing the **development** version of `bids-validator`, see [here](https://github.com/bids-standard/bids-validator/blob/master/CONTRIBUTING.md#using-the-development-version-of-bids-validator).

Test your installation by running:

    $ bids-validator --version

#### Local (per-user) development installation

Install [Node.js](https://nodejs.org/en/). If you're use `conda`, you can
install the `nodejs` package from `conda-forge` by running
`conda install -c conda-forge nodejs`.

Then, retrieve the validator and install all its dependencies via `npm`.

    $ git clone git@github.com:bids-standard bids-validator.git
    $ cd bids-validator/bids-validator
    $ npm i

Test your installation by running:

    $ ./bin/bids-validator --version


### Invoke pytest
Now you can finally run the tests by running `pytest` in the
`mne-bids` directory.

    $ cd mne-bids
    $ pytest

If you have installed the `bids-validator`
on a per-user basis, set the environment variable `VALIDATOR_EXECUTABLE` to point to the path of the `bids-validator` before invoking `pytest`:

    $ VALIDATOR_EXECUTABLE=../bids-validator/bids-validator/bin/bids-validator pytest

## Building the documentation

The documentation can be built using sphinx. For that, please additionally
install the following:

    $ pip install matplotlib nilearn sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme pillow

To build the documentation locally, one can run:

    $ cd doc/
    $ make html

or

    $ make html-noplot
    
if you don't want to run the examples to build the documentation. This will result in a faster build but produce no plots in the examples.
