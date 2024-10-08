# Contributions

Contributions are welcome in the form of feedback and discussion in issues,
or pull requests for changes to the code.

Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both in the API docs and with an example script),
it can be incorporated into the `main` branch.

To help developing `mne-bids`, you will need a few adjustments to your
installation as shown below.

Before submitting a pull request, we recommend that you run all style checks
and the test suite, and build the documentation **locally** on your machine.
That way, you can fix errors with your changes before submitting something
for review.

**All contributions are expected to follow the**
[**Code of Conduct of the mne-tools GitHub organization.**](https://github.com/mne-tools/.github/blob/master/CODE_OF_CONDUCT.md)

## Setting up a development environment

To start with, you should install `mne-bids` as described in our
[installation documentation](https://mne.tools/mne-bids/dev/install.html).
For a development environment we recommend that you perform the installation in
a dedicated Python environment,
for example using `conda` (see: https://docs.conda.io/en/latest/miniconda.html).
Afterwards, a few additional steps need to be performed.

**For all of the steps below we assume that you work in your dedicated `mne-bids` Python environment.**

### Clone MNE-Python and install it from the git repository

Use `git clone` to obtain the MNE-Python repository from https://github.com/mne-tools/mne-python/,
then navigate to the cloned repository using the `cd` command.

Then from the `mne-python` root directory call:

```Shell
pip uninstall mne --yes
pip install -e .
```

This will uninstall the current MNE-Python installation and instead install the MNE-Python
development version, including access to several internal test files that are needed
for `mne-bids` as well.

### Install the development version of MNE-BIDS

Now [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the `mne-bids` repository.
Then, `git clone` your fork and install it in "editable" mode.

```Shell
git clone https://github.com/<your-GitHub-username>/mne-bids
cd ./mne-bids
pip install -e ".[dev]"
git config --local blame.ignoreRevsFile .git-blame-ignore-revs
```

The last command is needed for `git diff` to work properly.
You should now have both the `mne` and `mne-bids` development versions available in your Python environment.

### Install the BIDS validator

For a complete development setup of `mne-bids`, it is necessary to install the
[BIDS validator](https://github.com/bids-standard/bids-validator).
The outputs of `mne-bids` are run through the BIDS validator to check if the conversion
worked properly and produced datasets conform to the
[BIDS specification](https://bids-specification.readthedocs.io/en/stable/).

To install the **stable** version of `bids-validator`, please follow the instructions
in the [bids-validator README](https://github.com/bids-standard/bids-validator#quickstart).
You will need the "command line version" of the bids-validator.

To install the **development** version of `bids-validator`, see the
[contributor documentation of bids-validator](https://github.com/bids-standard/bids-validator/blob/master/CONTRIBUTING.md#using-the-development-version-of-bids-validator).

In brief, the most convenient installation can be done using `conda`.
In your `conda` development environment for `mne-bids` simply call:

```Shell
conda install -c conda-forge nodejs
npm install --global npm@^7
npm install --global bids-validator
```

This would install the **stable** `bids-validator` and make it globally available
on your system.

Check your installation by running:

```Shell
bids-validator --version
```

### Install GNU Make

We use [GNU Make](https://www.gnu.org/software/make/) for developing `mne-bids`.
We recommend that you install GNU Make and make use of our `Makefile` at the root
of the repository.
For most Linux and macOS operating systems, GNU Make will be already installed by default.
Windows users can download the [Chocolatey package manager](https://chocolatey.org/)
and install [GNU Make from their repository](https://community.chocolatey.org/packages/make).

If for some reason you can't install GNU Make,
it might suffice to inspect the `Makefile` and to
figure out how to run the commands without invoking `make`.

## Making style checks

We run several style checks on `mne-bids`.
If you have accurately followed the steps to setup your `mne-bids` development version,
you can simply use the following command from the root of the `mne-bids` repository:

```Shell
make pep
```

We use [ruff](https://docs.astral.sh/ruff/) to format our code.
You can simply call `make ruff-format` from the root of the `mne-bids` repository
to automatically convert your code to follow the appropriate style.

### git pre-commit hooks

We suggest installing our git pre-commit hooks to automatically run style
checks before a commit is created:

```Shell
pre-commit install
```

You can also manually run the hooks via

```Shell
pre-commit run -a
```

## Running tests

We run tests using `pytest`.

First you will need to download the MNE-Python testing data.
Use the following command:

```Shell
python -c 'import mne; mne.datasets.testing.data_path(verbose=True)'
```

If you have accurately followed the steps to setup your `mne-bids` development version,
you can then simply use the following command from the root of the `mne-bids` repository:

```Shell
make test
```

## Building the documentation

The documentation can be built using [Sphinx](https://www.sphinx-doc.org).
If you have accurately followed the steps to setup your `mne-bids` development version,
you can simply use the following command from the root of the `mne-bids` repository:

```Shell
make build-doc
```

or, if you don't want to run the examples to build the documentation:

```Shell
make -C doc/ html-noplot
```

The latter command will result in a faster build but produce no plots in the examples.

More information on our documentation setup can be found in our
[mne-bids WIKI](https://github.com/mne-tools/mne-bids/wiki).

## A note on MNE-Python compatibility

We aim to make `mne-bids` compatible with the current stable release of `mne`,  as well as the previous stable release.
Both of these versions are tested in our continuous integration infrastructure.
Additional versions of `mne` may be compatible, but this is not guaranteed.

## Instructions for first-time contributors

When you are making your first contribution to `mne-bids`, we kindly request you to:

1. Create an account at [circleci](https://circleci.com/), because this will simplify running the automated test suite
   of `mne-bids`.
   Note: you can simply use your GitHub account to log in.
1. Add yourself to the [list of authors](https://github.com/mne-tools/mne-bids/blob/main/doc/authors.rst).
1. Add yourself to the [CITATION.cff](https://github.com/mne-tools/mne-bids/blob/main/CITATION.cff) file.
   Note: please add yourself in the
   ["authors" section](https://github.com/mne-tools/mne-bids/blob/fff6e90984ea0aa1e2914bb55e4197f7ec2800bf/CITATION.cff#L7C3-L7C3)
   of that file, towards the end of the list of authors, but **before** `Alexandre Gramfort` and `Mainak Jas`.
1. Update the [changelog](https://github.com/mne-tools/mne-bids/blob/main/doc/whats_new.rst) with your contribution.
   Note: please follow the existing format and add your name as a list item under the heading saying:
   `The following authors contributed for the first time. Thank you so much!`.

## Making a release

Usually only core developers make a release after consensus has been reached.
There is dedicated documentation for this in our
[mne-bids WIKI](https://github.com/mne-tools/mne-bids/wiki).
