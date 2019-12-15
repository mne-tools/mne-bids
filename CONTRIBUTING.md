Contributions
-------------

`mne-bids` development takes place
[on Github](https://github.com/mne-tools/mne-bids).
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

##### Updating the conda package

The `conda-forge` channel includes a `conda` package of `mne-bids`.
A dedicated [git repository](https://github.com/conda-forge/mne-bids-feedstock),
which is called a `feedstock` in `conda-forge` terms, contains the
recipe describing all steps required to build and test the package; it can be
found at [recipe/meta.yaml](https://github.com/conda-forge/mne-bids-feedstock/blob/master/recipe/meta.yaml).

Whenever a new version of `mne-bids` is published on PyPI, a `conda-forge` bot
will automatically open a Pull Request (PR) to update the following parts of
`meta.yaml`:

- version number, e.g. `{% set version = "0.3" %}`

- SHA256 hash of the source archive, e.g. `sha256: 187add419562e3607c161eaf04971863c134febad2e70313260a58238e519ba5`

- build number is reset to zero: `number: 0`

Before merging, please check whether the dependencies of `mne-bids` have
changed. If that's the case, you have two options.

###### Updating dependencies via GitHub browser interface

This is the most straightforward option:

- using your web browser, navigate to the bot's branch of `mne-bids-feedstock`
  it wants to merge (you can simply follow the link on top of the respective
  PR page)
- navigate to `recipe/meta.yaml`, and click on the edit button
- adjust the package dependencies in the `requirements: run` section
- commit the changes directly to the bot's branch (all recipe maintainers have
  write access)
- wait for CI to successfully finish and merge the PR

###### Updating dependencies locally

If you prefer not to use the web interface:

- fork the bot's fork of the `mne-bids-feedstock` repository (or add it as a
  remote)
- checkout the respective branch (it's typically called `vXY`, where `XY` is
  the version number, e.g., `v0.3`)
- adjust the package dependencies in the `requirements: run` section of
  `recipe/meta.yaml`
- commit and push the changes to the bot's branch
- wait for CI to successfully finish and merge the PR

After the changes have been merged and the package is built, it may take up
to 60 minutes to propagate across all `conda-forge` mirrors.

###### Other changes to the recipe

Of course, changes to the recipe are not bound to a new upstream release.
You may change any part of the recipe at any time. 

It is important to always work on a branch of your fork of `mne-bids-feedstock`.
Even if you are a recipe maintainer (which provides you with write access
to the `mne-bids-feedstock` repository), _do not_ create branches on that
repository directly (or, even worse, commit to its `master` branch).

- increase the `build` number
- open a a PR
- rerender the feedstock by posting `@conda-forge-admin, please rerender`
  as a comment to the PR
- merge the PR once all tests have passed

Again, it may take up to 60 minutes for the new package to appear on all
mirrors.

See the
[conda-forge documentation](https://conda-forge.org/docs/maintainer/updating_pkgs.html)
for additional information.
