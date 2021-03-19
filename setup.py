"""Setup MNE-BIDS."""
import sys

from setuptools import setup

# Give setuptools a hint to complain if it's too old a version
SETUP_REQUIRES = ["setuptools >= 46.4.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []

if __name__ == "__main__":
    setup(
        setup_requires=SETUP_REQUIRES,
    )
