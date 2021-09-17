#!/bin/bash -ef

pushd ..
# Get npm 7+
npm install -g npm
git clone --depth 1 https://github.com/bids-standard/bids-validator
cd bids-validator
# Generate the full development node_modules
npm install
# Build & bundle the bids-validator CLI package
npm -w bids-validator run build
# Generate a package to install globally
npm -w bids-validator pack
# Install the package globally (different path expansion for Windows)
if [ "${THIS_OS}" == 'windows-latest' ]; then
  npm install -g | dir /b bids-validator-*.tgz
else
  npm install -g bids-validator-*.tgz
fi
popd