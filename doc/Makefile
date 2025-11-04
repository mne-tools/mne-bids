# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -nWT --keep-going
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = mne_bids
PAPER         =
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile view

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(O)

# Build without executing the example gallery (faster)
html-noplot:
	@$(SPHINXBUILD) $(SPHINXOPTS) -D plot_gallery=0 $(SOURCEDIR) $(BUILDDIR)

# View the built documentation
# If we build the example gallery, the index file is at _build/html/index.html
# If we don't (e.g. html-noplot), then it is at _build/index.html
view:
	@if [ -f "$(BUILDDIR)/html/index.html" ]; then \
		python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/$(BUILDDIR)/html/index.html')"; \
	elif [ -f "$(BUILDDIR)/index.html" ]; then \
		python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/$(BUILDDIR)/index.html')"; \
	else echo "No index.html file found to open. Did you build the docs?"; \
	fi

clean:
	-rm -rf _build auto_examples generated
