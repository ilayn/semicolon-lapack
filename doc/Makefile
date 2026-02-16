# Makefile for semicolon-lapack documentation

SPHINXBUILD   = sphinx-build
SPHINXOPTS    =
SOURCEDIR     = source
BUILDDIR      = build
DOXYGEN       = doxygen

# Default target
help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  html       Build HTML documentation"
	@echo "  clean      Remove all build artifacts"
	@echo "  rst        Generate API RST files from source tree"
	@echo "  doxygen    Generate Doxygen XML (usually called automatically)"
	@echo "  livehtml   Start live-reloading server (requires sphinx-autobuild)"

# Generate API RST files from source tree
rst:
	@python ../scripts/gen_docs.py
	@echo "RST files generated in $(SOURCEDIR)/api/"

# Build: rst -> doxygen -> sphinx
html: rst doxygen
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Generate Doxygen XML for Breathe
doxygen:
	@mkdir -p $(BUILDDIR)/doxygen
	@cd $(SOURCEDIR) && $(DOXYGEN) Doxyfile
	@echo "Doxygen XML generated in $(BUILDDIR)/doxygen/xml"

# Clean all build artifacts
clean:
	rm -rf $(BUILDDIR)

# Live-reloading server for development
livehtml: doxygen
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) \
		--watch ../*.c --watch ../*.h

.PHONY: help html rst doxygen clean livehtml
