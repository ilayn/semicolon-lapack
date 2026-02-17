========================
semicolon-lapack
========================

**semicolon-lapack** is a C implementation of LAPACK linear algebra routines,
designed to work with any CBLAS implementation while achieving comparable performance
to vendor LAPACK libraries.

The original LAPACK is written in Fortran 77, which creates friction for non-Fortran
projects. Instead, users must often deal with name mangling conventions, Fortran calling
conventions, 1-indexing assumptions, and linking against a Fortran runtime. A native C
implementation sidesteps many of these issues.

Most optimized BLAS/LAPACK libraries only optimize the BLAS layer in C and Assembly, while
the remaining LAPACK routines are left as Fortran and compiled as-is on top of this BLAS.
This still requires a Fortran compiler in the toolchain and limits LAPACK to platforms
where one is available.

Our goal is to provide a C implementation of LAPACK on top of any vendor's BLAS via the
CBLAS interface, removing the Fortran dependency entirely. This makes LAPACK portable to
any platform with a C compiler and opens the door to optimizing the LAPACK layer itself.

Key Features
------------

- Pure C implementation (no Fortran dependency).
- Uses standard CBLAS interface for vendor-agnostic BLAS calls.
- Column-major storage.
- 0-based indexing throughout both input and output parameters.
- Static struct lookup for optimal block sizes, replacing ``ilaenv`` calls.

.. note::

   This project is under active development. Currently, we are exploring what is
   possible both in terms of code and documentation.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Overview

   building
   dense-details
   conventions

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
