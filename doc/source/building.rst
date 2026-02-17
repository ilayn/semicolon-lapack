===================
Building
===================

Requirements
------------

- A C11 compiler (GCC, Clang, or llvm-mingw)
- Meson >= 1.1.0
- A CBLAS implementation (OpenBLAS, BLIS, MKL, or any library that provides ``cblas.h``)

Basic Build
-----------

.. code-block:: bash

   meson setup builddir
   ninja -C builddir

By default, Meson auto-detects the BLAS library via pkg-config, trying OpenBLAS
first and then a bunch of standard options.


Selecting a BLAS Vendor
-----------------------

The ``-Dblas`` option controls which CBLAS implementation to link against.

**Named vendors:**

.. code-block:: bash

   meson setup builddir -Dblas=auto       # default: try openblas, blis, cblas, blas
   meson setup builddir -Dblas=openblas   # OpenBLAS
   meson setup builddir -Dblas=blis       # BLIS

**Direct pkg-config package name:**

Any value that isn't one of the named vendors above is passed directly to
pkg-config as a package name. This is useful when your BLAS library ships a
``.pc`` file with a non-standard name, when you have multiple versions
installed and want to point at a specific one, or when the vendor has many
configuration variants (like Intel MKL).

.. code-block:: bash

   # Intel MKL (sequential, LP64):
   meson setup builddir -Dblas=mkl-dynamic-lp64-seq

   # Intel MKL (GNU OpenMP threading, LP64):
   meson setup builddir -Dblas=mkl-dynamic-lp64-gomp

   # Custom BLAS at a non-standard location:
   PKG_CONFIG_PATH=/opt/blas/lib/pkgconfig \
       meson setup builddir -Dblas=my-blas

Meson resolves the package name through pkg-config, which appends ``.pc``
automatically. The ``PKG_CONFIG_PATH`` environment variable tells pkg-config
where to find ``.pc`` files that aren't in the default search path.


Using Intel MKL
^^^^^^^^^^^^^^^

MKL ships 16 different pkg-config files, one for each combination of:

- **Integer size**: ``lp64`` (32-bit ``int``, standard) or ``ilp64`` (64-bit ``int``)
- **Threading**: ``seq`` (sequential), ``gomp`` (GNU OpenMP), ``iomp`` (Intel OpenMP), ``tbb`` (Threading Building Blocks)
- **Linking**: ``dynamic`` (shared libraries) or ``static``

The naming pattern is ``mkl-{linking}-{integer}-{threading}``. For example:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Package name
     - Configuration
   * - ``mkl-dynamic-lp64-seq``
     - Shared, 32-bit int, single-threaded
   * - ``mkl-dynamic-lp64-gomp``
     - Shared, 32-bit int, GNU OpenMP
   * - ``mkl-static-lp64-seq``
     - Static, 32-bit int, single-threaded
   * - ``mkl-dynamic-ilp64-seq``
     - Shared, 64-bit int, single-threaded

For most users, ``mkl-dynamic-lp64-seq`` is the right choice (this project
uses 32-bit ``int`` for LAPACK indices).

If MKL is installed via conda, you need to point pkg-config at the conda
environment's ``lib/pkgconfig`` directory:

.. code-block:: bash

   # Replace <env> with your conda environment name
   PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig \
       meson setup builddir -Dblas=mkl-dynamic-lp64-seq

You can verify which ``.pc`` files are available:

.. code-block:: bash

   ls $CONDA_PREFIX/lib/pkgconfig/mkl*.pc


Build Options
-------------

All options are passed to ``meson setup`` with the ``-D`` prefix:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``blas``
     - string
     - ``auto``
     - BLAS vendor or pkg-config package name
   * - ``tests``
     - boolean
     - ``true``
     - Build the test suite (requires CMocka >= 2.0)
   * - ``benchmarks``
     - boolean
     - ``false``
     - Build benchmark executables for profiling
   * - ``default_library``
     - string
     - ``static``
     - ``static``, ``shared``, or ``both``
   * - ``buildtype``
     - string
     - ``release``
     - ``release``, ``debug``, ``debugoptimized``


Installing
----------

.. code-block:: bash

   meson install -C builddir

This installs:

- ``libsemilapack.a`` (or ``.so``) to ``<prefix>/lib64/``
- All public headers to ``<prefix>/include/semicolon_lapack/``
- A pkg-config file to ``<prefix>/lib64/pkgconfig/semicolon-lapack.pc``

To install to a custom location, set ``--prefix`` at configure time:

.. code-block:: bash

   meson setup builddir --prefix=/opt/semicolon-lapack
   ninja -C builddir
   meson install -C builddir


Using the Installed Library
---------------------------

A single include gives access to all routines and precisions:

.. code-block:: c

   #include <semicolon_lapack/semicolon_lapack.h>

   int main(void) {
       double A[] = {2.0, 1.0, 1.0, 3.0};
       int ipiv[2], info;
       dgetrf(2, 2, A, 2, ipiv, &info);
       return info;
   }

Compile and link using pkg-config:

.. code-block:: bash

   gcc myprogram.c $(pkg-config --cflags --libs semicolon-lapack) -o myprogram

The pkg-config file automatically propagates the BLAS dependency, so you do not
need to add ``-lopenblas`` or ``-lblis`` manually.

If the library was installed to a non-standard prefix:

.. code-block:: bash

   PKG_CONFIG_PATH=/opt/semicolon-lapack/lib64/pkgconfig \
       gcc myprogram.c $(pkg-config --cflags --libs semicolon-lapack) -o myprogram


Reconfiguring
-------------

To change options on an existing build directory:

.. code-block:: bash

   meson setup builddir --reconfigure -Dbenchmarks=true

**Switching BLAS vendors requires a full wipe.** A simple ``--reconfigure`` will
update the dependency but leave stale object files and runtime paths from the
previous vendor, which can cause the wrong library to be loaded at runtime.
Always use ``--wipe`` when changing the BLAS vendor:

.. code-block:: bash

   meson setup builddir --wipe -Dblas=blis
