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

For most users, ``mkl-dynamic-lp64-seq`` is the right choice. For ILP64
builds, use an ``ilp64`` variant (see :ref:`ilp64-builds`).

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
   :widths: 25 15 15 45

   * - Option
     - Type
     - Default
     - Description
   * - ``blas``
     - string
     - ``auto``
     - BLAS vendor or pkg-config package name
   * - ``USE_INT64``
     - boolean
     - ``false``
     - Build with 64-bit integers (ILP64). See :ref:`ilp64-builds`.
   * - ``SYMBOL_MANGLING``
     - string
     - ``name##_64``
     - ILP64 symbol naming pattern. See :ref:`ilp64-symbol-naming`.
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


.. _ilp64-builds:

ILP64 Builds (64-bit Integers)
------------------------------

Why ILP64
^^^^^^^^^

Some applications need indices larger than 2**31 (>2 billion) elements. Standard
32-bit integers overflow at dimension 46340 for dense square matrices. ILP64
uses 64-bit integers for all LAPACK integer parameters, matching an ILP64 BLAS
library underneath.

The integer width of the library **must** match the linked BLAS. If the widths
do not match, every CBLAS call silently corrupts memory. There are no compiler
warnings and no linker errors for this mismatch on value parameters.

To catch this, ``meson setup`` runs a configure-time probe that detects the
actual integer width of the linked BLAS library. If the result contradicts
``USE_INT64``, the build fails immediately with a clear error message instead
of producing a silently broken library. The technique is adapted from
`libblastrampoline <https://github.com/JuliaLinearAlgebra/libblastrampoline>`_
(Julia's BLAS dispatch library): it calls ``cblas_idamax`` with a crafted ``n``
value whose lower 32 bits equal 3 and whose full 64 bits are negative.  An LP64
library reads only the lower 32 bits and returns normally; an ILP64 library
reads the full value and returns early.

The probe is skipped during cross-compilation since the compiled binary cannot
run on the build host. In that case a warning is emitted and the user is
responsible for ensuring consistency.

Building
^^^^^^^^

.. code-block:: bash

   # With OpenBLAS ILP64:
   meson setup builddir_ilp64 -DUSE_INT64=true -Dblas=openblas

   # With MKL ILP64:
   PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig \
       meson setup builddir_ilp64 -DUSE_INT64=true -Dblas=mkl-dynamic-ilp64-seq

   ninja -C builddir_ilp64
   meson install -C builddir_ilp64


.. _ilp64-symbol-naming:

Symbol Naming
^^^^^^^^^^^^^

ILP64 symbols carry a suffix (or prefix, or both) to distinguish them from LP64
symbols. The default is ``_64``, following the convention used by Reference
LAPACK, Julia, NumPy, and SciPy.

This is typically the most annoying part of writing vendor agnostic code that
tries to support arbitrary BLAS/LAPACK vendors. The SYMBOL_MANGLING option
is provided to alleviate this pain by allowing users to specify whatever
symbol naming convention they want.


The ``SYMBOL_MANGLING`` option uses C token-pasting syntax. The literal ``name``
is the placeholder for the base function name:

.. list-table::
   :header-rows: 1
   :widths: 45 25

   * - Meson option
     - ``dgetrf`` becomes
   * - (default) ``name##_64``
     - ``dgetrf_64``
   * - ``name##_ilp64``
     - ``dgetrf_ilp64``
   * - ``ilp_##name``
     - ``ilp_dgetrf``
   * - ``LAPACKE_##name##_64``
     - ``LAPACKE_dgetrf_64``

Example:

.. code-block:: bash

   meson setup builddir -DUSE_INT64=true -DSYMBOL_MANGLING='scipy_##name##_64'

This option has no effect for LP64 builds.


Shared Libraries
----------------

.. code-block:: bash

   meson setup builddir --default-library=shared

For details on symbol visibility and the ``SEMICOLON_API`` macro, see
:doc:`installing`.


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
