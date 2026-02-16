=============
Dense Details
=============

Requirements
------------

- A C11-compatible compiler (MSVC is not supported, because they refuse to comply with ``complex.h``)
- A CBLAS implementation (OpenBLAS, Intel MKL, Apple Accelerate, etc.)
- `Meson <https://mesonbuild.com/>`_ >= 1.1.0

For running the test suite:

- `CMocka <https://cmocka.org/>`_ >= 2.0

Building
--------

.. code-block:: bash

   meson setup builddir
   ninja -C builddir

Running Tests
^^^^^^^^^^^^^

.. note ::
   The following is a temporary grouping for testing purposes. It already shows
   some problems, hence probably will be removed or heavily modified in the near
   future.

.. code-block:: bash

   # Run all tests
   meson test -C builddir

   # Run tests by suite
   meson test -C builddir --suite lu         # LU factorization family
   meson test -C builddir --suite tridiag    # Tridiagonal systems
   meson test -C builddir --suite solve      # Iterative refinement / expert drivers

   # Run a single test
   meson test -C builddir dgetrf

   # Verbose output
   meson test -C builddir -v

Basic Usage
-----------

Include the header and link against your CBLAS library:

.. code-block:: c

   #include "semicolon_lapack/semicolon_lapack.h"

   int main() {
       double A[9] = {
           2.0, 1.0, 1.0,
           4.0, 3.0, 3.0,
           8.0, 7.0, 9.0
       };
       int ipiv[3];
       int info;

       // Compute LU factorization: A = P * L * U
       dgetrf(3, 3, A, 3, ipiv, &info);

       if (info == 0) {
           printf("LU factorization successful\n");
       }
       return 0;
   }

Differences from Fortran LAPACK
-------------------------------

1. **0-based indexing**: All input and output variables, pointers etc. are 0-indexed, e.g.,
   indices in ``ipiv`` are 0-based, not 1-based.

2. **Simplified ``ilaenv``**: Block sizes are determined from static lookup
   tables rather than the full ``ilaenv`` dispatch mechanism.

3. **Character parameters**: LAPACK's ``CHARACTER*1`` parameters are
   ``const char*`` pointers, not ``char`` by value. Call sites typically use
   string literals:

   .. code-block:: c

      dgetrs("N", n, nrhs, A, lda, ipiv, B, ldb, &info);

   Internally, only the first character is inspected (via ``param[0]``), so
   the pointer need not be a null-terminated string — any ``const char*``
   pointing to at least one character is valid.

4. **Column-major storage**: Matrix storage remains column-major, matching
   Fortran LAPACK and the CBLAS interfaces. Changing this is a major algorithmic
   rewrite which is a different type of monster.

5. **``const`` and ``restrict`` qualifiers**: Function signatures use ``const``
   for read-only parameters and ``restrict`` to express non-aliasing guarantees.
   These are enforced by Fortran's language rules but violations are silent.
   Making them explicit in C documents the contract, enables compiler
   optimizations that rely on non-aliasing, and allows compilers to warn
   when aliased pointers are passed to ``restrict``-qualified parameters.

6. **Pass scalars by value**: One of the more tedious aspects of calling
   LAPACK from C is that the Fortran ABI requires every argument to be
   passed by reference — including simple scalars like matrix dimensions
   and leading dimensions. This forces C callers into one of two
   unpleasant patterns:

   .. code-block:: c

      // Temporary variables just to take addresses
      int n = 100, nrhs = 1, lda = 100, ldb = 100, info;
      dgetrs_("N", &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);

      // Or compound literals, which hurt readability
      dgetrs_("N", &(int){100}, &(int){1}, A, &(int){100}, ipiv, B, &(int){100}, &(int){0});

   Since we control the API, we pass scalars by value wherever possible:

   .. code-block:: c

      // Clean(er), natural C calling convention
      dgetrs("N", 100, 1, A, 100, ipiv, B, 100, &info);

   Output parameters like ``info`` remain pointers since the callee
   must write back through them.


Documenting C Code
-------------------

semicolon-lapack uses Doxygen-style comments that are processed by Breathe
into Sphinx documentation.

Comment Style
^^^^^^^^^^^^^

Use Javadoc-style comments with ``@param``:

.. code-block:: c

   /**
    * Brief description (first sentence ends with period).
    *
    * Detailed description can span multiple paragraphs.
    *
    * @param[in]     m     Number of rows of matrix A (m >= 0)
    * @param[in]     n     Number of columns of matrix A (n >= 0)
    * @param[in,out] A     On entry, the M-by-N matrix A.
    *                      On exit, the factors L and U from A = P*L*U.
    * @param[in]     lda   Leading dimension of A (lda >= max(1,m))
    * @param[out]    ipiv  Pivot indices, dimension min(m,n). Row i was
    *                      interchanged with row ipiv[i].
    * @param[out]    info  = 0: successful exit
    *                      < 0: if -i, the i-th argument had an illegal value
    *                      > 0: if i, U(i,i) is exactly zero
    */
   void dgetrf(const int m, const int n, double* const restrict A,
               const int lda, int* const restrict ipiv, int* info);

Parameter Direction
^^^^^^^^^^^^^^^^^^^

Always specify parameter direction:

- ``@param[in]`` — Input only, not modified
- ``@param[out]`` — Output only, input value ignored
- ``@param[in,out]`` — Both input and output

Building Documentation
----------------------

The documentation pipeline uses Doxygen (C) and Sphinx (Python), connected by
the Breathe extension. You will need:

- `Doxygen <https://www.doxygen.nl/>`_ for parsing C source files
- Python 3 with the following packages:

  - `Sphinx <https://www.sphinx-doc.org/>`_ >= 7.2
  - `Breathe <https://breathe.readthedocs.io/>`_ >= 4.35 (Doxygen-to-Sphinx bridge)
  - `pydata-sphinx-theme <https://pydata-sphinx-theme.readthedocs.io/>`_ >= 0.15
  - `sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/>`_ >= 0.5

These are also listed in ``doc/requirements.txt`` for convenience:

.. code-block:: bash

   pip install -r doc/requirements.txt

(or ``conda``, ``uv`` or whichever tool you use)

Then build and serve locally:

.. code-block:: bash

   cd doc
   make html
   cd build/html && python -m http.server 8000

The output will be in ``doc/build/html/`` and served at
``http://localhost:8000``.

.. note ::
   If you prefer Doxygen's native HTML output (no judgement), you can run
   ``doxygen doc/source/Doxyfile`` directly and skip the Python toolchain
   entirely. We specifically avoid the caller diagrams and source browser
   that Doxygen generates. There is no technical reason, we just don't
   like them.
