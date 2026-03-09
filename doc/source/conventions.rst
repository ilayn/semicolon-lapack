===================
Conventions
===================

This page covers the conventions used throughout semicolon-lapack — useful
if you haven't worked with LAPACK before.

Naming Convention
-----------------

semicolon-lapack follows the standard LAPACK naming scheme:

.. code-block:: text

   XYYZZZ
   ││ └── Operation
   ││
   │└──── Matrix type (e.g., ge = general, sy = symmetric)
   └───── Precision (s = single, d = double, c = complex single, z = complex double)

Examples:

- ``dgetrf`` — **d**\ouble precision, **ge**\neral matrix, **tr**\iangular **f**\actorization
- ``dsytf2`` — **d**\ouble precision, **sy**\mmetric matrix, **t**\riangular **f**\actorization, level **2**
- ``dlaswp`` — **d**\ouble precision, **la**\pack **s**\wap (row interchange)


Matrix Storage
--------------

All matrices use **column-major** (or Fortran-major) storage order:

.. math::

   A = \begin{pmatrix}
   a_{00} & a_{01} & a_{02} \\
   a_{10} & a_{11} & a_{12} \\
   a_{20} & a_{21} & a_{22}
   \end{pmatrix}

is stored in memory as:

.. code-block:: text

   [a_00, a_10, a_20, a_01, a_11, a_21, a_02, a_12, a_22]

unlike a C-major array;

.. code-block:: text

   [a_00, a_01, a_02, a_10, a_11, a_12, a_20, a_21, a_22]

Hence one models the column scanning, the other models the row scanning.


The element at row ``i``, column ``j`` of an ``m``-by-``n`` matrix with leading
dimension ``lda`` is accessed as:

.. code-block:: c

   A[i + j * lda]

where ``lda >= max(1, m)``.


.. note::
   The leading dimension variables ``ld..``, tell the routine how far apart
   consecutive columns are in memory. Since LAPACK routines only receive a
   pointer and dimensions, ``lda`` is the only way to know where the next
   column starts. This is what makes it possible to operate on arbitrary
   submatrices of a larger array without copying any data.


Indexing, Pivoting, and Permutations
------------------------------------

All indices are **0-based**, unlike Fortran LAPACK which uses 1-based indexing.

In a number of LAPACK routines, row or column swapping occurs during the
algorithm. These swaps are recorded in pivot arrays, where each entry says
which row was swapped at that step. This is different from a permutation
array that records the final ordering directly. The distinction can be
jarring at first. For example, the pivot array:

.. code-block:: c

   int ipiv[4] = {0, 3, 3, 1};

encodes the following sequence of swaps, with the resulting row order shown
after each step:

- ``ipiv[0] = 0``: row 0 swapped with row 0 (no-op) → [0, 1, 2, 3]
- ``ipiv[1] = 3``: row 1 swapped with row 3 → [0, 3, 2, 1]
- ``ipiv[2] = 3``: row 2 swapped with row 3 → [0, 3, 1, 2]
- ``ipiv[3] = 1``: row 3 swapped with row 1 → [0, 2, 1, 3]


Parameter Conventions
---------------------

Common parameter patterns:

``[in]``
   Input parameter, not modified by the function.

``[out]``
   Output parameter, value on entry is not used.

``[in,out]``
   Input/output parameter, value on entry is used and may be modified.

.. note ::
   In Fortran 77, every argument is passed by reference and subroutines are
   free to modify any of them. The original LAPACK API predates Fortran 90's
   ``INTENT`` attribute, so there is no way to tell from the interface alone
   which arguments are read-only. In this project, ``const`` qualifiers and
   ``@param`` direction tags try to make that information explicit for the
   user.


Error Handling
--------------

Functions modify an ``out`` integer pointer called ``info``. The value typically
denotes. These error codes are left **1-indexed** because ``0`` encodes success.

- ``info = 0``: Successful exit
- ``info < 0``: The ``(-info)``-th argument had an illegal value
- ``info > 0``: Algorithm-specific error (e.g., singular matrix)


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
