==========
Installing
==========

This page covers installing semicolon-lapack, the installed file layout, and
using the library from your own project.

For building from source (configure, compile, run tests), see :doc:`building`.
For ILP64 build options (``USE_INT64``, ``SYMBOL_MANGLING``), see
:ref:`ilp64-builds`.


Running the Install
-------------------

After building:

.. code-block:: bash

   meson install -C builddir

This places:

- ``libsemilapack.a`` (or ``.so`` / ``.dylib``) in ``<prefix>/lib/``
- Public headers in ``<prefix>/include/semicolon_lapack/``
- A pkg-config file at ``<prefix>/lib/pkgconfig/semicolon-lapack.pc``

To install to a custom prefix:

.. code-block:: bash

   meson setup builddir --prefix=/opt/semicolon-lapack
   ninja -C builddir
   meson install -C builddir


Installed Headers
-----------------

The installed headers have **concrete types baked in**. There are no macros to
define, no ``#ifdef`` guards, and no internal build headers. IDEs and static
analyzers parse them directly with full autocomplete and type checking.

**LP64 layout** (default):

.. code-block:: text

   <prefix>/include/semicolon_lapack/
   ├── types.h                            # i32, i64, f32, f64, c64, c128
   ├── semicolon_lapack.h                 # Aggregator (includes everything below)
   ├── semicolon_lapack_double.h          # Double precision (f64)
   ├── semicolon_lapack_single.h          # Single precision (f32)
   ├── semicolon_lapack_complex_double.h  # Double complex (c128)
   ├── semicolon_lapack_complex_single.h  # Single complex (c64)
   └── semicolon_lapack_auxiliary.h       # Auxiliary routines

LP64 headers contain ``i32`` integer parameters and bare function names:

.. code-block:: c

   SEMICOLON_API void dgesv(const i32 n, const i32 nrhs, f64* A, const i32 lda,
                            i32* ipiv, f64* B, const i32 ldb, i32* info);

**ILP64 layout** (``-DUSE_INT64=true``):

Same files, but the aggregator is named ``semicolon_lapack_64.h``. All headers
contain ``i64`` parameters and suffixed function names:

.. code-block:: c

   SEMICOLON_API void dgesv_64(const i64 n, const i64 nrhs, f64* A, const i64 lda,
                               i64* ipiv, f64* B, const i64 ldb, i64* info);

**What is NOT installed:**

Internal build headers (``lapack_name_map.h``, ``semicolon_cblas.h``) are not
installed. They are used only during library compilation.


Type Aliases
------------

``types.h`` defines the following aliases, available in all build modes:

.. list-table::
   :header-rows: 1
   :widths: 15 25 15

   * - Alias
     - Underlying type
     - Size
   * - ``i32``
     - ``int32_t``
     - 4 bytes
   * - ``i64``
     - ``int64_t``
     - 8 bytes
   * - ``f32``
     - ``float``
     - 4 bytes
   * - ``f64``
     - ``double``
     - 8 bytes
   * - ``c64``
     - ``float _Complex``
     - 8 bytes
   * - ``c128``
     - ``double _Complex``
     - 16 bytes

Both ``i32`` and ``i64`` are always defined. LP64 headers use ``i32`` for integer
parameters; ILP64 headers use ``i64``.


Using the Library
-----------------

pkg-config
^^^^^^^^^^

.. code-block:: bash

   gcc myprogram.c $(pkg-config --cflags --libs semicolon-lapack) -o myprogram

The BLAS dependency is in ``Requires.private``, so pkg-config automatically
pulls in the correct BLAS flags when linking statically. For shared libraries,
the BLAS dependency is resolved at runtime via ``DT_NEEDED``.

If the library was installed to a non-standard prefix:

.. code-block:: bash

   PKG_CONFIG_PATH=/opt/semicolon-lapack/lib/pkgconfig \
       gcc myprogram.c $(pkg-config --cflags --libs semicolon-lapack) -o myprogram

Meson subproject
^^^^^^^^^^^^^^^^

.. code-block:: meson

   semilapack_dep = dependency('semicolon-lapack')
   executable('myprogram', 'myprogram.c', dependencies: semilapack_dep)

CMake via pkg-config
^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

   find_package(PkgConfig REQUIRED)
   pkg_check_modules(SEMILAPACK REQUIRED semicolon-lapack)

   add_executable(myprogram myprogram.c)
   target_link_libraries(myprogram ${SEMILAPACK_LIBRARIES})
   target_include_directories(myprogram PRIVATE ${SEMILAPACK_INCLUDE_DIRS})


LP64 Example
------------

.. code-block:: c

   #include <semicolon_lapack/semicolon_lapack.h>
   #include <stdio.h>

   int main(void) {
       i32 n = 3, nrhs = 1, lda = 3, ldb = 3, info;
       i32 ipiv[3];
       f64 A[9] = {2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0};
       f64 B[3] = {1.0, 2.0, 3.0};

       dgesv(n, nrhs, A, lda, ipiv, B, ldb, &info);
       printf("info = %d, solution: %f %f %f\n", info, B[0], B[1], B[2]);
       return 0;
   }


ILP64 Example
-------------

Notice the ``_64`` suffix on the header and function names (if default symbol
mangling is used). For how to configure the suffix, see :ref:`ilp64-symbol-naming`.

.. code-block:: c

   #include <semicolon_lapack/semicolon_lapack_64.h>
   #include <stdio.h>

   int main(void) {
       i64 n = 3, nrhs = 1, lda = 3, ldb = 3, info;
       i64 ipiv[3];
       f64 A[9] = {2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0};
       f64 B[3] = {1.0, 2.0, 3.0};

       dgesv_64(n, nrhs, A, lda, ipiv, B, ldb, &info);
       printf("info = %lld, solution: %f %f %f\n",
              (long long)info, B[0], B[1], B[2]);
       return 0;
   }


Type Safety
-----------

**Compile time** -- pointer parameters catch type errors:

.. code-block:: c

   #include <semicolon_lapack/semicolon_lapack_64.h>

   int ipiv[100], info;           /* WRONG: should be i64 for ILP64 */
   dgetrf_64(m, n, A, lda, ipiv, &info);
   /*                      ^^^^   ^^^^^
    * compiler error: incompatible pointer types (int* vs i64*) */

Value parameters (``m``, ``n``) silently widen from ``int`` to ``int64_t``,
which is harmless. However, generated concrete types avoid difficult to
debug segfaults and silent mismatches. Pointer parameters (``ipiv``,
``info``) produce type mismatch compiler errors.

**Link time** -- symbol names catch header/library mismatches:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - User includes
     - Library provides
     - Result
   * - LP64 header (``dgesv``)
     - ILP64 library (``dgesv_64``)
     - Linker error: undefined ``dgesv``
   * - ILP64 header (``dgesv_64``)
     - LP64 library (``dgesv``)
     - Linker error: undefined ``dgesv_64``


Shared Libraries
----------------

By default, shared libraries on ELF platforms (Linux, FreeBSD) export **every**
symbol. This is a problem for a library with ~1930 functions and many internal
helpers: consumers can accidentally link against an internal symbol that was never
meant to be part of the public API, and the dynamic linker has to resolve
thousands of unnecessary symbols at load time.

The public LAPACK functions are marked with
``__attribute__((visibility("default")))`` via the ``SEMICOLON_API`` macro in the
headers. When you compile with ``-fvisibility=hidden`` (which is recommended for
shared builds), only these marked functions are exported. Everything else —
internal helpers, static functions, CBLAS wrappers — becomes invisible to the
dynamic linker. Note that hidden symbols are still physically present in the
binary and can be found with introspection tools like ``nm`` or ``readelf``, but
they are not part of the public ABI and will not be resolved for external
callers.

On Windows (non-MSVC compilers such as llvm-mingw), the same problem is solved
differently. Public symbols must be annotated with ``__declspec(dllexport)`` when
**building** the DLL, and ``__declspec(dllimport)`` when **consuming** it. The
``SEMICOLON_API`` macro handles the build side automatically. Consumers of the
shared library on Windows should define ``SEMICOLON_USE_SHARED`` before including
the header to get the import annotation.
