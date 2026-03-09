# semicolon-lapack

A C implementation of LAPACK linear algebra library, removing the Fortran dependency while leveraging optimized vendor BLAS through the standard CBLAS interface.


## Rationale
[BLAS and LAPACK libraries](https://netlib.org/lapack) have been the foundation of numerical linear algebra for decades. Vendors have rewritten BLAS in C/Assembly for performance, but LAPACK remains in Fortran 77, and embedding Fortran into C projects brings a cascade of integration concerns, from name mangling and calling conventions to 1-based indexing and a Fortran runtime dependency. On top of that, the BLAS ecosystem is fragmented across providers (OpenBLAS, MKL, BLIS, Accelerate) each with their own symbol and linking conventions. A native C implementation built on the standard CBLAS interface sidesteps both problems.

This project is a line-by-line C translation of the reference LAPACK. Because we control the LAPACK layer code, our only external dependency is the ~150 CBLAS functions interface. This drastically reduces the ABI surface compared to projects wrapping vendor Fortran LAPACK, and makes LP64/ILP64 support a clean dual build without symbol mangling. A compile-time probe auto-detects the linked BLAS integer width (via a clever trick we learned from [libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline)).

In practice, this means a full BLAS/LAPACK stack that needs only a C compiler. For example, one can build OpenBLAS with only its CBLAS option and link it against this project, and you have a complete LAPACK stack built entirely with a C compiler.

## Current Status

All four precisions (double, single, complex, double-complex) are 99% translated (DMD and a few auxiliary routines are missing).

> [!NOTE]
> XBLAS extra-precision variants are out of scope.

Tests are ported from LAPACK's official test suite and fully ported (~450K parametrized test cases).


## Disclosure

This work is heavily assisted by a paid LLM subscription with personal financing. While there is a lot of manual labor involved in porting the critical routines, a significant portion of the codebase was produced predominantly by the LLM, in particular:

- some of the C code for the double precision routines,
- porting the test code and Meson/CMocka conversion,
- S, C, and Z precision generation from D routines (via script),
- `.rst` file generation script,
- generating C-style docstrings from the original Fortran comments.

This creates an obvious ethical concern and we acknowledge and share it, hence the disclosure. A lot of care has gone into ensuring the resulting C code is a mechanical translation with no algorithmic changes. If you notice something that looks like a breach of copyright, please let us know. That should not happen.

There are also intentional, human-made modifications. For example, the LU factorization in `getrf` incorporates a technique we learned from the excellent [faer](https://codeberg.org/sarah-quinones/faer) project. These are deliberate improvements, not accidental machine-generated code.


## Dependencies

A C11 compiler, [Meson](https://mesonbuild.com) >= 1.1.0, and a CBLAS implementation (OpenBLAS, MKL, ...). Tests require [CMocka](https://cmocka.org) >= 2.0. See the [building guide](doc/source/building.rst) for full details including BLAS vendor selection and ILP64 options.

## Building

```bash
meson setup builddir
ninja -C builddir
meson test -C builddir
```

## Documentation

```bash
pip install -r doc/requirements.txt
cd doc && make html
```

See the [contributing guide](doc/source/contributing.rst) for details on the Doxygen + Sphinx pipeline.


## License

BSD-3-Clause. See [LICENSE](LICENSE).
