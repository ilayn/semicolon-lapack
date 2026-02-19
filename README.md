# semicolon-lapack

A C implementation of LAPACK linear algebra library, removing the Fortran dependency while leveraging optimized vendor BLAS through the standard CBLAS interface.


## Rationale
BLAS and LAPACK have been the foundation of numerical linear algebra for decades. Vendors have rewritten BLAS in C/Assembly for performance, but LAPACK remains in Fortran 77, and embedding Fortran into C projects brings a cascade of integration concerns, from name mangling and calling conventions to 1-based indexing and a Fortran runtime dependency. On top of that, the BLAS ecosystem is fragmented across providers (OpenBLAS, MKL, BLIS, Accelerate) each with their own symbol and linking conventions. A native C implementation built on the standard CBLAS interface sidesteps both problems.

This project is a line-by-line C translation of the reference LAPACK. Because we control the LAPACK layer code, our only external dependency is the ~100 CBLAS functions interface. This drastically reduces the ABI surface compared to projects wrapping vendor Fortran LAPACK, and makes LP64/ILP64 support a clean dual build without symbol mangling. A compile-time probe auto-detects the linked BLAS integer width (via a clever trick we learned from [libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline)).

In practice, this means a full BLAS/LAPACK stack that needs only a C compiler. For example, one can build OpenBLAS with only its CBLAS option and link it against this project, and you have a complete LAPACK stack built entirely with a C compiler.

## Current Status

All four precisions (double, single, complex, double-complex) are 99% translated (DMD and a few auxiliary routines are missing).

> [!NOTE]
> XBLAS extra-precision variants are out of scope.

Testing is ported from LAPACK's official test suite, for double and single precision versions:

| | Ported | Total |
|--|--------|-------|
| LIN tests (dchk* + ddrv*) | 54 | 54 |
| EIG tests (dchk* + ddrv*) | 22 | 33 |
| Verification routines | 123 | 123 |
| Matrix generators + helpers | 39 | 39 |

The complex precision tests have not been ported yet. We expect frequent code changes in the near term.


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

**Build**
- C11 compiler with `complex.h` support (hence no MSVC, unfortunately)
- Meson >= 1.1.0
- A CBLAS implementation (OpenBLAS, MKL, ...)

**Test**
- CMocka >= 2.0

**Docs**
- Doxygen
- Python with Sphinx and Breathe packages, to convert doxygen Javadoc comments to Sphinx format
- PyData Sphinx Theme

## Testing Suite

### Framework

Tests use [CMocka 2.0+](https://cmocka.org/) testing framework with custom assertion macros for LAPACK-style normalized residual checks. There is no main reasons for this choice other than off-the-shelf support from meson.

### Building and Running Tests

```bash
# Configure and build
meson setup somebuilddir
ninja -C somebuilddir

# Run all tests
meson test -C somebuilddir

# Run by suite
meson test -C somebuilddir --suite lu         # LU factorization tests
meson test -C somebuilddir --suite tridiag    # Tridiagonal tests
meson test -C somebuilddir --suite solve      # Solve/refinement tests
...

# Run a single test
meson test -C somebuilddir dgetrf

# Verbose output
meson test -C somebuilddir -v

# With valgrind
meson test -C somebuilddir --wrapper='valgrind --leak-check=full'
```

### Documentation

Install Python dependencies (use pip, conda, uv, or whichever package manager you prefer):

```bash
pip install -r doc/requirements.txt
```

Build HTML docs (runs Doxygen first, then Sphinx, takes a while) and preview locally:

```bash
cd doc
make html  # Creates build/html folder
cd build/html
python -m http.server 8080  # start a local server
```

Then open http://localhost:8080 (or any port of your choice).


## License

BSD-3-Clause. See [LICENSE](LICENSE).
