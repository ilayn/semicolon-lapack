# semicolon-lapack

A C implementation of LAPACK linear algebra library, removing the Fortran dependency while leveraging optimized vendor BLAS through the standard CBLAS interface.


## Rationale
The original LAPACK is written in Fortran77, which creates friction for non-Fortran projects, to be included in a flexible fashion. Instead, often users must deal with name mangling conventions, Fortran calling conventions, 1-indexing assumptions, and linking against a Fortran runtime. A native C implementation sidesteps many of these issues.

Moreover, the "well-optimized" BLAS/LAPACK vendors, typically, optimize only the BLAS layer in C/Assembly, however, directly compile the reference implementation LAPACK F77 code linking back to BLAS. In this project, our goal is still to use the optimized BLAS, via CBLAS interface, from the vendors but use C implementation of LAPACK on top.

## Dependencies

- **CMocka** >= 2.0 (test framework)
- A BLAS backend that provides CBLAS interface such as **OpenBLAS**, **MKL** and so on.
- **Meson** >= 1.1.0 (build system)


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

## License

BSD-3-Clause. See [LICENSE](LICENSE).
