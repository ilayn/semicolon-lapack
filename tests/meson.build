# tests/meson.build
# Test suite for semicolon-lapack using CMocka

# Build test utilities library first (verification routines, matrix generators, etc.)
subdir('testutils')

# Test include directory (for test_harness.h and test_rng.h)
test_inc = include_directories('.', 'testutils')

# =============================================================================
# LAPACK Test Driver Ports (dchk*.f and ddrv*.f)
# These are line-by-line translations of the official LAPACK test suite
# =============================================================================

lapack_tests = {
  # Comprehensive test drivers (dchk*.f ports)
  'dchkge': ['test_dchkge.c', 'lu'],         # General dense LU
  'dchkgb': ['test_dchkgb.c', 'band'],       # General banded
  'dchkpb': ['test_dchkpb.c', 'band'],       # Positive definite banded
  'dchkpo': ['test_dchkpo.c', 'cholesky'],   # Positive definite (Cholesky)
  'dchkpp': ['test_dchkpp.c', 'cholesky'],   # Positive definite packed
  'dchkps': ['test_dchkps.c', 'cholesky'],   # Positive semidefinite pivoted Cholesky
  'dchkgt': ['test_dchkgt.c', 'tridiag'],    # General tridiagonal
  'dchkpt': ['test_dchkpt.c', 'tridiag'],    # Positive definite tridiagonal
  'dchksy': ['test_dchksy.c', 'symmetric'],  # Symmetric indefinite
  'dchksp': ['test_dchksp.c', 'symmetric'],    # Symmetric indefinite packed
  'dchkqr': ['test_dchkqr.c', 'qr'],         # QR factorization
  'dchklq': ['test_dchklq.c', 'qr'],         # LQ factorization
  'dchkql': ['test_dchkql.c', 'qr'],         # QL factorization
  'dchkrq': ['test_dchkrq.c', 'qr'],         # RQ factorization
  'dchkq3': ['test_dchkq3.c', 'qr'],         # QR with column pivoting
  'dchkqp3rk': ['test_dchkqp3rk.c', 'qr'],   # QR with column pivoting (rank-revealing)
  'dchktz': ['test_dchktz.c', 'qr'],         # RZ factorization
  'dchklqt': ['test_dchklqt.c', 'qr'],       # Blocked LQ (DGELQT/DGEMLQT)
  'dchklqtp': ['test_dchklqtp.c', 'qr'],     # Triangular-pentagonal LQ (DTPLQT/DTPMLQT)
  'dchkqrt': ['test_dchkqrt.c', 'qr'],       # Blocked QR (DGEQRT/DGEMQRT)
  'dchkqrtp': ['test_dchkqrtp.c', 'qr'],     # Triangular-pentagonal QR (DTPQRT/DTPMQRT)
  'dchkorhr_col': ['test_dchkorhr_col.c', 'qr'], # Householder reconstruction
  'dchktr': ['test_dchktr.c', 'triangular'], # Triangular matrices
  'dchktp': ['test_dchktp.c', 'triangular'], # Triangular packed matrices
  'dchkeq': ['test_dchkeq.c', 'condition'],  # Equilibration routines
  'dchkrfp': ['test_dchkrfp.c', 'rfp'],      # RFP (Rectangular Full Packed) format

  # Expert driver tests (ddrv*.f ports)
  'ddrvge': ['test_ddrvge.c', 'lu'],         # DGESV/DGESVX drivers
  'ddrvpo': ['test_ddrvpo.c', 'cholesky'],   # DPOSV/DPOSVX drivers
  'ddrvsy': ['test_ddrvsy.c', 'symmetric'],  # DSYSV/DSYSVX drivers
  'ddrvgt': ['test_ddrvgt.c', 'tridiag'],    # DGTSV/DGTSVX drivers
  'ddrvpt': ['test_ddrvpt.c', 'tridiag'],    # DPTSV/DPTSVX drivers
  'ddrvls': ['test_ddrvls.c', 'ls'],         # Least squares drivers
  'ddrvbd': ['test_ddrvbd.c', 'svd'],        # SVD drivers
  'ddrves': ['test_ddrves.c', 'eigen'],      # Non-symmetric Schur (DGEES)
  'ddrvev': ['test_ddrvev.c', 'eigen'],      # Non-symmetric eigenvalue (DGEEV)
  'ddrvst': ['test_ddrvst.c', 'eigen'],      # Symmetric eigenvalue drivers
  'ddrvab': ['test_ddrvab.c', 'solve'],      # DSGESV mixed-precision solver
  'ddrvac': ['test_ddrvac.c', 'solve'],      # DSPOSV mixed-precision solver
}

# Build and register LAPACK test drivers
foreach name, info : lapack_tests
  src = info[0]
  suite = info[1]

  exe = executable('test_' + name,
    src,
    include_directories: test_inc,
    dependencies: [verify_dep, cmocka_dep],
  )

  test(name, exe,
    suite: suite,
    timeout: 120,
    is_parallel: true,
  )
endforeach

# =============================================================================
# Smoke Tests (individual routine tests)
# Quick sanity checks, not comprehensive like LAPACK test drivers
# =============================================================================

subdir('smoke')
