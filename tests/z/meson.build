# tests/z/meson.build
# Complex double precision test suite

# Build complex double precision test utilities (verification routines)
subdir('testutils')

# Include directories:
#   '..'       -> tests/ (for test_harness.h, test_rng.h)
#   'testutils' -> tests/z/testutils/ (for verify.h)
test_inc_z = include_directories('..', 'testutils')

# =============================================================================
# LAPACK Test Driver Ports (zchk*.f and zdrv*.f)
# Line-by-line translations of the official LAPACK test suite
# =============================================================================

lapack_tests_z = {
  'zchkeq': ['test_zchkeq.c', 'condition'],
  'zchkge': ['test_zchkge.c', 'lu'],
  'zchkgt': ['test_zchkgt.c', 'tridiag'],
  'zchkgb': ['test_zchkgb.c', 'band'],
  'zdrvab': ['test_zdrvab.c', 'solve'],
  'zdrvac': ['test_zdrvac.c', 'solve'],
  'zchkhe': ['test_zchkhe.c', 'hermitian'],
  'zchkhe_aa': ['test_zchkhe_aa.c', 'hermitian'],
  'zchkhe_aa_2stage': ['test_zchkhe_aa_2stage.c', 'hermitian'],
  'zchkhe_rk': ['test_zchkhe_rk.c', 'hermitian'],
  'zchkhe_rook': ['test_zchkhe_rook.c', 'hermitian'],
  'zchkhp': ['test_zchkhp.c', 'hermitian'],
  'zchklq': ['test_zchklq.c', 'qr'],
  'zchklqt': ['test_zchklqt.c', 'qr'],
  'zchklqtp': ['test_zchklqtp.c', 'qr'],
  'zchkps': ['test_zchkps.c', 'cholesky'],
  'zchkpo': ['test_zchkpo.c', 'cholesky'],
  'zchkpp': ['test_zchkpp.c', 'cholesky'],
  'zchkpb': ['test_zchkpb.c', 'band'],
  'zchkpt': ['test_zchkpt.c', 'tridiag'],
}

foreach name, info : lapack_tests_z
  test(name,
    executable('test_' + name, info[0],
      include_directories: test_inc_z,
      dependencies: [verify_dep_z, cmocka_dep],
    ),
    suite: info[1],
    timeout: 300,
    is_parallel: true,
  )
endforeach
