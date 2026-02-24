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
  # Comprehensive test drivers will be added here as they are ported
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
