# tests/meson.build
# Test suite for semicolon-lapack using CMocka
#
# Shared infrastructure (test_harness.h, test_rng.h, xerbla_test.c, etc.)
# lives at this level. Each precision gets its own subdirectory with
# test drivers, verification utilities, and smoke tests.

# Double precision tests
subdir('d')

# Single precision tests
subdir('s')

# Precision-independent tests
test('rng_validation',
  executable('test_rng_validation', 'test_rng_validation.c',
    include_directories: include_directories('.'),
    dependencies: [semilapack_dep, cmocka_dep],
  ),
  suite: 'rng',
  timeout: 120,
  is_parallel: true,
)
