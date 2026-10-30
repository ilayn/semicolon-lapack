# tests/meson.build
# Test suite for semicolon-lapack using CMocka
#
# Shared infrastructure (test_harness.h, test_rng.h, xerbla_test.c, etc.)
# lives at this level. Each precision gets its own subdirectory with
# test drivers, verification utilities, and smoke tests.

test_env = environment({'OPENBLAS_NUM_THREADS': '1', 'CMOCKA_MESSAGE_OUTPUT': 'TAP'})

# Double precision tests
subdir('d')

# Single precision tests
subdir('s')

# Complex double precision tests
subdir('z')

# Complex single precision tests
subdir('c')

# Precision-independent tests
test('name_map_current',
  python,
  args: [files('../scripts/generate_name_map.py'), '--check'],
  suite: 'build',
  timeout: 60,
)

test('rng_validation',
  executable('test_rng_validation', 'test_rng_validation.c',
    include_directories: include_directories('.'),
    dependencies: [semilapack_dep, cmocka_dep],
  ),
  suite: 'rng',
  timeout: 120,
  is_parallel: true,
  protocol: 'tap',
  env: test_env,
)

test('iparmq_case',
  executable('test_iparmq_case', 'test_iparmq_case.c',
    include_directories: include_directories('.'),
    dependencies: [semilapack_dep, cmocka_dep],
  ),
  suite: 'auxiliary',
  timeout: 60,
  is_parallel: true,
  protocol: 'tap',
  env: test_env,
)
