# semicolon-lapack: LAPACK in C
# Root meson.build

project('semicolon-lapack', 'c',
  version: '0.1.0',
  license: 'BSD-3-Clause',
  meson_version: '>=1.1.0',
  default_options: [
    'c_std=c99',
    'warning_level=3',
    'buildtype=debugoptimized',
  ]
)

# Compiler setup
cc = meson.get_compiler('c')

# Extra warning flags for thorough checking
# Note: -Wconversion and -Wsign-conversion are disabled because LAPACK uses
# int everywhere while CBLAS uses size_t, creating thousands of false positives
# for well-bounded values.
extra_warnings = cc.get_supported_arguments([
  '-Wextra',
  '-Wpedantic',
  '-Wdouble-promotion',
  '-Wformat=2',
  '-Wundef',
  '-Wshadow',
  '-Wcast-qual',
  '-Wcast-align',
  '-Wwrite-strings',
  '-Wstrict-prototypes',
  '-Wold-style-definition',
  '-Wmissing-prototypes',
  '-Wmissing-declarations',
  '-Wredundant-decls',
  '-Wnested-externs',
  '-Winline',
  '-Wdisabled-optimization',
  '-Wstack-protector',
  '-Wlogical-op',
  '-Wnull-dereference',
  '-Wjump-misses-init',
  '-Walloc-zero',
  '-Wduplicated-cond',
  '-Wduplicated-branches',
])
add_project_arguments(extra_warnings, language: 'c')

# Math library (required on some platforms)
m_dep = cc.find_library('m', required: false)

# OpenBLAS dependency - prefer pkg-config, fall back to manual detection
openblas_dep = dependency('openblas', required: false)
if not openblas_dep.found()
  # Try blas-openblas (some distros use this name)
  openblas_dep = dependency('blas-openblas', required: false)
endif
if not openblas_dep.found()
  # Manual detection as last resort
  openblas_dep = cc.find_library('openblas', required: true)
endif

# Common include directories
inc_pub = include_directories('include')
inc_src = include_directories('src/include')

# Build the library
subdir('src')

# Testing (optional, enabled by default)
if get_option('tests')
  cmocka_dep = dependency('cmocka', required: true)
  subdir('tests')
endif

# Benchmarks (optional, disabled by default)
if get_option('benchmarks')
  subdir('benchmark')
endif
