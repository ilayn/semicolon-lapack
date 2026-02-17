# semicolon-lapack: LAPACK in C
# Root meson.build

project('semicolon-lapack', 'c',
  version: '0.1.0',
  license: 'BSD-3-Clause',
  meson_version: '>=1.1.0',
  default_options: [
    'c_std=c11',
    'warning_level=3',
    'buildtype=release',
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

# BLAS/CBLAS dependency
# Use -Dblas=<vendor> to select: auto, openblas, blis,
# or any pkg-config package name for direct control
# (e.g., -Dblas=mkl-dynamic-lp64-seq).
blas_vendor = get_option('blas')

if blas_vendor == 'auto'
  blas_dep = dependency('openblas', required: false)
  if not blas_dep.found()
    blas_dep = dependency('blas-openblas', required: false)
  endif
  if not blas_dep.found()
    blas_dep = dependency('blis', required: false)
  endif
  if not blas_dep.found()
    blas_dep = dependency('cblas', required: false)
  endif
  if not blas_dep.found()
    blas_dep = dependency('blas', required: false)
  endif
  if not blas_dep.found()
    blas_dep = cc.find_library('openblas', required: false)
  endif
  if not blas_dep.found()
    blas_dep = cc.find_library('blas', required: true)
  endif
elif blas_vendor == 'openblas'
  blas_dep = dependency('openblas', required: false)
  if not blas_dep.found()
    blas_dep = dependency('blas-openblas', required: false)
  endif
  if not blas_dep.found()
    blas_dep = cc.find_library('openblas', required: true)
  endif
elif blas_vendor == 'blis'
  blas_dep = dependency('blis', required: true)
else
  # Treat as a direct pkg-config package name
  blas_dep = dependency(blas_vendor, required: true)
endif

# Common include directories
inc_pub = include_directories('include')
inc_src = include_directories('src/include')

# Build the library
subdir('src')

# Install public headers
install_headers(
  'include/semicolon_lapack/semicolon_lapack.h',
  'include/semicolon_lapack/types.h',
  subdir: 'semicolon_lapack',
)
install_headers(
  'src/include/semicolon_lapack_double.h',
  'src/include/semicolon_lapack_single.h',
  'src/include/semicolon_lapack_complex_double.h',
  'src/include/semicolon_lapack_complex_single.h',
  'src/include/semicolon_lapack_auxiliary.h',
  subdir: 'semicolon_lapack',
)

# Generate pkg-config file
pkg = import('pkgconfig')
pkg.generate(libsemilapack,
  name: 'semicolon-lapack',
  description: 'LAPACK linear algebra routines implemented in C',
  url: 'https://github.com/ilayn/semicolon-lapack',
)

# Testing (optional, enabled by default)
if get_option('tests')
  cmocka_dep = dependency('cmocka', required: true)
  subdir('tests')
endif

# Benchmarks (optional, disabled by default)
if get_option('benchmarks')
  subdir('benchmark')
endif
