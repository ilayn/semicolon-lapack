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

# Complex arithmetic: -fcx-limited-range drops NaN/Inf recovery from complex
# mult and div, inlining 4 FP ops instead of calling __muldc3/__mulsc3.
# This flag is part of the -fast-math family and reduces complex routine
# sizes ~3x (e.g. zlarfx: 94KB -> 29KB) but trades away C99/C11 math
# correctness. Disabled: this library prioritizes correctness over size.
# The flag here is to make such experiments easy.
# cx_limited = cc.get_supported_arguments(['-fcx-limited-range'])
# add_project_arguments(cx_limited, language: 'c')

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
elif blas_vendor == 'accelerate'
  blas_dep = dependency('appleframeworks', modules: ['Accelerate'], required: true)
else
  # Treat as a direct pkg-config package name
  blas_dep = dependency(blas_vendor, required: true)
endif

# Report which BLAS was found and where it lives
if blas_dep.type_name() == 'pkgconfig'
  _blas_libdir = blas_dep.get_variable(pkgconfig: 'libdir', default_value: '?')
  message('BLAS: ' + blas_dep.name() + ' ' + blas_dep.version() + ' (' + _blas_libdir + ')')
else
  message('BLAS: ' + blas_dep.name() + ' (found via ' + blas_dep.type_name() + ')')
endif

# ILP64 configuration
use_int64 = get_option('USE_INT64')

# Probe the linked BLAS to detect its actual integer width.
# Technique adapted from libblastrampoline (Julia): call cblas_idamax with a
# crafted n value whose lower 32 bits = 3 and full 64 bits are negative.
# LP64 reads lower 32 bits (n=3, returns 1), ILP64 reads full 64 bits (n<0, returns 0).
# Skip during cross-compilation since we cannot run host binaries.
if not meson.is_cross_build()
  # Build the rpath arg so the probe binary loads the correct libopenblas at
  # runtime, even when another version sits on the default search path (common
  # with conda toolchains that inject their own -rpath).
  _probe_args = []
  if blas_dep.type_name() == 'pkgconfig'
    _probe_rpath = blas_dep.get_variable(pkgconfig: 'libdir', default_value: '')
    if _probe_rpath != ''
      _probe_args += '-Wl,-rpath,' + _probe_rpath
    endif
  endif
  probe = cc.run(files('scripts/probe_blas_int.c'),
                 dependencies: blas_dep,
                 args: _probe_args,
                 name: 'BLAS integer width probe')
  if probe.returncode() == 0
    blas_is_ilp64 = false
    message('BLAS integer width probe: LP64 (32-bit)')
  elif probe.returncode() == 1
    blas_is_ilp64 = true
    message('BLAS integer width probe: ILP64 (64-bit)')
  else
    warning('BLAS integer width probe failed (return code ' + probe.returncode().to_string() + '). Cannot verify BLAS integer width.')
    blas_is_ilp64 = use_int64
  endif

  if use_int64 and not blas_is_ilp64
    error('USE_INT64=true but the linked BLAS uses 32-bit integers (LP64). ' +
          'Link against an ILP64 BLAS (e.g., -Dblas=openblas64 or -Dblas=mkl-dynamic-ilp64-seq).')
  elif not use_int64 and blas_is_ilp64
    error('USE_INT64=false but the linked BLAS uses 64-bit integers (ILP64). ' +
          'Either set -DUSE_INT64=true or link against an LP64 BLAS.')
  endif
else
  warning('Cross-compiling: skipping BLAS integer width probe. ' +
          'Ensure the linked BLAS matches USE_INT64=' + use_int64.to_string() + '.')
endif

if use_int64
  add_project_arguments('-DSEMICOLON_ILP64', language: 'c')
  symbol_mangling = get_option('SYMBOL_MANGLING')
  add_project_arguments('-DLAPACK_NAME(name)=' + symbol_mangling, language: 'c')
endif

# CBLAS gemmtr availability.
# The BLAS standard name is cblas_dgemmtr (reference BLAS).  Some older
# vendors only provide the legacy cblas_dgemmt name.  Our code uses gemmtr;
# if the linked library only has the old name, alias it.
if not cc.has_function('cblas_dgemmtr', dependencies: blas_dep)
  if cc.has_function('cblas_dgemmt', dependencies: blas_dep)
    message('BLAS provides cblas_dgemmt but not cblas_dgemmtr — adding aliases')
    add_project_arguments(
      '-Dcblas_sgemmtr=cblas_sgemmt',
      '-Dcblas_dgemmtr=cblas_dgemmt',
      '-Dcblas_cgemmtr=cblas_cgemmt',
      '-Dcblas_zgemmtr=cblas_zgemmt',
      language: 'c',
    )
  else
    warning('BLAS provides neither cblas_dgemmtr nor cblas_dgemmt — ' +
            'routines using gemmtr (dlasyf, slasyf, zlahef, clahef) will fail to link.')
  endif
endif

# Common include directories
inc_pub = include_directories('include')
inc_src = include_directories('src/include')

# Build the library
subdir('src')

# Install public headers — types.h is installed as-is (concrete typedefs, no macros)
install_headers(
  'include/semicolon_lapack/types.h',
  subdir: 'semicolon_lapack',
)

# Generate concrete installed headers with all macros resolved.
# Per ILP64_DESIGN.md: installed headers have concrete types baked in,
# no INT macro, no internal_build_defs.h, no lapack_name_map.h.
python = find_program('python3')
if use_int64
  gen_int_type = 'i64'
  gen_naming = get_option('SYMBOL_MANGLING')
else
  gen_int_type = 'i32'
  gen_naming = ''
endif

if use_int64
  gen_aggregator = 'semicolon_lapack_64.h'
else
  gen_aggregator = 'semicolon_lapack.h'
endif

generated_headers = custom_target('installed_headers',
  command: [python, files('scripts/generate_installed_headers.py'),
            '--int-type', gen_int_type,
            '--naming', gen_naming,
            '--input-dir', meson.current_source_dir() / 'src' / 'include',
            '--public-dir', meson.current_source_dir() / 'include' / 'semicolon_lapack',
            '--output-dir', '@OUTDIR@'],
  output: [
    gen_aggregator,
    'semicolon_lapack_double.h',
    'semicolon_lapack_single.h',
    'semicolon_lapack_complex_double.h',
    'semicolon_lapack_complex_single.h',
    'semicolon_lapack_auxiliary.h',
  ],
  install: true,
  install_dir: get_option('includedir') / 'semicolon_lapack',
)

# Generate pkg-config file
pkg = import('pkgconfig')
pkg.generate(libsemilapack,
  name: 'semicolon-lapack',
  description: 'LAPACK linear algebra routines implemented in C',
  url: 'https://github.com/ilayn/semicolon-lapack',
)

# Fortran-ABI compatibility shim (optional, disabled by default)
if get_option('fabi_shim')
  subdir('src/fortran_shim')
endif

# Testing (optional, enabled by default)
if get_option('tests')
  cmocka_dep = dependency('cmocka', required: true)
  subdir('tests')
endif

# Benchmarks (optional, disabled by default)
if get_option('benchmarks')
  subdir('benchmark')
endif
