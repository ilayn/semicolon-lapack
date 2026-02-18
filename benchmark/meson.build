# Benchmark executables for profiling
#
# Common c_args:
#   -fno-omit-frame-pointer   for samply / perf profiling

bench_c_args = ['-fno-omit-frame-pointer']

# Our dgetrf implementation
bench_dgetrf = executable('bench_dgetrf',
    'bench_dgetrf.c',
    include_directories: [inc_pub, inc_src],
    link_with: libsemilapack,
    dependencies: [blas_dep, m_dep],
    c_args: bench_c_args,
    build_by_default: false,
)

# Vendor dgetrf via Fortran LAPACK interface
bench_dgetrf_fortran = executable('bench_dgetrf_fortran',
    'bench_dgetrf_fortran.c',
    dependencies: [blas_dep],
    c_args: bench_c_args,
    build_by_default: false,
)

# Vendor dgetrf via LAPACKE C interface
bench_dgetrf_lapacke = executable('bench_dgetrf_lapacke',
    'bench_dgetrf_lapacke.c',
    dependencies: [blas_dep],
    c_args: bench_c_args,
    build_by_default: false,
)

# Our dgeqrf implementation
bench_dgeqrf = executable('bench_dgeqrf',
    'bench_dgeqrf.c',
    include_directories: [inc_pub, inc_src],
    link_with: libsemilapack,
    dependencies: [blas_dep, m_dep],
    c_args: bench_c_args,
    build_by_default: false,
)

# Vendor dgeqrf via Fortran LAPACK interface
bench_dgeqrf_fortran = executable('bench_dgeqrf_fortran',
    'bench_dgeqrf_fortran.c',
    dependencies: [blas_dep],
    c_args: bench_c_args,
    build_by_default: false,
)
