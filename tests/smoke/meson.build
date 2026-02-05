# tests/smoke/meson.build
# Individual routine smoke tests - quick sanity checks

smoke_tests = {
  # LU factorization family
  'dgetrf': ['test_dgetrf.c', 'lu'],
  'dgetrs': ['test_dgetrs.c', 'lu'],
  'dgesv':  ['test_dgesv.c',  'lu'],
  'dgetri': ['test_dgetri.c', 'lu'],
  'dgetc2': ['test_dgetc2.c', 'lu'],
  'dgesc2': ['test_dgesc2.c', 'lu'],
  'dgecon': ['test_dgecon.c', 'lu'],
  'dgeequ': ['test_dgeequ.c', 'lu'],
  'dgerfs': ['test_dgerfs.c', 'lu'],
  'dgesvx': ['test_dgesvx.c', 'lu'],
  'dsgesv': ['test_dsgesv.c', 'lu'],

  # Cholesky factorization family
  'dpotrf': ['test_dpotrf.c', 'cholesky'],
  'dpotrs': ['test_dpotrs.c', 'cholesky'],
  'dposv':  ['test_dposv.c',  'cholesky'],
  'dpotri': ['test_dpotri.c', 'cholesky'],
  'dpocon': ['test_dpocon.c', 'cholesky'],
  'dpoequ': ['test_dpoequ.c', 'cholesky'],
  'dporfs': ['test_dporfs.c', 'cholesky'],
  'dposvx': ['test_dposvx.c', 'cholesky'],

  # Symmetric indefinite family
  'dsytrf': ['test_dsytrf.c', 'symmetric'],
  'dsytrs': ['test_dsytrs.c', 'symmetric'],
  'dsytri': ['test_dsytri.c', 'symmetric'],
  'dsysv':  ['test_dsysv.c',  'symmetric'],
  'dsycon': ['test_dsycon.c', 'symmetric'],
  'dsyrfs': ['test_dsyrfs.c', 'symmetric'],
  'dsysvx': ['test_dsysvx.c', 'symmetric'],

  # Tridiagonal family
  'dgttrf': ['test_dgttrf.c', 'tridiag'],
  'dgttrs': ['test_dgttrs.c', 'tridiag'],
  'dgtsv':  ['test_dgtsv.c',  'tridiag'],
  'dgtcon': ['test_dgtcon.c', 'tridiag'],
  'dgtrfs': ['test_dgtrfs.c', 'tridiag'],
  'dgtsvx': ['test_dgtsvx.c', 'tridiag'],

  # QR/LQ/QL/RQ factorization family
  'dgeqrf': ['test_dgeqrf.c', 'qr'],
  'dgelqf': ['test_dgelqf.c', 'qr'],
  'dgeqlf': ['test_dgeqlf.c', 'qr'],
  'dgerqf': ['test_dgerqf.c', 'qr'],
  'dorgqr': ['test_dorgqr.c', 'qr'],
  'dormqr': ['test_dormqr.c', 'qr'],
  'dgeqp3': ['test_dgeqp3.c', 'qr'],

  # Least squares
  'dgels':  ['test_dgels.c',  'ls'],
  'dgelsy': ['test_dgelsy.c', 'ls'],

  # Auxiliary
  'dlaswp': ['test_dlaswp.c', 'auxiliary'],

  # Eigenvalue family (symmetric tridiagonal)
  'dsterf': ['test_dsterf.c', 'eigen'],
  'dsteqr': ['test_dsteqr.c', 'eigen'],
  'dstebz': ['test_dstebz.c', 'eigen'],
  'dstein': ['test_dstein.c', 'eigen'],
  'dstev':  ['test_dstev.c',  'eigen'],
  'dstevx': ['test_dstevx.c', 'eigen'],

  # Eigenvalue family (non-symmetric)
  'dtrevc': ['test_dtrevc.c', 'eigen'],
  'dgehrd': ['test_dgehrd.c', 'eigen'],
  'dhseqr': ['test_dhseqr.c', 'eigen'],
  'dgeev':  ['test_dgeev.c',  'eigen'],
  'dgees':  ['test_dgees.c',  'eigen'],
  'dgeevx': ['test_dgeevx.c', 'eigen'],
  'dgeesx': ['test_dgeesx.c', 'eigen'],

  # SVD family
  'dgesvd': ['test_dgesvd.c', 'svd'],

  # GSVD family
  'dggsvd3': ['test_dggsvd3.c', 'gsvd'],

  # Matrix generator tests
  'dlatmt_indexing': ['test_dlatmt_indexing.c', 'matgen'],
}

# Build and register smoke tests
foreach name, info : smoke_tests
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
