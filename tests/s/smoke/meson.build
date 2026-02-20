# tests/s/smoke/meson.build
# Single precision smoke tests - quick sanity checks

smoke_tests_s = {
  # LU factorization family
  'sgetrf': ['test_sgetrf.c', 'lu'],
  'sgetrs': ['test_sgetrs.c', 'lu'],
  'sgesv':  ['test_sgesv.c',  'lu'],
  'sgetri': ['test_sgetri.c', 'lu'],
  'sgetc2': ['test_sgetc2.c', 'lu'],
  'sgesc2': ['test_sgesc2.c', 'lu'],
  'sgecon': ['test_sgecon.c', 'lu'],
  'sgeequ': ['test_sgeequ.c', 'lu'],
  'sgerfs': ['test_sgerfs.c', 'lu'],
  'sgesvx': ['test_sgesvx.c', 'lu'],

  # Cholesky factorization family
  'spotrf': ['test_spotrf.c', 'cholesky'],
  'spotrs': ['test_spotrs.c', 'cholesky'],
  'sposv':  ['test_sposv.c',  'cholesky'],
  'spotri': ['test_spotri.c', 'cholesky'],
  'spocon': ['test_spocon.c', 'cholesky'],
  'spoequ': ['test_spoequ.c', 'cholesky'],
  'sporfs': ['test_sporfs.c', 'cholesky'],
  'sposvx': ['test_sposvx.c', 'cholesky'],

  # Symmetric indefinite family
  'ssytrf': ['test_ssytrf.c', 'symmetric'],
  'ssytrs': ['test_ssytrs.c', 'symmetric'],
  'ssytri': ['test_ssytri.c', 'symmetric'],
  'ssysv':  ['test_ssysv.c',  'symmetric'],
  'ssycon': ['test_ssycon.c', 'symmetric'],
  'ssyrfs': ['test_ssyrfs.c', 'symmetric'],
  'ssysvx': ['test_ssysvx.c', 'symmetric'],

  # Tridiagonal family
  'sgttrf': ['test_sgttrf.c', 'tridiag'],
  'sgttrs': ['test_sgttrs.c', 'tridiag'],
  'sgtsv':  ['test_sgtsv.c',  'tridiag'],
  'sgtcon': ['test_sgtcon.c', 'tridiag'],
  'sgtrfs': ['test_sgtrfs.c', 'tridiag'],
  'sgtsvx': ['test_sgtsvx.c', 'tridiag'],

  # QR/LQ/QL/RQ factorization family
  'sgeqrf': ['test_sgeqrf.c', 'qr'],
  'sgelqf': ['test_sgelqf.c', 'qr'],
  'sgeqlf': ['test_sgeqlf.c', 'qr'],
  'sgerqf': ['test_sgerqf.c', 'qr'],
  'sorgqr': ['test_sorgqr.c', 'qr'],
  'sormqr': ['test_sormqr.c', 'qr'],
  'sgeqp3': ['test_sgeqp3.c', 'qr'],

  # Least squares
  'sgels':  ['test_sgels.c',  'ls'],
  'sgelsy': ['test_sgelsy.c', 'ls'],

  # Auxiliary
  'slaswp': ['test_slaswp.c', 'auxiliary'],

  # Eigenvalue family (symmetric tridiagonal)
  'ssterf': ['test_ssterf.c', 'eigen'],
  'ssteqr': ['test_ssteqr.c', 'eigen'],
  'sstebz': ['test_sstebz.c', 'eigen'],
  'sstein': ['test_sstein.c', 'eigen'],
  'sstev':  ['test_sstev.c',  'eigen'],
  'sstevx': ['test_sstevx.c', 'eigen'],

  # Eigenvalue family (non-symmetric)
  'strevc': ['test_strevc.c', 'eigen'],
  'sgehrd': ['test_sgehrd.c', 'eigen'],
  'shseqr': ['test_shseqr.c', 'eigen'],
  'sgeev':  ['test_sgeev.c',  'eigen'],
  'sgees':  ['test_sgees.c',  'eigen'],
  'sgeevx': ['test_sgeevx.c', 'eigen'],
  'sgeesx': ['test_sgeesx.c', 'eigen'],

  # SVD family
  'sgesvd': ['test_sgesvd.c', 'svd'],

  # GSVD family
  'sggsvd3': ['test_sggsvd3.c', 'gsvd'],

  # Matrix generator tests
  'slatmt_indexing': ['test_slatmt_indexing.c', 'matgen'],
}

foreach name, info : smoke_tests_s
  src = info[0]
  suite = info[1]

  exe = executable('test_' + name,
    src,
    include_directories: test_inc_s,
    dependencies: [verify_dep_s, cmocka_dep],
  )

  test(name, exe,
    suite: suite,
    timeout: 120,
    is_parallel: true,
  )
endforeach
