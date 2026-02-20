# tests/s/testutils/meson.build
# Single precision test utilities: verification routines, matrix generators
# Ported from LAPACK TESTING/LIN and TESTING/MATGEN

testutils_s_sources = files(
  # Shared infrastructure (at tests/ level)
  '../../lapack_tuning_test.c',
  '../../xerbla_test.c',
  '../../chkxer.c',

  # General matrix verification (sget*)
  'sget01.c',
  'sget02.c',
  'sget03.c',
  'sget04.c',
  'sget06.c',
  'sget07.c',

  # QR/LQ/QL/RQ verification
  'sqrt01.c',
  'sqrt01p.c',
  'sqrt02.c',
  'sqrt03.c',
  'sgennd.c',
  'slqt01.c',
  'slqt02.c',
  'slqt03.c',
  'slqt04.c',
  'slqt05.c',
  'sqrt04.c',
  'sqrt05.c',
  'sorhr_col01.c',
  'sorhr_col02.c',
  'stsqr01.c',
  'sqlt01.c',
  'sqlt02.c',
  'sqlt03.c',
  'srqt01.c',
  'srqt02.c',
  'srqt03.c',

  # QR with pivoting verification
  'sqpt01.c',
  'sqrt11.c',
  'sqrt12.c',
  'slaord.c',

  # RZ factorization verification
  'srzt01.c',
  'srzt02.c',

  # Least squares verification
  'sqrt13.c',
  'sqrt14.c',
  'sqrt15.c',
  'sqrt16.c',
  'sqrt17.c',

  # Triangular verification (strt*)
  'strt01.c',
  'strt02.c',
  'strt03.c',
  'strt05.c',
  'strt06.c',

  # Triangular packed verification (stpt*)
  'stpt01.c',
  'stpt02.c',
  'stpt03.c',
  'stpt05.c',
  'stpt06.c',

  # Triangular band verification (stbt*)
  'stbt02.c',
  'stbt03.c',
  'stbt05.c',
  'stbt06.c',

  # Triangular matrix generation
  'slattr.c',
  'slattp.c',
  'slattb.c',

  # Positive definite (Cholesky) verification (spot*)
  'spot01.c',
  'spot02.c',
  'spot03.c',
  'spot05.c',

  # Positive semidefinite (pivoted Cholesky) verification (spst*)
  'spst01.c',

  # Positive definite packed (Cholesky) verification (sppt*)
  'sppt01.c',
  'sppt02.c',
  'sppt03.c',
  'sppt05.c',

  # Symmetric indefinite verification (ssyt*)
  'slavsy.c',
  'slavsp.c',
  'slavsy_rook.c',
  'ssyt01.c',
  'ssyt01_rook.c',
  'ssyt01_3.c',
  'ssyt01_aa.c',
  'sspt01.c',

  # General band verification (sgbt*)
  'sgbt01.c',
  'sgbt02.c',
  'sgbt05.c',

  # Positive definite band verification (spbt*)
  'spbt01.c',
  'spbt02.c',
  'spbt05.c',

  # Tridiagonal verification (sgtt*)
  'sgtt01.c',
  'sgtt02.c',
  'sgtt05.c',

  # Positive definite tridiagonal verification (sptt*)
  'sptt01.c',
  'sptt02.c',
  'sptt05.c',
  'slaptm.c',

  # Eigenvalue verification (sst*)
  'sstt21.c',
  'sstt22.c',
  'ssyt21.c',
  'ssyt22.c',
  'ssbt21.c',
  'ssgt01.c',
  'sspt21.c',
  'ssxt1.c',
  'sstech.c',
  'sstect.c',

  # Eigenvector verification
  'sget22.c',
  'sget10.c',
  'sget23.c',
  'sget24.c',

  # Generalized eigenvalue verification
  'sget51.c',
  'sget52.c',
  'sget53.c',
  'sget54.c',
  'sget31.c',
  'sget32.c',
  'sget33.c',
  'sget34.c',
  'sget35.c',
  'sget39.c',
  'sget36.c',
  'sget37.c',
  'sget38.c',
  'sget40.c',
  'ssyl01.c',

  # Error exit tests
  'serrec.c',

  # SVD verification
  'ssvdct.c',
  'ssvdch.c',

  # Non-symmetric eigenvalue verification
  'sort01.c',
  'sort03.c',
  'shst01.c',

  # SVD verification
  'sbdt01.c',
  'sbdt02.c',
  'sbdt03.c',
  'sbdt04.c',
  'sbdt05.c',

  # GSVD verification
  'sgsvts3.c',

  # GLM verification
  'sglmts.c',

  # LSE verification
  'slsets.c',

  # GQR/GRQ verification
  'sgqrts.c',
  'sgrqts.c',

  # Matrix generation
  'slatb4.c',
  'slatb5.c',
  'slatb9.c',
  'slatms.c',
  'slatm1.c',
  'slagge.c',
  'slagsy.c',
  'slarot.c',
  'slatm4.c',
  'slatme.c',
  'slatm2.c',
  'slatm3.c',
  'slatmr.c',
  'slahilb.c',
  'slakf2.c',
  'slatm5.c',
  'slatm6.c',
  'slatm7.c',
  'slatmt.c',

  # Right-hand side generation
  'slarhs.c',

  # Orthogonal random matrix generator
  'slaror.c',
  'slarge.c',

  # Random number generation (LAPACK-style interface)
  'slarnv.c',

  # QL/RQ solve helpers
  'sgeqls.c',
  'sgerqs.c',

  # CS decomposition verification
  'scsdts.c',
)

libtestutils_s = static_library('testutils_s',
  testutils_s_sources,
  include_directories: include_directories('.', '../..'),
  dependencies: [semilapack_dep],
)

verify_dep_s = declare_dependency(
  link_whole: libtestutils_s,
  include_directories: include_directories('.', '../..'),
  dependencies: [semilapack_dep],
)
