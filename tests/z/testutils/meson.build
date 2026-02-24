# tests/z/testutils/meson.build
# Complex double precision test utilities: verification routines
# Ported from LAPACK TESTING/LIN
#
# This builds a test version of the library that uses lapack_tuning_test.c
# instead of the production lapack_tuning.c. This allows tests to override
# block sizes via xlaenv(), mirroring LAPACK's TESTING/LIN/ilaenv.f approach.

testutils_z_sources = files(
  # Shared infrastructure (at tests/ level)
  '../../lapack_tuning_test.c',
  '../../xerbla_test.c',
  '../../chkxer.c',

  # General matrix verification (zget*)
  'zget01.c',   # LU factorization: ||L*U - A|| / (N * ||A|| * eps)
  'zget02.c',   # Solution residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zget03.c',   # Inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'zget04.c',   # Solution vs exact: ||X - XACT|| scaled by condition
  'zget07.c',   # Error bounds (FERR/BERR)
  'zget08.c',   # Mixed-precision residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zget10.c',   # Compare two matrices: ||A - B|| / (||A|| * eps)
  'zget22.c',   # Nonsymmetric eigenvalue check: A*E - E*W
  'zget23.c',   # ZGEEVX nonsymmetric eigenvalue driver test (11 ratios)
  'zget24.c',   # ZGEESX nonsymmetric Schur form expert driver test (17 ratios)
  'zget35.c',   # Sylvester equation solver (ZTRSYL) test
  'zget36.c',   # Schur form reordering (ZTREXC) test
  'zget37.c',   # Condition numbers (ZTRSNA/ZTREVC) test
  'zget38.c',   # Condition numbers (ZTRSEN) test
  'zget51.c',   # Decomposition check: ||A - U*B*V'|| / (||A|| * n * ulp)
  'zget52.c',   # Generalized eigenvector check: |b*A*E - a*B*E| / (n*ulp*max)
  'zget54.c',   # Generalized Schur check: ||A - U*S*V', B - U*T*V'||

  # General band verification (zgbt*)
  'zgbt01.c',   # Band LU factorization: ||L*U - A|| / (N * ||A|| * eps)
  'zgbt02.c',   # Band solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zgbt05.c',   # Band refinement error bounds (FERR/BERR)

  # Tridiagonal verification (zgtt*)
  'zgtt01.c',   # Tridiagonal LU factorization
  'zgtt02.c',   # Tridiagonal solve residual
  'zgtt05.c',   # Tridiagonal refinement

  # Hermitian indefinite verification (zhet*)
  'zlavhe.c',       # Multiply by Hermitian factors (helper for zhet01)
  'zlavhe_rook.c',  # Multiply by Hermitian Rook factors (helper for zhet01_rook)
  'zlavhp.c',       # Multiply by Hermitian packed factors (helper for zhpt01)
  'zhet01.c',       # Hermitian factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'zhet01_rook.c',  # Hermitian Rook factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'zhet01_3.c',     # Hermitian RK/BK factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'zhet01_aa.c',    # Hermitian Aasen factorization: ||L*D*L' - A|| / (N * ||A|| * eps)

  # Symmetric indefinite verification
  'zlavsy.c',       # Multiply by symmetric factors (helper for zsyt01)
  'zlavsy_rook.c',  # Multiply by symmetric Rook factors (helper for zsyt01_rook)
  'zlavsp.c',       # Multiply by symmetric packed factors (helper for zspt01)

  # Matrix generation (TESTING/MATGEN z-prefix)
  'zlarnv.c',       # Complex random number/vector generation (zlarnd_rng + zlarnv_rng)
  'zlatm1.c',       # Complex diagonal distribution
  'zlarot.c',       # Complex Givens rotation application
  'zlahilb.c',      # Complex Hilbert matrix generation
  'zlakf2.c',       # Complex Kronecker product for gen. Sylvester
  'zlatm5.c',       # Generalized Sylvester test matrices
  'zlatm6.c',       # Generalized eigenvalue test matrices
  'zlatm2.c',       # Complex random matrix entry (column-wise RNG)
  'zlatm3.c',       # Complex random matrix entry (pivot-first RNG)
  'zlagge.c',       # Complex general matrix with bandwidth reduction
  'zlaghe.c',       # Complex Hermitian matrix generation
  'zlagsy.c',       # Complex symmetric matrix generation
  'zlarge.c',       # Random unitary similarity transformation
  'zlaror.c',       # Random unitary matrix application
  'zlatmr.c',       # Complex matrix generation driver
  'zlatms.c',       # Complex matrix generator with specified singular values
  'zlatmt.c',       # Complex test matrix generator with rank control
  'zlatme.c',       # Non-symmetric matrix with specified eigenvalues

  # Test utility routines (TESTING/LIN z-prefix)
  'zgennd.c',       # Non-negative real diagonal check
  'zlaipd.c',       # Set diagonal imaginary parts to BIGNUM (Hermitian test helper)
  'zlaptm.c',       # Hermitian tridiagonal matrix-vector multiply
  'zlatb4.c',       # Matrix parameter setup (all types)
  'zlatb5.c',       # Matrix parameter setup (symmetric/Hermitian)
  'zlarhs.c',       # Right-hand side generation for linear system tests
  'zlattr.c',       # Triangular test matrix generator (full storage)
  'zlattb.c',       # Triangular test matrix generator (band storage)
  'zlattp.c',       # Triangular test matrix generator (packed storage)
  'zlatsp.c',       # Symmetric packed test matrix generator
  'zlatsy.c',       # Symmetric full-storage test matrix generator
  'zgeqls.c',       # QL factorization solve
  'zgerqs.c',       # RQ factorization solve

  # Hermitian eigenvalue verification (TESTING/EIG)
  'zhet21.c',       # Hermitian eigendecomposition: |A - U*S*U'| / (|A| n ulp)
  'zhet22.c',       # Hermitian partial eigendecomposition: |U'AU - S| / (|A| m ulp)
  'zhpt21.c',       # Hermitian packed eigendecomposition: |A - U*S*U'| / (|A| n ulp)

  # EIG test helpers
  'zlsets.c',       # LSE test helper (tests ZGGLSE)
  'zglmts.c',       # GLM test helper (tests ZGGGLM)
  'zgqrts.c',       # GQR test helper (tests ZGGQRF)
  'zgrqts.c',       # GRQ test helper (tests ZGGRQF)
  'zgsvts3.c',      # GSVD test helper (tests ZGGSVD3)
  'zlatm4.c',       # Eigenvalue test matrix generator
  'zsbmv.c',        # Complex symmetric band matrix-vector product

  # SVD verification
  'zbdt01.c',       # Bidiagonal reduction: ||A - Q*B*P'|| / (||A|| * max(m,n) * eps)
  'zbdt02.c',       # Bidiagonal multiply: ||B - U'*A|| / (||A|| * max(m,n) * eps)
  'zbdt03.c',       # SVD of bidiagonal: ||B*V - U*S|| / (||B|| * max(m,n) * eps)
  'zbdt05.c',       # Partial SVD: |S - U'*A*V| / (n * |A| * eps)

  # Hessenberg reduction verification
  'zhst01.c',       # Hessenberg reduction: ||A - Q*H*Q'|| / (||A|| * n * eps)

  # Hermitian band eigenvalue verification
  'zhbt21.c',       # Hermitian band eigendecomposition: |A - U*S*U'| / (|A| n ulp)

  # Hermitian tridiagonal eigenvalue verification
  'zstt21.c',       # Tridiagonal eigendecomposition: |A - U*S*U'| / (|A| n ulp)
  'zstt22.c',       # Tridiagonal partial eigendecomposition: |A*U - U*S| / (|A| m ulp)

  # Generalized Hermitian-definite eigenvalue verification
  'zsgt01.c',       # Generalized Hermitian eigenvector check

  # Unitary matrix verification
  'zunt01.c',       # Unitary check: ||I - Q'*Q|| / (n * eps)
  'zunt03.c',       # Unitary comparison: span(U) vs span(V)

  # Sylvester equation test
  'zsyl01.c',       # Sylvester equation (ZTRSYL/ZTRSYL3) test

  # CSD (cosine-sine decomposition) verification
  'zcsdts.c',       # CSD verification: ZUNCSD/ZUNCSD2BY1

  '../../d/testutils/dlatm1.c',  # Real diagonal gen (zlatms calls DLATM1, not ZLATM1)
  '../../d/testutils/dlatm7.c',  # Real diagonal gen with rank (zlatmt calls DLATM7)
)

# Build complex double precision test utilities library with test tuning implementation.
# The test tuning is compiled into libtestutils_z, which is linked BEFORE semilapack_dep.
# This means the test xlaenv/lapack_get_nb symbols take precedence.
#
# Include directories:
#   '.'    -> tests/z/testutils/ (for verify.h)
#   '../..' -> tests/ (for test_rng.h, test_harness.h used by shared .c files)
libtestutils_z = static_library('testutils_z',
  testutils_z_sources,
  include_directories: include_directories('.', '../..'),
  dependencies: [semilapack_dep],
)

# Use link_whole to ensure lapack_tuning_test.o is always included.
# This is necessary because the strong symbols from the test tuning
# implementation must override the weak symbols in libsemilapack.
# Without link_whole, the linker won't pull in lapack_tuning_test.o
# since libsemilapack already provides (weak) definitions.
verify_dep_z = declare_dependency(
  link_whole: libtestutils_z,
  include_directories: include_directories('.', '../..'),
  dependencies: [semilapack_dep],
)
