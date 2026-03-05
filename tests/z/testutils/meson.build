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

  # Real-valued shared verification (from d-prefix, precision-independent)
  '../../d/testutils/dget06.c',  # Condition number comparison
  '../../d/testutils/dsvdct.c',  # Sturm sequence singular value count
  '../../d/testutils/dsvdch.c',  # Sturm sequence singular value check
  '../../d/testutils/dstect.c',  # Sturm sequence eigenvalue count (dstech dependency)
  '../../d/testutils/dstech.c',  # Sturm sequence eigenvalue check
  '../../d/testutils/dsxt1.c',   # Eigenvalue range comparison

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

  # Hermitian positive definite tridiagonal verification (zptt*)
  'zptt01.c',   # Hermitian tridiagonal factorization: ||L*D*L' - A|| / (n * ||A|| * eps)
  'zptt02.c',   # Hermitian tridiagonal solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zptt05.c',   # Hermitian tridiagonal refinement error bounds (FERR/BERR)

  # Full triangular verification (ztrt*)
  'ztrt01.c',   # Triangular factorization: ||A*AINV - I|| / (N * ||A|| * ||AINV|| * eps)
  'ztrt02.c',   # Triangular solve residual: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'ztrt03.c',   # Triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'ztrt05.c',   # Triangular error bounds (FERR/BERR)
  'ztrt06.c',   # Triangular condition number comparison

  # Banded triangular verification (ztbt*)
  'ztbt02.c',   # Band triangular solve residual: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'ztbt03.c',   # Band triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'ztbt05.c',   # Band triangular error bounds (FERR/BERR)
  'ztbt06.c',   # Band triangular condition number comparison

  # Packed triangular verification (ztpt*)
  'ztpt01.c',   # Packed triangular factorization: ||A*AINV - I|| / (N * ||A|| * ||AINV|| * eps)
  'ztpt02.c',   # Packed triangular solve residual: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'ztpt03.c',   # Packed triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'ztpt05.c',   # Packed triangular error bounds (FERR/BERR)
  'ztpt06.c',   # Packed triangular condition number comparison

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
  'zsyt01.c',       # Symmetric factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'zsyt01_rook.c',  # Symmetric Rook factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'zsyt01_3.c',     # Symmetric RK/BK factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'zsyt01_aa.c',    # Symmetric Aasen factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'zspt01.c',       # Symmetric packed factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'zspt02.c',       # Symmetric packed solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zspt03.c',       # Symmetric packed inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'zsyt02.c',       # Symmetric solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zsyt03.c',       # Symmetric inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)

  # Hermitian packed factorization verification
  'zhpt01.c',       # Hermitian packed factorization: ||L*D*L' - A|| / (N * ||A|| * eps)

  # Hermitian positive definite verification (zpot*)
  'zpot01.c',       # Hermitian PD Cholesky: ||L*L' - A|| / (N * ||A|| * eps)
  'zpot02.c',       # Hermitian PD solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zpot03.c',       # Hermitian PD inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'zpot05.c',       # Hermitian PD error bounds (FERR/BERR)
  'zpot06.c',       # Hermitian PD norm comparison

  # Hermitian positive definite packed verification (zppt*)
  'zppt01.c',       # Hermitian PD packed Cholesky: ||L*L' - A|| / (N * ||A|| * eps)
  'zppt02.c',       # Hermitian PD packed solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zppt03.c',       # Hermitian PD packed inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'zppt05.c',       # Hermitian PD packed error bounds: forward/backward error estimates

  # Hermitian positive definite band verification (zpbt*)
  'zpbt01.c',       # Hermitian PD band Cholesky: ||L*L' - A|| / (N * ||A|| * eps)
  'zpbt02.c',       # Hermitian PD band solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'zpbt05.c',       # Hermitian PD band error bounds (FERR/BERR)

  # Hermitian positive semidefinite pivoted Cholesky verification (zpst*)
  'zpst01.c',       # Hermitian PSD pivoted Cholesky: ||P*L*L'*P' - A|| / (N * ||A|| * eps)

  # QR verification routines (zqrt*, zqpt*)
  'zqrt01.c',       # QR factorization: ||Q'*Q - I|| and ||R - Q'*A|| / (M * ||A|| * eps)
  'zqrt01p.c',      # QR factorization with non-negative R diagonal
  'zqrt02.c',       # Partial Q generation: ||R - Q'*A|| / (M * ||A|| * eps)
  'zqrt03.c',       # Q multiply test: ||Q*C - Q*C|| / (M * ||C|| * eps)
  'zqrt04.c',       # Block QR (ZGEQRT/ZGEMQRT) test
  'zqrt05.c',       # Block triangular QR (ZTPQRT/ZTPMQRT) test
  'zqrt11.c',       # Orthogonality test: ||Q'*Q - I|| / (eps * M)
  'zqrt12.c',       # SVD rank test: ||svd(R) - s|| / (||s|| * eps * max(M,N))
  'zqrt13.c',       # Scale-balanced matrix generation
  'zqrt14.c',       # Least-squares residual orthogonality
  'zqrt15.c',       # Rank-deficient matrix generation
  'zqrt16.c',       # Residual: ||B - A*X|| / (max(M,N) * ||A|| * ||X|| * eps)
  'zqrt17.c',       # Least-squares residual norm
  'zqpt01.c',       # Pivoted QR: ||A*P - Q*R|| / (||A|| * eps * max(M,N))

  # QL verification routines (zqlt*)
  'zqlt01.c',       # QL factorization: ||Q'*Q - I|| and ||L - Q'*A|| / (M * ||A|| * eps)
  'zqlt02.c',       # QL partial Q generation
  'zqlt03.c',       # QL Q multiply test

  # LQ verification routines (zlqt*)
  'zlqt01.c',       # LQ factorization: ||Q*Q' - I|| and ||L - A*Q'|| / (N * ||A|| * eps)
  'zlqt02.c',       # LQ partial Q generation
  'zlqt03.c',       # LQ Q multiply test
  'zlqt04.c',       # Block LQ (ZGELQT/ZGEMLQT) test
  'zlqt05.c',       # Block triangular LQ (ZTPLQT/ZTPMLQT) test

  # RQ verification routines (zrqt*)
  'zrqt01.c',       # RQ factorization: ||Q*Q' - I|| and ||R - A*Q'|| / (N * ||A|| * eps)
  'zrqt02.c',       # RQ partial Q generation
  'zrqt03.c',       # RQ Q multiply test

  # RZ verification routines (zrzt*)
  'zrzt01.c',       # RZ factorization residual: ||A - R*Z|| / (||A|| * eps * max(M,N))
  'zrzt02.c',       # RZ Q orthogonality: ||I - Z*Z'|| / (N * eps)

  # Special verification routines
  'ztsqr01.c',      # Tall-skinny QR (ZGEQR/ZGEMQR/ZGELQ/ZGEMLQ) test
  'zunhr_col01.c',  # Unitary Householder reconstruction (ZUNHR_COL) test
  'zunhr_col02.c',  # Unitary Householder reconstruction (ZGETSQRHRT) test

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
  'zlctes.c',       # Eigenvalue selection: Re(alpha/beta) < 0
  'zlctsx.c',       # Stateful eigenvalue selection for ZGGESX reordering
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
  '../../d/testutils/dlaord.c',  # Real sort (dlaord used by zchkq3/zchkqp3rk for singular values)
  '../../d/testutils/dlatb9.c',  # EIG matrix type params (real-only, shared by zck* tests)
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
