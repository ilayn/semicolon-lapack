# tests/c/testutils/meson.build
# Complex single precision test utilities: verification routines
# Ported from LAPACK TESTING/LIN
#
# This builds a test version of the library that uses lapack_tuning_test.c
# instead of the production lapack_tuning.c. This allows tests to override
# block sizes via xlaenv(), mirroring LAPACK's TESTING/LIN/ilaenv.f approach.

testutils_c_sources = files(
  # Shared infrastructure (at tests/ level)
  '../../lapack_tuning_test.c',
  '../../xerbla_test.c',
  '../../chkxer.c',

  # Real-valued shared verification (from s-prefix, precision-independent)
  '../../s/testutils/sget06.c',  # Condition number comparison
  '../../s/testutils/ssvdct.c',  # Sturm sequence singular value count
  '../../s/testutils/ssvdch.c',  # Sturm sequence singular value check
  '../../s/testutils/sstect.c',  # Sturm sequence eigenvalue count (dstech dependency)
  '../../s/testutils/sstech.c',  # Sturm sequence eigenvalue check
  '../../s/testutils/ssxt1.c',   # Eigenvalue range comparison

  # General matrix verification (zget*)
  'cget01.c',   # LU factorization: ||L*U - A|| / (N * ||A|| * eps)
  'cget02.c',   # Solution residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'cget03.c',   # Inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'cget04.c',   # Solution vs exact: ||X - XACT|| scaled by condition
  'cget07.c',   # Error bounds (FERR/BERR)
  'cget08.c',   # Mixed-precision residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'cget10.c',   # Compare two matrices: ||A - B|| / (||A|| * eps)
  'cget22.c',   # Nonsymmetric eigenvalue check: A*E - E*W
  'cget23.c',   # ZGEEVX nonsymmetric eigenvalue driver test (11 ratios)
  'cget24.c',   # ZGEESX nonsymmetric Schur form expert driver test (17 ratios)
  'cget35.c',   # Sylvester equation solver (ZTRSYL) test
  'cget36.c',   # Schur form reordering (ZTREXC) test
  'cget37.c',   # Condition numbers (ZTRSNA/ZTREVC) test
  'cget38.c',   # Condition numbers (ZTRSEN) test
  'cget51.c',   # Decomposition check: ||A - U*B*V'|| / (||A|| * n * ulp)
  'cget52.c',   # Generalized eigenvector check: |b*A*E - a*B*E| / (n*ulp*max)
  'cget54.c',   # Generalized Schur check: ||A - U*S*V', B - U*T*V'||

  # General band verification (zgbt*)
  'cgbt01.c',   # Band LU factorization: ||L*U - A|| / (N * ||A|| * eps)
  'cgbt02.c',   # Band solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'cgbt05.c',   # Band refinement error bounds (FERR/BERR)

  # Tridiagonal verification (zgtt*)
  'cgtt01.c',   # Tridiagonal LU factorization
  'cgtt02.c',   # Tridiagonal solve residual
  'cgtt05.c',   # Tridiagonal refinement

  # Hermitian positive definite tridiagonal verification (zptt*)
  'cptt01.c',   # Hermitian tridiagonal factorization: ||L*D*L' - A|| / (n * ||A|| * eps)
  'cptt02.c',   # Hermitian tridiagonal solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'cptt05.c',   # Hermitian tridiagonal refinement error bounds (FERR/BERR)

  # Full triangular verification (ztrt*)
  'ctrt01.c',   # Triangular factorization: ||A*AINV - I|| / (N * ||A|| * ||AINV|| * eps)
  'ctrt02.c',   # Triangular solve residual: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'ctrt03.c',   # Triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'ctrt05.c',   # Triangular error bounds (FERR/BERR)
  'ctrt06.c',   # Triangular condition number comparison

  # Banded triangular verification (ztbt*)
  'ctbt02.c',   # Band triangular solve residual: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'ctbt03.c',   # Band triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'ctbt05.c',   # Band triangular error bounds (FERR/BERR)
  'ctbt06.c',   # Band triangular condition number comparison

  # Packed triangular verification (ztpt*)
  'ctpt01.c',   # Packed triangular factorization: ||A*AINV - I|| / (N * ||A|| * ||AINV|| * eps)
  'ctpt02.c',   # Packed triangular solve residual: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'ctpt03.c',   # Packed triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'ctpt05.c',   # Packed triangular error bounds (FERR/BERR)
  'ctpt06.c',   # Packed triangular condition number comparison

  # Hermitian indefinite verification (zhet*)
  'clavhe.c',       # Multiply by Hermitian factors (helper for zhet01)
  'clavhe_rook.c',  # Multiply by Hermitian Rook factors (helper for zhet01_rook)
  'clavhp.c',       # Multiply by Hermitian packed factors (helper for zhpt01)
  'chet01.c',       # Hermitian factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'chet01_rook.c',  # Hermitian Rook factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'chet01_3.c',     # Hermitian RK/BK factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'chet01_aa.c',    # Hermitian Aasen factorization: ||L*D*L' - A|| / (N * ||A|| * eps)

  # Symmetric indefinite verification
  'clavsy.c',       # Multiply by symmetric factors (helper for zsyt01)
  'clavsy_rook.c',  # Multiply by symmetric Rook factors (helper for zsyt01_rook)
  'clavsp.c',       # Multiply by symmetric packed factors (helper for zspt01)
  'csyt01.c',       # Symmetric factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'csyt01_rook.c',  # Symmetric Rook factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'csyt01_3.c',     # Symmetric RK/BK factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'csyt01_aa.c',    # Symmetric Aasen factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'cspt01.c',       # Symmetric packed factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'cspt02.c',       # Symmetric packed solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'cspt03.c',       # Symmetric packed inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'csyt02.c',       # Symmetric solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'csyt03.c',       # Symmetric inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)

  # Hermitian packed factorization verification
  'chpt01.c',       # Hermitian packed factorization: ||L*D*L' - A|| / (N * ||A|| * eps)

  # Hermitian positive definite verification (zpot*)
  'cpot01.c',       # Hermitian PD Cholesky: ||L*L' - A|| / (N * ||A|| * eps)
  'cpot02.c',       # Hermitian PD solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'cpot03.c',       # Hermitian PD inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'cpot05.c',       # Hermitian PD error bounds (FERR/BERR)
  'cpot06.c',       # Hermitian PD norm comparison

  # Hermitian positive definite packed verification (zppt*)
  'cppt01.c',       # Hermitian PD packed Cholesky: ||L*L' - A|| / (N * ||A|| * eps)
  'cppt02.c',       # Hermitian PD packed solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'cppt03.c',       # Hermitian PD packed inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'cppt05.c',       # Hermitian PD packed error bounds: forward/backward error estimates

  # Hermitian positive definite band verification (zpbt*)
  'cpbt01.c',       # Hermitian PD band Cholesky: ||L*L' - A|| / (N * ||A|| * eps)
  'cpbt02.c',       # Hermitian PD band solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'cpbt05.c',       # Hermitian PD band error bounds (FERR/BERR)

  # Hermitian positive semidefinite pivoted Cholesky verification (zpst*)
  'cpst01.c',       # Hermitian PSD pivoted Cholesky: ||P*L*L'*P' - A|| / (N * ||A|| * eps)

  # QR verification routines (zqrt*, zqpt*)
  'cqrt01.c',       # QR factorization: ||Q'*Q - I|| and ||R - Q'*A|| / (M * ||A|| * eps)
  'cqrt01p.c',      # QR factorization with non-negative R diagonal
  'cqrt02.c',       # Partial Q generation: ||R - Q'*A|| / (M * ||A|| * eps)
  'cqrt03.c',       # Q multiply test: ||Q*C - Q*C|| / (M * ||C|| * eps)
  'cqrt04.c',       # Block QR (ZGEQRT/ZGEMQRT) test
  'cqrt05.c',       # Block triangular QR (ZTPQRT/ZTPMQRT) test
  'cqrt11.c',       # Orthogonality test: ||Q'*Q - I|| / (eps * M)
  'cqrt12.c',       # SVD rank test: ||svd(R) - s|| / (||s|| * eps * max(M,N))
  'cqrt13.c',       # Scale-balanced matrix generation
  'cqrt14.c',       # Least-squares residual orthogonality
  'cqrt15.c',       # Rank-deficient matrix generation
  'cqrt16.c',       # Residual: ||B - A*X|| / (max(M,N) * ||A|| * ||X|| * eps)
  'cqrt17.c',       # Least-squares residual norm
  'cqpt01.c',       # Pivoted QR: ||A*P - Q*R|| / (||A|| * eps * max(M,N))

  # QL verification routines (zqlt*)
  'cqlt01.c',       # QL factorization: ||Q'*Q - I|| and ||L - Q'*A|| / (M * ||A|| * eps)
  'cqlt02.c',       # QL partial Q generation
  'cqlt03.c',       # QL Q multiply test

  # LQ verification routines (zlqt*)
  'clqt01.c',       # LQ factorization: ||Q*Q' - I|| and ||L - A*Q'|| / (N * ||A|| * eps)
  'clqt02.c',       # LQ partial Q generation
  'clqt03.c',       # LQ Q multiply test
  'clqt04.c',       # Block LQ (ZGELQT/ZGEMLQT) test
  'clqt05.c',       # Block triangular LQ (ZTPLQT/ZTPMLQT) test

  # RQ verification routines (zrqt*)
  'crqt01.c',       # RQ factorization: ||Q*Q' - I|| and ||R - A*Q'|| / (N * ||A|| * eps)
  'crqt02.c',       # RQ partial Q generation
  'crqt03.c',       # RQ Q multiply test

  # RZ verification routines (zrzt*)
  'crzt01.c',       # RZ factorization residual: ||A - R*Z|| / (||A|| * eps * max(M,N))
  'crzt02.c',       # RZ Q orthogonality: ||I - Z*Z'|| / (N * eps)

  # Special verification routines
  'ctsqr01.c',      # Tall-skinny QR (ZGEQR/ZGEMQR/ZGELQ/ZGEMLQ) test
  'cunhr_col01.c',  # Unitary Householder reconstruction (ZUNHR_COL) test
  'cunhr_col02.c',  # Unitary Householder reconstruction (ZGETSQRHRT) test

  # Matrix generation (TESTING/MATGEN c-prefix)
  'clarnv.c',       # Complex random number/vector generation (zlarnd_rng + zlarnv_rng)
  'clatm1.c',       # Complex diagonal distribution
  'clarot.c',       # Complex Givens rotation application
  'clahilb.c',      # Complex Hilbert matrix generation
  'clakf2.c',       # Complex Kronecker product for gen. Sylvester
  'clatm5.c',       # Generalized Sylvester test matrices
  'clatm6.c',       # Generalized eigenvalue test matrices
  'clatm2.c',       # Complex random matrix entry (column-wise RNG)
  'clatm3.c',       # Complex random matrix entry (pivot-first RNG)
  'clagge.c',       # Complex general matrix with bandwidth reduction
  'claghe.c',       # Complex Hermitian matrix generation
  'clagsy.c',       # Complex symmetric matrix generation
  'clarge.c',       # Random unitary similarity transformation
  'claror.c',       # Random unitary matrix application
  'clatmr.c',       # Complex matrix generation driver
  'clatms.c',       # Complex matrix generator with specified singular values
  'clatmt.c',       # Complex test matrix generator with rank control
  'clatme.c',       # Non-symmetric matrix with specified eigenvalues

  # Test utility routines (TESTING/LIN c-prefix)
  'cgennd.c',       # Non-negative real diagonal check
  'claipd.c',       # Set diagonal imaginary parts to BIGNUM (Hermitian test helper)
  'claptm.c',       # Hermitian tridiagonal matrix-vector multiply
  'clatb4.c',       # Matrix parameter setup (all types)
  'clatb5.c',       # Matrix parameter setup (symmetric/Hermitian)
  'clarhs.c',       # Right-hand side generation for linear system tests
  'clattr.c',       # Triangular test matrix generator (full storage)
  'clattb.c',       # Triangular test matrix generator (band storage)
  'clattp.c',       # Triangular test matrix generator (packed storage)
  'clatsp.c',       # Symmetric packed test matrix generator
  'clatsy.c',       # Symmetric full-storage test matrix generator
  'cgeqls.c',       # QL factorization solve
  'cgerqs.c',       # RQ factorization solve

  # Hermitian eigenvalue verification (TESTING/EIG)
  'chet21.c',       # Hermitian eigendecomposition: |A - U*S*U'| / (|A| n ulp)
  'chet22.c',       # Hermitian partial eigendecomposition: |U'AU - S| / (|A| m ulp)
  'chpt21.c',       # Hermitian packed eigendecomposition: |A - U*S*U'| / (|A| n ulp)

  # EIG test helpers
  'clsets.c',       # LSE test helper (tests ZGGLSE)
  'cglmts.c',       # GLM test helper (tests ZGGGLM)
  'cgqrts.c',       # GQR test helper (tests ZGGQRF)
  'cgrqts.c',       # GRQ test helper (tests ZGGRQF)
  'cgsvts3.c',      # GSVD test helper (tests ZGGSVD3)
  'clatm4.c',       # Eigenvalue test matrix generator
  'clctes.c',       # Eigenvalue selection: Re(alpha/beta) < 0
  'clctsx.c',       # Stateful eigenvalue selection for ZGGESX reordering
  'csbmv.c',        # Complex symmetric band matrix-vector product

  # SVD verification
  'cbdt01.c',       # Bidiagonal reduction: ||A - Q*B*P'|| / (||A|| * max(m,n) * eps)
  'cbdt02.c',       # Bidiagonal multiply: ||B - U'*A|| / (||A|| * max(m,n) * eps)
  'cbdt03.c',       # SVD of bidiagonal: ||B*V - U*S|| / (||B|| * max(m,n) * eps)
  'cbdt05.c',       # Partial SVD: |S - U'*A*V| / (n * |A| * eps)

  # Hessenberg reduction verification
  'chst01.c',       # Hessenberg reduction: ||A - Q*H*Q'|| / (||A|| * n * eps)

  # Hermitian band eigenvalue verification
  'chbt21.c',       # Hermitian band eigendecomposition: |A - U*S*U'| / (|A| n ulp)

  # Hermitian tridiagonal eigenvalue verification
  'cstt21.c',       # Tridiagonal eigendecomposition: |A - U*S*U'| / (|A| n ulp)
  'cstt22.c',       # Tridiagonal partial eigendecomposition: |A*U - U*S| / (|A| m ulp)

  # Generalized Hermitian-definite eigenvalue verification
  'csgt01.c',       # Generalized Hermitian eigenvector check

  # Unitary matrix verification
  'cunt01.c',       # Unitary check: ||I - Q'*Q|| / (n * eps)
  'cunt03.c',       # Unitary comparison: span(U) vs span(V)

  # Sylvester equation test
  'csyl01.c',       # Sylvester equation (ZTRSYL/ZTRSYL3) test

  # CSD (cosine-sine decomposition) verification
  'ccsdts.c',       # CSD verification: ZUNCSD/ZUNCSD2BY1

  '../../s/testutils/slatm1.c',  # Real diagonal gen (zlatms calls DLATM1, not ZLATM1)
  '../../s/testutils/slatm7.c',  # Real diagonal gen with rank (zlatmt calls DLATM7)
  '../../s/testutils/slaord.c',  # Real sort (dlaord used by zchkq3/zchkqp3rk for singular values)
  '../../s/testutils/slatb9.c',  # EIG matrix type params (real-only, shared by zck* tests)
)

# Build complex single precision test utilities library with test tuning implementation.
# The test tuning is compiled into libtestutils_c, which is linked BEFORE semilapack_dep.
# This means the test xlaenv/lapack_get_nb symbols take precedence.
#
# Include directories:
#   '.'    -> tests/z/testutils/ (for verify.h)
#   '../..' -> tests/ (for test_rng.h, test_harness.h used by shared .c files)
libtestutils_c = static_library('testutils_c',
  testutils_c_sources,
  include_directories: include_directories('.', '../..'),
  dependencies: [semilapack_dep],
)

# Use link_whole to ensure lapack_tuning_test.o is always included.
# This is necessary because the strong symbols from the test tuning
# implementation must override the weak symbols in libsemilapack.
# Without link_whole, the linker won't pull in lapack_tuning_test.o
# since libsemilapack already provides (weak) definitions.
verify_dep_c = declare_dependency(
  link_whole: libtestutils_c,
  include_directories: include_directories('.', '../..'),
  dependencies: [semilapack_dep],
)
