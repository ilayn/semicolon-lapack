# tests/testutils/meson.build
# Test utilities: verification routines, matrix generators, and test infrastructure
# Ported from LAPACK TESTING/LIN and TESTING/MATGEN
#
# This builds a test version of the library that uses lapack_tuning_test.c
# instead of the production lapack_tuning.c. This allows tests to override
# block sizes via xlaenv(), mirroring LAPACK's TESTING/LIN/ilaenv.f approach.

testutils_sources = files(
  # Test version of tuning parameter lookup (provides xlaenv)
  'lapack_tuning_test.c',

  # General matrix verification (dget*)
  'dget01.c',   # LU factorization: ||L*U - A|| / (N * ||A|| * eps)
  'dget02.c',   # Solution residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'dget03.c',   # Inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'dget04.c',   # Solution vs exact: ||X - XACT|| scaled by condition
  'dget06.c',   # Condition number comparison
  'dget07.c',   # Error bounds (FERR/BERR)
  'dget08.c',   # Mixed-precision residual: ||B - A*X|| / (||A|| * ||X|| * eps)

  # QR/LQ/QL/RQ verification
  'dqrt01.c',   # QR factorization: ||R - Q'*A|| and ||I - Q'*Q||
  'dqrt01p.c',  # QR factorization (DGEQRFP): same as dqrt01 but with positive diagonal R
  'dqrt02.c',   # QR Q-generation: partial Q orthogonality
  'dqrt03.c',   # QR Q-application: DORMQR vs explicit DGEMM
  'dgennd.c',   # Check non-negative diagonal (for DGEQRFP tests)
  'dlqt01.c',   # LQ factorization: ||L - A*Q'|| and ||I - Q*Q'||
  'dlqt02.c',   # LQ Q-generation: partial Q orthogonality
  'dlqt03.c',   # LQ Q-application: DORMLQ vs explicit DGEMM
  'dlqt04.c',   # Blocked LQ (DGELQT/DGEMLQT): 6 tests
  'dlqt05.c',   # Pentagonal LQ (DTPLQT/DTPMLQT): 6 tests
  'dqrt04.c',   # Blocked QR (DGEQRT/DGEMQRT): 6 tests
  'dqrt05.c',   # Pentagonal QR (DTPQRT/DTPMQRT): 6 tests
  'dorhr_col01.c',  # DORGTSQR + DORHR_COL: 6 tests
  'dorhr_col02.c',  # DORGTSQR_ROW + DORHR_COL via DGETSQRHRT: 6 tests
  'dtsqr01.c',  # Tall-skinny QR / Short-wide LQ: DGEQR, DGELQ, DGEMQR, DGEMLQ
  'dqlt01.c',   # QL factorization: ||L - Q'*A|| and ||I - Q'*Q||
  'dqlt02.c',   # QL Q-generation: partial Q orthogonality
  'dqlt03.c',   # QL Q-application: DORMQL vs explicit DGEMM
  'drqt01.c',   # RQ factorization: ||R - A*Q'|| and ||I - Q*Q'||
  'drqt02.c',   # RQ Q-generation: partial Q orthogonality
  'drqt03.c',   # RQ Q-application: DORMRQ vs explicit DGEMM

  # QR with pivoting verification
  'dqpt01.c',   # QR pivot: ||A*P - Q*R|| / (||A|| * eps * max(M,N))
  'dqrt11.c',   # QR orthogonality: ||Q'*Q - I|| / (eps * m)
  'dqrt12.c',   # QR SVD comparison: ||svd(R) - svd(A)||
  'dlaord.c',   # Sort vector in increasing or decreasing order

  # RZ factorization verification
  'drzt01.c',   # RZ factorization: ||A - R*Q|| / (||A|| * eps * max(M,N))
  'drzt02.c',   # RZ orthogonality: ||I - Q'*Q|| / (eps * max(M,N))

  # Least squares verification (dqrt13-17)
  'dqrt13.c',   # Full-rank matrix generator with scaling
  'dqrt14.c',   # Check X in row space of A: ||trailing triangle|| / (max(M,N,NRHS)*eps)
  'dqrt15.c',   # Rank-deficient matrix generator
  'dqrt16.c',   # LS residual: ||B - A*X|| / (max(m,n) * ||A|| * ||X|| * eps)
  'dqrt17.c',   # LS normal equations: ||R'*op(A)|| / (||A|| * alpha * max(M,N,NRHS) * eps)

  # Triangular verification (dtrt*)
  'dtrt01.c',   # Triangular inverse: ||A*AINV - I|| / (N * ||A|| * ||AINV|| * eps)
  'dtrt02.c',   # Triangular solve: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'dtrt03.c',   # Triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'dtrt05.c',   # Triangular FERR/BERR bounds verification
  'dtrt06.c',   # Triangular condition number comparison

  # Triangular packed verification (dtpt*)
  'dtpt01.c',   # Packed triangular inverse: ||A*AINV - I|| / (N * ||A|| * ||AINV|| * eps)
  'dtpt02.c',   # Packed triangular solve: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'dtpt03.c',   # Packed triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'dtpt05.c',   # Packed triangular FERR/BERR bounds verification
  'dtpt06.c',   # Packed triangular condition number comparison

  # Triangular band verification (dtbt*)
  'dtbt02.c',   # Band triangular solve: ||op(A)*X - B|| / (||op(A)|| * ||X|| * eps)
  'dtbt03.c',   # Band triangular scaled solve: ||s*b - op(A)*x|| / (||op(A)|| * ||x|| * eps)
  'dtbt05.c',   # Band triangular FERR/BERR bounds verification
  'dtbt06.c',   # Band triangular condition number comparison

  # Triangular matrix generation
  'dlattr.c',   # Generate triangular test matrix (18 types)
  'dlattp.c',   # Generate triangular packed test matrix (18 types)
  'dlattb.c',   # Generate triangular banded test matrix (18 types)

  # Positive definite (Cholesky) verification (dpot*)
  'dpot01.c',   # Cholesky factorization: ||L*L' - A|| / (N * ||A|| * eps)
  'dpot02.c',   # Solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'dpot03.c',   # Inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'dpot05.c',   # Error bounds (FERR/BERR)
  'dpot06.c',   # Mixed-precision residual: ||B - A*X|| / (||A|| * ||X|| * eps)

  # Positive semidefinite (pivoted Cholesky) verification (dpst*)
  'dpst01.c',   # Pivoted Cholesky: ||P*L*L'*P' - A|| / (N * ||A|| * eps)

  # Positive definite packed (Cholesky) verification (dppt*)
  'dppt01.c',   # Packed Cholesky factorization: ||L*L' - A|| / (N * ||A|| * eps)
  'dppt02.c',   # Packed solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'dppt03.c',   # Packed inverse: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
  'dppt05.c',   # Packed error bounds (FERR/BERR)

  # Symmetric indefinite verification (dsyt*)
  'dlavsy.c',   # Multiply by symmetric factors (helper for dsyt01)
  'dlavsp.c',   # Multiply by packed symmetric factors (helper for dspt01)
  'dlavsy_rook.c', # Multiply by symmetric Rook factors (helper for dsyt01_rook)
  'dsyt01.c',   # Symmetric factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'dsyt01_rook.c', # Symmetric Rook factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'dsyt01_3.c',    # Symmetric RK/BK factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'dsyt01_aa.c',   # Symmetric Aasen factorization: ||L*D*L' - A|| / (N * ||A|| * eps)
  'dspt01.c',   # Packed symmetric factorization: ||L*D*L' - A|| / (N * ||A|| * eps)

  # General band verification (dgbt*)
  'dgbt01.c',   # Band LU factorization: ||L*U - A|| / (N * ||A|| * eps)
  'dgbt02.c',   # Band solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'dgbt05.c',   # Band refinement error bounds (FERR/BERR)

  # Positive definite band verification (dpbt*)
  'dpbt01.c',   # Band Cholesky: ||L*L' - A|| / (N * ||A|| * eps)
  'dpbt02.c',   # Band solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'dpbt05.c',   # Band refinement error bounds (FERR/BERR)

  # Tridiagonal verification (dgtt*)
  'dgtt01.c',   # Tridiagonal LU factorization
  'dgtt02.c',   # Tridiagonal solve residual
  'dgtt05.c',   # Tridiagonal refinement

  # Positive definite tridiagonal verification (dptt*)
  'dptt01.c',   # PT factorization: ||L*D*L' - A|| / (n * ||A|| * eps)
  'dptt02.c',   # PT solve residual: ||B - A*X|| / (||A|| * ||X|| * eps)
  'dptt05.c',   # PT error bounds (FERR/BERR)
  'dlaptm.c',   # Symmetric tridiagonal matrix-vector multiply

  # Eigenvalue verification (dst*)
  'dstt21.c',   # Tridiagonal eigenvalue decomposition: |A - USU'| and |I - UU'|
  'dstt22.c',   # Partial tridiagonal eigenvalue: |U'AU - S| and |I - U'U|
  'dsyt21.c',   # Symmetric eigenvalue decomposition: |A - USU'| and |I - UU'|
  'dsyt22.c',   # Partial symmetric eigenvalue: |U'AU - S| and |I - U'U|
  'dsbt21.c',   # Symmetric band eigenvector: |A - USU'| and |I - UU'|
  'dsgt01.c',   # Symmetric gen. eigenvector: |AZ - BZD| etc.
  'dspt21.c',   # Symmetric packed eigenvector: |A - USU'| and |I - UU'|
  'dsxt1.c',    # Compare eigenvalue sets
  'dstech.c',   # Eigenvalue accuracy via Sturm sequence
  'dstect.c',   # Tridiagonal eigenvalue count via Sturm sequence

  # Eigenvector verification (dget22, dget10)
  'dget22.c',   # Eigenvector check: |A*E - E*W| / (|A| |E| ulp)
  'dget10.c',   # Matrix comparison: |A - B| / (|A| M eps)
  'dget23.c',   # DGEEVX verification: 11 tests for eigenvalues/eigenvectors/condition numbers
  'dget24.c',   # DGEESX verification: 17 tests for Schur form/sorting/condition numbers

  # Generalized eigenvalue verification
  'dget51.c',   # Decomposition check: |A - UBV'| / (|A| n ulp)
  'dget52.c',   # Generalized eigenvector check: |bA*E - aB*E| / (n ulp max(|bA|,|aB|))
  'dget53.c',   # 2x2 generalized eigenvalue check: |det(sA - wB)| / ...

  # Non-symmetric eigenvalue verification
  'dort01.c',   # Orthogonality check: |I - U*U'| / (n * eps)
  'dort03.c',   # Compare orthogonal matrices U and V
  'dhst01.c',   # Hessenberg reduction: |A - Q*H*Q'| / (|A| n eps)

  # SVD verification
  'dbdt01.c',   # Bidiagonal reconstruction: |A - Q*B*P'| / (n * |A| * eps)
  'dbdt02.c',   # Change of basis: |B - U*C| / (max(m,n) * |B| * eps)
  'dbdt03.c',   # Full bidiagonal SVD: |B - U*S*VT| / (n * |B| * eps)
  'dbdt04.c',   # Partial bidiagonal SVD: |S - U'*B*V| / (n * |B| * eps)
  'dbdt05.c',   # Partial SVD: |S - U'*A*V| / (n * |A| * eps)

  # GSVD verification
  'dgsvts3.c',  # GSVD: |U'*A*Q - D1*R|, |V'*B*Q - D2*R|, orthogonality, sorting

  # GLM verification
  'dglmts.c',   # GLM: |d - A*x - B*u| / ((|A|+|B|)*(|x|+|u|)*eps)

  # LSE verification
  'dlsets.c',   # LSE: |A*x - c| / (|A|*|x|*eps), |B*x - d| / (|B|*|x|*eps)

  # GQR/GRQ verification
  'dgqrts.c',   # GQR: |R - Q'A|, |TZ - Q'B|, |I - Q'Q|, |I - Z'Z|
  'dgrqts.c',   # GRQ: |R - AQ'|, |TQ - Z'B|, |I - QQ'|, |I - Z'Z|

  # Matrix generation
  'dlatb4.c',   # Matrix parameter setup (11 types)
  'dlatb5.c',   # Matrix parameter setup for tridiag/banded tests
  'dlatb9.c',   # Matrix parameter setup for GLM/GQR/GRQ/GSV/LSE tests
  'dlatms.c',   # Matrix generator with specified singular values
  'dlatm1.c',   # Diagonal value generator (for dlatms)
  'dlagge.c',   # Generate nonsymmetric matrix with random orthogonal transforms
  'dlagsy.c',   # Generate symmetric matrix with random orthogonal transforms
  'dlarot.c',   # Apply plane rotation (for band matrix generation)
  'dlatm4.c',   # Eigenvalue test matrix generator
  'dlatme.c',   # Non-symmetric eigenvalue test matrix generator
  'dlatm2.c',   # Random matrix entry with pivoting (for dlatmr)
  'dlatm3.c',   # Random matrix entry with pivoting variant (for dlatmr)
  'dlatmr.c',   # Random matrix generator with grading/pivoting/sparsity
  'dlahilb.c',  # Hilbert matrix generator
  'dlakf2.c',   # Kronecker product block matrix
  'dlatm5.c',   # Generalized Sylvester test matrices
  'dlatm6.c',   # Generalized eigenvalue test matrices
  'dlatm7.c',   # Singular value distribution
  'dlatmt.c',   # Triangular structure matrix generator

  # Right-hand side generation
  'dlarhs.c',   # Generate X and compute B = op(A) * X

  # Orthogonal random matrix generator
  'dlaror.c',   # Pre/post multiply by random orthogonal matrix (for dqrt15)
  'dlarge.c',   # Pre/post multiply by random orthogonal: A = U*A*U' (for dlatme)

  # Random number generation (LAPACK-style interface)
  'dlarnv.c',   # Vector of random numbers from uniform/normal distribution

  # QL/RQ solve helpers
  'dgeqls.c',   # Solve using QL factorization from DGEQLF
  'dgerqs.c',   # Solve using RQ factorization from DGERQF
)

# Get the main library sources from parent scope
# We need to rebuild all LAPACK routines with the test tuning implementation
# so that lapack_get_nb() calls in dgetrf.c etc. use the test version.
#
# This mirrors how LAPACK links test executables against TESTING/LIN/ilaenv.f
# instead of SRC/ilaenv.f.

# Import source file lists from src/meson.build
# Note: We exclude lapack_tuning.c since we provide lapack_tuning_test.c
src_dir = meson.project_source_root() / 'src'

# Build test utilities library with test tuning implementation
# The test tuning is compiled into libtestutils, which is linked BEFORE semilapack_dep.
# This means the test xlaenv/lapack_get_nb symbols take precedence.
libtestutils = static_library('testutils',
  testutils_sources,
  include_directories: include_directories('.'),
  dependencies: [semilapack_dep],
)

# Use link_whole to ensure lapack_tuning_test.o is always included.
# This is necessary because the strong symbols from the test tuning
# implementation must override the weak symbols in libsemilapack.
# Without link_whole, the linker won't pull in lapack_tuning_test.o
# since libsemilapack already provides (weak) definitions.
testutils_dep = declare_dependency(
  link_whole: libtestutils,
  include_directories: include_directories('.'),
  dependencies: [semilapack_dep],
)

# Keep verify_dep as alias for backward compatibility
verify_dep = testutils_dep
