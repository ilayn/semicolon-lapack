/**
 * @file ztgsy2.c
 * @brief ZTGSY2 solves the generalized Sylvester equation (unblocked algorithm).
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

#define LDZ 2

/**
 * ZTGSY2 solves the generalized Sylvester equation
 *
 *             A * R - L * B = scale * C               (1)
 *             D * R - L * E = scale * F
 *
 * using Level 1 and 2 BLAS, where R and L are unknown M-by-N matrices,
 * (A, D), (B, E) and (C, F) are given matrix pairs of size M-by-M,
 * N-by-N and M-by-N, respectively. A, B, D and E are upper triangular
 * (i.e., (A,D) and (B,E) in generalized Schur form).
 *
 * The solution (R, L) overwrites (C, F). 0 <= SCALE <= 1 is an output
 * scaling factor chosen to avoid overflow.
 *
 * In matrix notation solving equation (1) corresponds to solve
 * Zx = scale * b, where Z is defined as
 *
 *        Z = [ kron(In, A)  -kron(B**H, Im) ]             (2)
 *            [ kron(In, D)  -kron(E**H, Im) ],
 *
 * Ik is the identity matrix of size k and X**H is the conjugate transpose
 * of X. kron(X, Y) is the Kronecker product between the matrices X and Y.
 *
 * If TRANS = 'C', y in the conjugate transposed system Z**H*y = scale*b
 * is solved for, which is equivalent to solve for R and L in
 *
 *             A**H * R  + D**H * L   = scale * C           (3)
 *             R  * B**H + L  * E**H  = scale * -F
 *
 * This case is used to compute an estimate of Dif[(A, D), (B, E)] =
 * = sigma_min(Z) using reverse communication with ZLACON.
 *
 * ZTGSY2 also (IJOB >= 1) contributes to the computation in ZTGSYL
 * of an upper bound on the separation between to matrix pairs. Then
 * the input (A, D), (B, E) are sub-pencils of two matrix pairs in
 * ZTGSYL.
 *
 * @param[in]     trans   'N': solve the generalized Sylvester equation (1).
 *                        'C': solve the conjugate transposed system (3).
 * @param[in]     ijob    Specifies what kind of functionality to be performed.
 *                        = 0: solve (1) only.
 *                        = 1: A contribution from this subsystem to a Frobenius
 *                             norm-based estimate (look ahead strategy).
 *                        = 2: A contribution using DGECON on sub-systems.
 *                        Not referenced if TRANS = 'C'.
 * @param[in]     m       The order of A and D, and the row dimension of C, F, R and L.
 * @param[in]     n       The order of B and E, and the column dimension of C, F, R and L.
 * @param[in]     A       Array of dimension (lda, m). Upper triangular matrix.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, m).
 * @param[in]     B       Array of dimension (ldb, n). Upper triangular matrix.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] C       Array of dimension (ldc, n). On entry, the right-hand-side.
 *                        On exit, if ijob = 0, overwritten by the solution R.
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1, m).
 * @param[in]     D       Array of dimension (ldd, m). Upper triangular matrix.
 * @param[in]     ldd     The leading dimension of D. ldd >= max(1, m).
 * @param[in]     E       Array of dimension (lde, n). Upper triangular matrix.
 * @param[in]     lde     The leading dimension of E. lde >= max(1, n).
 * @param[in,out] F       Array of dimension (ldf, n). On entry, the right-hand-side.
 *                        On exit, if ijob = 0, overwritten by the solution L.
 * @param[in]     ldf     The leading dimension of F. ldf >= max(1, m).
 * @param[out]    scale   On exit, 0 <= scale <= 1. Output scaling factor.
 * @param[in,out] rdsum   On entry/exit, sum of squares for Dif-estimate.
 * @param[in,out] rdscal  On entry/exit, scaling factor for rdsum.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: The matrix pairs (A, D) and (B, E) have common or very
 *                           close eigenvalues.
 */
void ztgsy2(
    const char* trans,
    const INT ijob,
    const INT m,
    const INT n,
    const c128* restrict A,
    const INT lda,
    const c128* restrict B,
    const INT ldb,
    c128* restrict C,
    const INT ldc,
    const c128* restrict D,
    const INT ldd,
    const c128* restrict E,
    const INT lde,
    c128* restrict F,
    const INT ldf,
    f64* scale,
    f64* rdsum,
    f64* rdscal,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT notran;
    INT i, ierr, j, k;
    f64 scaloc;
    c128 alpha;

    INT ipiv[LDZ], jpiv[LDZ];
    c128 rhs[LDZ], z[LDZ * LDZ];

    *info = 0;
    ierr = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (!notran && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -1;
    } else if (notran) {
        if ((ijob < 0) || (ijob > 2)) {
            *info = -2;
        }
    }
    if (*info == 0) {
        if (m <= 0) {
            *info = -3;
        } else if (n <= 0) {
            *info = -4;
        } else if (lda < (1 > m ? 1 : m)) {
            *info = -6;
        } else if (ldb < (1 > n ? 1 : n)) {
            *info = -8;
        } else if (ldc < (1 > m ? 1 : m)) {
            *info = -10;
        } else if (ldd < (1 > m ? 1 : m)) {
            *info = -12;
        } else if (lde < (1 > n ? 1 : n)) {
            *info = -14;
        } else if (ldf < (1 > m ? 1 : m)) {
            *info = -16;
        }
    }
    if (*info != 0) {
        xerbla("ZTGSY2", -(*info));
        return;
    }

    if (notran) {

        /* Solve (I, J) - system
           A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
           D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
           for I = M, M - 1, ..., 1; J = 1, 2, ..., N */

        *scale = ONE;
        scaloc = ONE;
        for (j = 0; j < n; j++) {
            for (i = m - 1; i >= 0; i--) {

                /* Build 2 by 2 system */
                z[0 + 0 * LDZ] = A[i + i * lda];
                z[1 + 0 * LDZ] = D[i + i * ldd];
                z[0 + 1 * LDZ] = -B[j + j * ldb];
                z[1 + 1 * LDZ] = -E[j + j * lde];

                /* Set up right hand side(s) */
                rhs[0] = C[i + j * ldc];
                rhs[1] = F[i + j * ldf];

                /* Solve Z * x = RHS */
                zgetc2(LDZ, z, LDZ, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = ierr;
                }
                if (ijob == 0) {
                    zgesc2(LDZ, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                    if (scaloc != ONE) {
                        c128 scaloc_cmplx = CMPLX(scaloc, ZERO);
                        for (k = 0; k < n; k++) {
                            cblas_zscal(m, &scaloc_cmplx, &C[0 + k * ldc], 1);
                            cblas_zscal(m, &scaloc_cmplx, &F[0 + k * ldf], 1);
                        }
                        *scale = (*scale) * scaloc;
                    }
                } else {
                    zlatdf(ijob, LDZ, z, LDZ, rhs, rdsum, rdscal,
                           ipiv, jpiv);
                }

                /* Unpack solution vector(s) */
                C[i + j * ldc] = rhs[0];
                F[i + j * ldf] = rhs[1];

                /* Substitute R(I, J) and L(I, J) into remaining equation. */
                if (i > 0) {
                    alpha = -rhs[0];
                    cblas_zaxpy(i, &alpha, &A[0 + i * lda], 1, &C[0 + j * ldc], 1);
                    cblas_zaxpy(i, &alpha, &D[0 + i * ldd], 1, &F[0 + j * ldf], 1);
                }
                if (j < n - 1) {
                    cblas_zaxpy(n - j - 1, &rhs[1], &B[j + (j + 1) * ldb], ldb,
                                &C[i + (j + 1) * ldc], ldc);
                    cblas_zaxpy(n - j - 1, &rhs[1], &E[j + (j + 1) * lde], lde,
                                &F[i + (j + 1) * ldf], ldf);
                }

            }
        }
    } else {

        /* Solve transposed (I, J) - system:
           A(I, I)**H * R(I, J) + D(I, I)**H * L(J, J) = C(I, J)
           R(I, I) * B(J, J) + L(I, J) * E(J, J)   = -F(I, J)
           for I = 1, 2, ..., M, J = N, N - 1, ..., 1 */

        *scale = ONE;
        scaloc = ONE;
        for (i = 0; i < m; i++) {
            for (j = n - 1; j >= 0; j--) {

                /* Build 2 by 2 system Z**H */
                z[0 + 0 * LDZ] = conj(A[i + i * lda]);
                z[1 + 0 * LDZ] = -conj(B[j + j * ldb]);
                z[0 + 1 * LDZ] = conj(D[i + i * ldd]);
                z[1 + 1 * LDZ] = -conj(E[j + j * lde]);

                /* Set up right hand side(s) */
                rhs[0] = C[i + j * ldc];
                rhs[1] = F[i + j * ldf];

                /* Solve Z**H * x = RHS */
                zgetc2(LDZ, z, LDZ, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = ierr;
                }
                zgesc2(LDZ, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                if (scaloc != ONE) {
                    c128 scaloc_cmplx = CMPLX(scaloc, ZERO);
                    for (k = 0; k < n; k++) {
                        cblas_zscal(m, &scaloc_cmplx, &C[0 + k * ldc], 1);
                        cblas_zscal(m, &scaloc_cmplx, &F[0 + k * ldf], 1);
                    }
                    *scale = (*scale) * scaloc;
                }

                /* Unpack solution vector(s) */
                C[i + j * ldc] = rhs[0];
                F[i + j * ldf] = rhs[1];

                /* Substitute R(I, J) and L(I, J) into remaining equation. */
                for (k = 0; k < j; k++) {
                    F[i + k * ldf] = F[i + k * ldf]
                        + rhs[0] * conj(B[k + j * ldb])
                        + rhs[1] * conj(E[k + j * lde]);
                }
                for (k = i + 1; k < m; k++) {
                    C[k + j * ldc] = C[k + j * ldc]
                        - conj(A[i + k * lda]) * rhs[0]
                        - conj(D[i + k * ldd]) * rhs[1];
                }

            }
        }
    }
}

#undef LDZ
