/**
 * @file zlatm6.c
 * @brief ZLATM6 generates test matrices for the generalized eigenvalue problem.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlatm6.f
 */

#include <math.h>
#include <complex.h>
#include "verify.h"

/* Forward declaration for test utility */
void zlakf2(const INT m, const INT n,
            const c128* A, const INT lda, const c128* B,
            const c128* D, const c128* E,
            c128* Z, const INT ldz);

/**
 * ZLATM6 generates test matrices for the generalized eigenvalue
 * problem, their corresponding right and left eigenvector matrices,
 * and also reciprocal condition numbers for all eigenvalues and
 * the reciprocal condition numbers of eigenvectors corresponding to
 * the 1th and 5th eigenvalues.
 *
 * @param[in] type   Specifies the problem type (see further details).
 * @param[in] n      Size of the matrices A and B.
 * @param[out] A     Complex array, dimension (lda, n).
 * @param[in] lda    The leading dimension of A and of B.
 * @param[out] B     Complex array, dimension (lda, n).
 * @param[out] X     Complex array, dimension (ldx, n).
 * @param[in] ldx    The leading dimension of X.
 * @param[out] Y     Complex array, dimension (ldy, n).
 * @param[in] ldy    The leading dimension of Y.
 * @param[in] alpha  Weighting constant for matrix A.
 * @param[in] beta   Weighting constant for matrix A.
 * @param[in] wx     Constant for right eigenvector matrix.
 * @param[in] wy     Constant for left eigenvector matrix.
 * @param[out] S     Double precision array, dimension (n).
 * @param[out] DIF   Double precision array, dimension (n).
 */
void zlatm6(const INT type, const INT n,
            c128* A, const INT lda, c128* B,
            c128* X, const INT ldx, c128* Y, const INT ldy,
            const c128 alpha, const c128 beta,
            const c128 wx, const c128 wy,
            f64* S, f64* DIF)
{
    const f64 RONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;
    const c128 ZERO = CMPLX(0.0, 0.0);
    const c128 ONE = CMPLX(1.0, 0.0);

    INT i, info, j;
    f64 rwork[50];
    c128 work[26];
    c128 Z[8 * 8];

    /* Generate test problem ...
     * (Da, Db) ... */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i + i * lda] = CMPLX((f64)(i + 1), 0.0) + alpha;
                B[i + i * lda] = ONE;
            } else {
                A[i + j * lda] = ZERO;
                B[i + j * lda] = ZERO;
            }
        }
    }
    if (type == 2) {
        A[0 + 0 * lda] = CMPLX(RONE, RONE);
        A[1 + 1 * lda] = conj(A[0 + 0 * lda]);
        A[2 + 2 * lda] = ONE;
        A[3 + 3 * lda] = CMPLX(creal(ONE + alpha), creal(ONE + beta));
        A[4 + 4 * lda] = conj(A[3 + 3 * lda]);
    }

    /* Form X and Y */
    zlacpy("F", n, n, B, lda, Y, ldy);
    Y[2 + 0 * ldy] = -conj(wy);
    Y[3 + 0 * ldy] = conj(wy);
    Y[4 + 0 * ldy] = -conj(wy);
    Y[2 + 1 * ldy] = -conj(wy);
    Y[3 + 1 * ldy] = conj(wy);
    Y[4 + 1 * ldy] = -conj(wy);

    zlacpy("F", n, n, B, lda, X, ldx);
    X[0 + 2 * ldx] = -wx;
    X[0 + 3 * ldx] = -wx;
    X[0 + 4 * ldx] = wx;
    X[1 + 2 * ldx] = wx;
    X[1 + 3 * ldx] = -wx;
    X[1 + 4 * ldx] = -wx;

    /* Form (A, B) */
    B[0 + 2 * lda] = wx + wy;
    B[1 + 2 * lda] = -wx + wy;
    B[0 + 3 * lda] = wx - wy;
    B[1 + 3 * lda] = wx - wy;
    B[0 + 4 * lda] = -wx + wy;
    B[1 + 4 * lda] = wx + wy;
    A[0 + 2 * lda] = wx * A[0 + 0 * lda] + wy * A[2 + 2 * lda];
    A[1 + 2 * lda] = -wx * A[1 + 1 * lda] + wy * A[2 + 2 * lda];
    A[0 + 3 * lda] = wx * A[0 + 0 * lda] - wy * A[3 + 3 * lda];
    A[1 + 3 * lda] = wx * A[1 + 1 * lda] - wy * A[3 + 3 * lda];
    A[0 + 4 * lda] = -wx * A[0 + 0 * lda] + wy * A[4 + 4 * lda];
    A[1 + 4 * lda] = wx * A[1 + 1 * lda] + wy * A[4 + 4 * lda];

    /* Compute condition numbers */
    S[0] = RONE / sqrt((RONE + THREE * cabs(wy) * cabs(wy)) /
                       (RONE + cabs(A[0 + 0 * lda]) * cabs(A[0 + 0 * lda])));
    S[1] = RONE / sqrt((RONE + THREE * cabs(wy) * cabs(wy)) /
                       (RONE + cabs(A[1 + 1 * lda]) * cabs(A[1 + 1 * lda])));
    S[2] = RONE / sqrt((RONE + TWO * cabs(wx) * cabs(wx)) /
                       (RONE + cabs(A[2 + 2 * lda]) * cabs(A[2 + 2 * lda])));
    S[3] = RONE / sqrt((RONE + TWO * cabs(wx) * cabs(wx)) /
                       (RONE + cabs(A[3 + 3 * lda]) * cabs(A[3 + 3 * lda])));
    S[4] = RONE / sqrt((RONE + TWO * cabs(wx) * cabs(wx)) /
                       (RONE + cabs(A[4 + 4 * lda]) * cabs(A[4 + 4 * lda])));

    zlakf2(1, 4, A, lda, &A[1 + 1 * lda], B, &B[1 + 1 * lda], Z, 8);
    zgesvd("N", "N", 8, 8, Z, 8, rwork, work, 1, &work[1], 1,
           &work[2], 24, &rwork[8], &info);
    DIF[0] = rwork[7];

    zlakf2(4, 1, A, lda, &A[4 + 4 * lda], B, &B[4 + 4 * lda], Z, 8);
    zgesvd("N", "N", 8, 8, Z, 8, rwork, work, 1, &work[1], 1,
           &work[2], 24, &rwork[8], &info);
    DIF[4] = rwork[7];
}
