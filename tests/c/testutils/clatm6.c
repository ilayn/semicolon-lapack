/**
 * @file clatm6.c
 * @brief CLATM6 generates test matrices for the generalized eigenvalue problem.
 *
 * Faithful port of LAPACK TESTING/MATGEN/clatm6.f
 */

#include <math.h>
#include <complex.h>
#include "verify.h"

/**
 * CLATM6 generates test matrices for the generalized eigenvalue
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
void clatm6(const INT type, const INT n,
            c64* A, const INT lda, c64* B,
            c64* X, const INT ldx, c64* Y, const INT ldy,
            const c64 alpha, const c64 beta,
            const c64 wx, const c64 wy,
            f32* S, f32* DIF)
{
    const f32 RONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    INT i, info, j;
    f32 rwork[50];
    c64 work[26];
    c64 Z[8 * 8];

    /* Generate test problem ...
     * (Da, Db) ... */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i + i * lda] = CMPLXF((f32)(i + 1), 0.0f) + alpha;
                B[i + i * lda] = ONE;
            } else {
                A[i + j * lda] = ZERO;
                B[i + j * lda] = ZERO;
            }
        }
    }
    if (type == 2) {
        A[0 + 0 * lda] = CMPLXF(RONE, RONE);
        A[1 + 1 * lda] = conjf(A[0 + 0 * lda]);
        A[2 + 2 * lda] = ONE;
        A[3 + 3 * lda] = CMPLXF(crealf(ONE + alpha), crealf(ONE + beta));
        A[4 + 4 * lda] = conjf(A[3 + 3 * lda]);
    }

    /* Form X and Y */
    clacpy("F", n, n, B, lda, Y, ldy);
    Y[2 + 0 * ldy] = -conjf(wy);
    Y[3 + 0 * ldy] = conjf(wy);
    Y[4 + 0 * ldy] = -conjf(wy);
    Y[2 + 1 * ldy] = -conjf(wy);
    Y[3 + 1 * ldy] = conjf(wy);
    Y[4 + 1 * ldy] = -conjf(wy);

    clacpy("F", n, n, B, lda, X, ldx);
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
    S[0] = RONE / sqrtf((RONE + THREE * cabsf(wy) * cabsf(wy)) /
                       (RONE + cabsf(A[0 + 0 * lda]) * cabsf(A[0 + 0 * lda])));
    S[1] = RONE / sqrtf((RONE + THREE * cabsf(wy) * cabsf(wy)) /
                       (RONE + cabsf(A[1 + 1 * lda]) * cabsf(A[1 + 1 * lda])));
    S[2] = RONE / sqrtf((RONE + TWO * cabsf(wx) * cabsf(wx)) /
                       (RONE + cabsf(A[2 + 2 * lda]) * cabsf(A[2 + 2 * lda])));
    S[3] = RONE / sqrtf((RONE + TWO * cabsf(wx) * cabsf(wx)) /
                       (RONE + cabsf(A[3 + 3 * lda]) * cabsf(A[3 + 3 * lda])));
    S[4] = RONE / sqrtf((RONE + TWO * cabsf(wx) * cabsf(wx)) /
                       (RONE + cabsf(A[4 + 4 * lda]) * cabsf(A[4 + 4 * lda])));

    clakf2(1, 4, A, lda, &A[1 + 1 * lda], B, &B[1 + 1 * lda], Z, 8);
    cgesvd("N", "N", 8, 8, Z, 8, rwork, work, 1, &work[1], 1,
           &work[2], 24, &rwork[8], &info);
    DIF[0] = rwork[7];

    clakf2(4, 1, A, lda, &A[4 + 4 * lda], B, &B[4 + 4 * lda], Z, 8);
    cgesvd("N", "N", 8, 8, Z, 8, rwork, work, 1, &work[1], 1,
           &work[2], 24, &rwork[8], &info);
    DIF[4] = rwork[7];
}
