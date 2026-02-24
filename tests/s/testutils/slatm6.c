/**
 * @file slatm6.c
 * @brief SLATM6 generates test matrices for the generalized eigenvalue problem.
 */

#include <math.h>
#include "verify.h"

/**
 * SLATM6 generates test matrices for the generalized eigenvalue
 * problem, their corresponding right and left eigenvector matrices,
 * and also reciprocal condition numbers for all eigenvalues and
 * the reciprocal condition numbers of eigenvectors corresponding to
 * the 1th and 5th eigenvalues.
 *
 * Test Matrices
 *
 * Two kinds of test matrix pairs
 *
 *       (A, B) = inverse(YH) * (Da, Db) * inverse(X)
 *
 * are used in the tests:
 *
 * Type 1:
 *    Da = 1+a   0    0    0    0    Db = 1   0   0   0   0
 *          0   2+a   0    0    0         0   1   0   0   0
 *          0    0   3+a   0    0         0   0   1   0   0
 *          0    0    0   4+a   0         0   0   0   1   0
 *          0    0    0    0   5+a ,      0   0   0   0   1 , and
 *
 * Type 2:
 *    Da =  1   -1    0    0    0    Db = 1   0   0   0   0
 *          1    1    0    0    0         0   1   0   0   0
 *          0    0    1    0    0         0   0   1   0   0
 *          0    0    0   1+a  1+b        0   0   0   1   0
 *          0    0    0  -1-b  1+a ,      0   0   0   0   1 .
 *
 * In both cases the same inverse(YH) and inverse(X) are used to compute
 * (A, B), giving the exact eigenvectors to (A,B) as (YH, X):
 *
 * YH:  =  1    0   -y    y   -y    X =  1   0  -x  -x   x
 *         0    1   -y    y   -y         0   1   x  -x  -x
 *         0    0    1    0    0         0   0   1   0   0
 *         0    0    0    1    0         0   0   0   1   0
 *         0    0    0    0    1,        0   0   0   0   1 ,
 *
 * where a, b, x and y will have all values independently of each other.
 *
 * @param[in] type
 *     Specifies the problem type (see further details).
 *
 * @param[in] n
 *     Size of the matrices A and B.
 *
 * @param[out] A
 *     Double precision array, dimension (lda, n).
 *     On exit A N-by-N is initialized according to TYPE.
 *
 * @param[in] lda
 *     The leading dimension of A and of B.
 *
 * @param[out] B
 *     Double precision array, dimension (lda, n).
 *     On exit B N-by-N is initialized according to TYPE.
 *
 * @param[out] X
 *     Double precision array, dimension (ldx, n).
 *     On exit X is the N-by-N matrix of right eigenvectors.
 *
 * @param[in] ldx
 *     The leading dimension of X.
 *
 * @param[out] Y
 *     Double precision array, dimension (ldy, n).
 *     On exit Y is the N-by-N matrix of left eigenvectors.
 *
 * @param[in] ldy
 *     The leading dimension of Y.
 *
 * @param[in] alpha
 *     Weighting constant for matrix A.
 *
 * @param[in] beta
 *     Weighting constant for matrix A.
 *
 * @param[in] wx
 *     Constant for right eigenvector matrix.
 *
 * @param[in] wy
 *     Constant for left eigenvector matrix.
 *
 * @param[out] S
 *     Double precision array, dimension (n).
 *     S(i) is the reciprocal condition number for eigenvalue i.
 *
 * @param[out] DIF
 *     Double precision array, dimension (n).
 *     DIF(i) is the reciprocal condition number for eigenvector i.
 */
void slatm6(const INT type, const INT n,
            f32* A, const INT lda, f32* B,
            f32* X, const INT ldx, f32* Y, const INT ldy,
            const f32 alpha, const f32 beta,
            const f32 wx, const f32 wy,
            f32* S, f32* DIF)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    INT i, info, j;
    f32 work[100];
    f32 Z[12 * 12];

    /* Generate test problem ...
     * (Da, Db) ... */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i + i * lda] = (f32)(i + 1) + alpha;
                B[i + i * lda] = ONE;
            } else {
                A[i + j * lda] = ZERO;
                B[i + j * lda] = ZERO;
            }
        }
    }

    /* Form X and Y */
    slacpy("F", n, n, B, lda, Y, ldy);
    /* Fortran indices: Y(3,1), Y(4,1), Y(5,1), Y(3,2), Y(4,2), Y(5,2)
     * C 0-based: Y[2,0], Y[3,0], Y[4,0], Y[2,1], Y[3,1], Y[4,1] */
    Y[2 + 0 * ldy] = -wy;
    Y[3 + 0 * ldy] = wy;
    Y[4 + 0 * ldy] = -wy;
    Y[2 + 1 * ldy] = -wy;
    Y[3 + 1 * ldy] = wy;
    Y[4 + 1 * ldy] = -wy;

    slacpy("F", n, n, B, lda, X, ldx);
    /* Fortran indices: X(1,3), X(1,4), X(1,5), X(2,3), X(2,4), X(2,5)
     * C 0-based: X[0,2], X[0,3], X[0,4], X[1,2], X[1,3], X[1,4] */
    X[0 + 2 * ldx] = -wx;
    X[0 + 3 * ldx] = -wx;
    X[0 + 4 * ldx] = wx;
    X[1 + 2 * ldx] = wx;
    X[1 + 3 * ldx] = -wx;
    X[1 + 4 * ldx] = -wx;

    /* Form (A, B) */
    /* Fortran: B(1,3), B(2,3), B(1,4), B(2,4), B(1,5), B(2,5)
     * C: B[0,2], B[1,2], B[0,3], B[1,3], B[0,4], B[1,4] */
    B[0 + 2 * lda] = wx + wy;
    B[1 + 2 * lda] = -wx + wy;
    B[0 + 3 * lda] = wx - wy;
    B[1 + 3 * lda] = wx - wy;
    B[0 + 4 * lda] = -wx + wy;
    B[1 + 4 * lda] = wx + wy;

    if (type == 1) {
        /* A(1,3) = WX*A(1,1) + WY*A(3,3) etc.
         * C: A[0,2] = wx*A[0,0] + wy*A[2,2] etc. */
        A[0 + 2 * lda] = wx * A[0 + 0 * lda] + wy * A[2 + 2 * lda];
        A[1 + 2 * lda] = -wx * A[1 + 1 * lda] + wy * A[2 + 2 * lda];
        A[0 + 3 * lda] = wx * A[0 + 0 * lda] - wy * A[3 + 3 * lda];
        A[1 + 3 * lda] = wx * A[1 + 1 * lda] - wy * A[3 + 3 * lda];
        A[0 + 4 * lda] = -wx * A[0 + 0 * lda] + wy * A[4 + 4 * lda];
        A[1 + 4 * lda] = wx * A[1 + 1 * lda] + wy * A[4 + 4 * lda];
    } else if (type == 2) {
        A[0 + 2 * lda] = TWO * wx + wy;
        A[1 + 2 * lda] = wy;
        A[0 + 3 * lda] = -wy * (TWO + alpha + beta);
        A[1 + 3 * lda] = TWO * wx - wy * (TWO + alpha + beta);
        A[0 + 4 * lda] = -TWO * wx + wy * (alpha - beta);
        A[1 + 4 * lda] = wy * (alpha - beta);
        A[0 + 0 * lda] = ONE;
        A[0 + 1 * lda] = -ONE;
        A[1 + 0 * lda] = ONE;
        A[1 + 1 * lda] = A[0 + 0 * lda];
        A[2 + 2 * lda] = ONE;
        A[3 + 3 * lda] = ONE + alpha;
        A[3 + 4 * lda] = ONE + beta;
        A[4 + 3 * lda] = -A[3 + 4 * lda];
        A[4 + 4 * lda] = A[3 + 3 * lda];
    }

    /* Compute condition numbers */
    if (type == 1) {
        /* S(1) = 1 / sqrt((1 + 3*WY^2) / (1 + A(1,1)^2))
         * C: S[0] = ... A[0,0]^2 */
        S[0] = ONE / sqrtf((ONE + THREE * wy * wy) /
                          (ONE + A[0 + 0 * lda] * A[0 + 0 * lda]));
        S[1] = ONE / sqrtf((ONE + THREE * wy * wy) /
                          (ONE + A[1 + 1 * lda] * A[1 + 1 * lda]));
        S[2] = ONE / sqrtf((ONE + TWO * wx * wx) /
                          (ONE + A[2 + 2 * lda] * A[2 + 2 * lda]));
        S[3] = ONE / sqrtf((ONE + TWO * wx * wx) /
                          (ONE + A[3 + 3 * lda] * A[3 + 3 * lda]));
        S[4] = ONE / sqrtf((ONE + TWO * wx * wx) /
                          (ONE + A[4 + 4 * lda] * A[4 + 4 * lda]));

        /* CALL SLAKF2(1, 4, A, LDA, A(2,2), B, B(2,2), Z, 12)
         * C: slakf2(1, 4, A, lda, &A[1+1*lda], B, &B[1+1*lda], Z, 12) */
        slakf2(1, 4, A, lda, &A[1 + 1 * lda], B, &B[1 + 1 * lda], Z, 12);
        sgesvd("N", "N", 8, 8, Z, 12, work, &work[8], 1, &work[9], 1,
               &work[10], 40, &info);
        DIF[0] = work[7];  /* Fortran WORK(8), C work[7] */

        /* CALL SLAKF2(4, 1, A, LDA, A(5,5), B, B(5,5), Z, 12)
         * C: slakf2(4, 1, A, lda, &A[4+4*lda], B, &B[4+4*lda], Z, 12) */
        slakf2(4, 1, A, lda, &A[4 + 4 * lda], B, &B[4 + 4 * lda], Z, 12);
        sgesvd("N", "N", 8, 8, Z, 12, work, &work[8], 1, &work[9], 1,
               &work[10], 40, &info);
        DIF[4] = work[7];  /* Fortran WORK(8), C work[7] */

    } else if (type == 2) {
        S[0] = ONE / sqrtf(ONE / THREE + wy * wy);
        S[1] = S[0];
        S[2] = ONE / sqrtf(ONE / TWO + wx * wx);
        S[3] = ONE / sqrtf((ONE + TWO * wx * wx) /
                          (ONE + (ONE + alpha) * (ONE + alpha) +
                           (ONE + beta) * (ONE + beta)));
        S[4] = S[3];

        /* CALL SLAKF2(2, 3, A, LDA, A(3,3), B, B(3,3), Z, 12)
         * C: slakf2(2, 3, A, lda, &A[2+2*lda], B, &B[2+2*lda], Z, 12) */
        slakf2(2, 3, A, lda, &A[2 + 2 * lda], B, &B[2 + 2 * lda], Z, 12);
        sgesvd("N", "N", 12, 12, Z, 12, work, &work[12], 1, &work[13], 1,
               &work[14], 60, &info);
        DIF[0] = work[11];  /* Fortran WORK(12), C work[11] */

        /* CALL SLAKF2(3, 2, A, LDA, A(4,4), B, B(4,4), Z, 12)
         * C: slakf2(3, 2, A, lda, &A[3+3*lda], B, &B[3+3*lda], Z, 12) */
        slakf2(3, 2, A, lda, &A[3 + 3 * lda], B, &B[3 + 3 * lda], Z, 12);
        sgesvd("N", "N", 12, 12, Z, 12, work, &work[12], 1, &work[13], 1,
               &work[14], 60, &info);
        DIF[4] = work[11];  /* Fortran WORK(12), C work[11] */
    }
}
