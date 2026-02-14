/**
 * @file slagv2.c
 * @brief SLAGV2 computes the Generalized Schur factorization of a real 2-by-2 matrix pencil.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAGV2 computes the Generalized Schur factorization of a real 2-by-2
 * matrix pencil (A,B) where B is upper triangular. This routine
 * computes orthogonal (rotation) matrices given by CSL, SNL and CSR,
 * SNR such that
 *
 * 1) if the pencil (A,B) has two real eigenvalues (include 0/0 or 1/0
 *    types), then
 *
 *    [ a11 a12 ] := [  CSL  SNL ] [ a11 a12 ] [  CSR -SNR ]
 *    [  0  a22 ]    [ -SNL  CSL ] [ a21 a22 ] [  SNR  CSR ]
 *
 *    [ b11 b12 ] := [  CSL  SNL ] [ b11 b12 ] [  CSR -SNR ]
 *    [  0  b22 ]    [ -SNL  CSL ] [  0  b22 ] [  SNR  CSR ],
 *
 * 2) if the pencil (A,B) has a pair of complex conjugate eigenvalues,
 *    then
 *
 *    [ a11 a12 ] := [  CSL  SNL ] [ a11 a12 ] [  CSR -SNR ]
 *    [ a21 a22 ]    [ -SNL  CSL ] [ a21 a22 ] [  SNR  CSR ]
 *
 *    [ b11  0  ] := [  CSL  SNL ] [ b11 b12 ] [  CSR -SNR ]
 *    [  0  b22 ]    [ -SNL  CSL ] [  0  b22 ] [  SNR  CSR ]
 *
 *    where b11 >= b22 > 0.
 *
 * @param[in,out] A       Array of dimension (lda, 2). On entry, the 2x2 matrix A.
 *                        On exit, overwritten by the "A-part" of the generalized Schur form.
 * @param[in]     lda     The leading dimension of A. lda >= 2.
 * @param[in,out] B       Array of dimension (ldb, 2). On entry, the upper triangular 2x2 matrix B.
 *                        On exit, overwritten by the "B-part" of the generalized Schur form.
 * @param[in]     ldb     The leading dimension of B. ldb >= 2.
 * @param[out]    alphar  Array of dimension (2). Real parts of eigenvalue numerators.
 * @param[out]    alphai  Array of dimension (2). Imaginary parts of eigenvalue numerators.
 * @param[out]    beta    Array of dimension (2). Eigenvalue denominators.
 * @param[out]    csl     The cosine of the left rotation matrix.
 * @param[out]    snl     The sine of the left rotation matrix.
 * @param[out]    csr     The cosine of the right rotation matrix.
 * @param[out]    snr     The sine of the right rotation matrix.
 */
void slagv2(
    f32* const restrict A,
    const int lda,
    f32* const restrict B,
    const int ldb,
    f32* const restrict alphar,
    f32* const restrict alphai,
    f32* const restrict beta,
    f32* csl,
    f32* snl,
    f32* csr,
    f32* snr)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    f32 anorm, ascale, bnorm, bscale;
    f32 h1, h2, h3, qq, r, rr, safmin, scale1, scale2, t, ulp, wi, wr1, wr2;

    safmin = slamch("S");
    ulp = slamch("P");

    /* Scale A */
    anorm = fmaxf(fmaxf(fabsf(A[0 + 0 * lda]) + fabsf(A[1 + 0 * lda]),
                      fabsf(A[0 + 1 * lda]) + fabsf(A[1 + 1 * lda])), safmin);
    ascale = ONE / anorm;
    A[0 + 0 * lda] = ascale * A[0 + 0 * lda];
    A[0 + 1 * lda] = ascale * A[0 + 1 * lda];
    A[1 + 0 * lda] = ascale * A[1 + 0 * lda];
    A[1 + 1 * lda] = ascale * A[1 + 1 * lda];

    /* Scale B */
    bnorm = fmaxf(fmaxf(fabsf(B[0 + 0 * ldb]),
                      fabsf(B[0 + 1 * ldb]) + fabsf(B[1 + 1 * ldb])), safmin);
    bscale = ONE / bnorm;
    B[0 + 0 * ldb] = bscale * B[0 + 0 * ldb];
    B[0 + 1 * ldb] = bscale * B[0 + 1 * ldb];
    B[1 + 1 * ldb] = bscale * B[1 + 1 * ldb];

    /* Check if A can be deflated */
    if (fabsf(A[1 + 0 * lda]) <= ulp) {
        *csl = ONE;
        *snl = ZERO;
        *csr = ONE;
        *snr = ZERO;
        A[1 + 0 * lda] = ZERO;
        B[1 + 0 * ldb] = ZERO;
        wi = ZERO;

    /* Check if B is singular */
    } else if (fabsf(B[0 + 0 * ldb]) <= ulp) {
        slartg(A[0 + 0 * lda], A[1 + 0 * lda], csl, snl, &r);
        *csr = ONE;
        *snr = ZERO;
        cblas_srot(2, &A[0 + 0 * lda], lda, &A[1 + 0 * lda], lda, *csl, *snl);
        cblas_srot(2, &B[0 + 0 * ldb], ldb, &B[1 + 0 * ldb], ldb, *csl, *snl);
        A[1 + 0 * lda] = ZERO;
        B[0 + 0 * ldb] = ZERO;
        B[1 + 0 * ldb] = ZERO;
        wi = ZERO;

    } else if (fabsf(B[1 + 1 * ldb]) <= ulp) {
        slartg(A[1 + 1 * lda], A[1 + 0 * lda], csr, snr, &t);
        *snr = -(*snr);
        cblas_srot(2, &A[0 + 0 * lda], 1, &A[0 + 1 * lda], 1, *csr, *snr);
        cblas_srot(2, &B[0 + 0 * ldb], 1, &B[0 + 1 * ldb], 1, *csr, *snr);
        *csl = ONE;
        *snl = ZERO;
        A[1 + 0 * lda] = ZERO;
        B[1 + 0 * ldb] = ZERO;
        B[1 + 1 * ldb] = ZERO;
        wi = ZERO;

    } else {
        /* B is nonsingular, first compute the eigenvalues of (A,B) */
        slag2(A, lda, B, ldb, safmin, &scale1, &scale2, &wr1, &wr2, &wi);

        if (wi == ZERO) {
            /* two real eigenvalues, compute s*A-w*B */
            h1 = scale1 * A[0 + 0 * lda] - wr1 * B[0 + 0 * ldb];
            h2 = scale1 * A[0 + 1 * lda] - wr1 * B[0 + 1 * ldb];
            h3 = scale1 * A[1 + 1 * lda] - wr1 * B[1 + 1 * ldb];

            rr = slapy2(h1, h2);
            qq = slapy2(scale1 * A[1 + 0 * lda], h3);

            if (rr > qq) {
                /* find right rotation matrix to zero 1,1 element of (sA - wB) */
                slartg(h2, h1, csr, snr, &t);
            } else {
                /* find right rotation matrix to zero 2,1 element of (sA - wB) */
                slartg(h3, scale1 * A[1 + 0 * lda], csr, snr, &t);
            }

            *snr = -(*snr);
            cblas_srot(2, &A[0 + 0 * lda], 1, &A[0 + 1 * lda], 1, *csr, *snr);
            cblas_srot(2, &B[0 + 0 * ldb], 1, &B[0 + 1 * ldb], 1, *csr, *snr);

            /* compute inf norms of A and B */
            h1 = fmaxf(fabsf(A[0 + 0 * lda]) + fabsf(A[0 + 1 * lda]),
                      fabsf(A[1 + 0 * lda]) + fabsf(A[1 + 1 * lda]));
            h2 = fmaxf(fabsf(B[0 + 0 * ldb]) + fabsf(B[0 + 1 * ldb]),
                      fabsf(B[1 + 0 * ldb]) + fabsf(B[1 + 1 * ldb]));

            if ((scale1 * h1) >= fabsf(wr1) * h2) {
                /* find left rotation matrix Q to zero out B(2,1) */
                slartg(B[0 + 0 * ldb], B[1 + 0 * ldb], csl, snl, &r);
            } else {
                /* find left rotation matrix Q to zero out A(2,1) */
                slartg(A[0 + 0 * lda], A[1 + 0 * lda], csl, snl, &r);
            }

            cblas_srot(2, &A[0 + 0 * lda], lda, &A[1 + 0 * lda], lda, *csl, *snl);
            cblas_srot(2, &B[0 + 0 * ldb], ldb, &B[1 + 0 * ldb], ldb, *csl, *snl);

            A[1 + 0 * lda] = ZERO;
            B[1 + 0 * ldb] = ZERO;

        } else {
            /* a pair of complex conjugate eigenvalues
               first compute the SVD of the matrix B */
            slasv2(B[0 + 0 * ldb], B[0 + 1 * ldb], B[1 + 1 * ldb],
                   &r, &t, snr, csr, snl, csl);

            /* Form (A,B) := Q(A,B)Z**T where Q is left rotation matrix and
               Z is right rotation matrix computed from SLASV2 */
            cblas_srot(2, &A[0 + 0 * lda], lda, &A[1 + 0 * lda], lda, *csl, *snl);
            cblas_srot(2, &B[0 + 0 * ldb], ldb, &B[1 + 0 * ldb], ldb, *csl, *snl);
            cblas_srot(2, &A[0 + 0 * lda], 1, &A[0 + 1 * lda], 1, *csr, *snr);
            cblas_srot(2, &B[0 + 0 * ldb], 1, &B[0 + 1 * ldb], 1, *csr, *snr);

            B[1 + 0 * ldb] = ZERO;
            B[0 + 1 * ldb] = ZERO;
        }
    }

    /* Unscaling */
    A[0 + 0 * lda] = anorm * A[0 + 0 * lda];
    A[1 + 0 * lda] = anorm * A[1 + 0 * lda];
    A[0 + 1 * lda] = anorm * A[0 + 1 * lda];
    A[1 + 1 * lda] = anorm * A[1 + 1 * lda];
    B[0 + 0 * ldb] = bnorm * B[0 + 0 * ldb];
    B[1 + 0 * ldb] = bnorm * B[1 + 0 * ldb];
    B[0 + 1 * ldb] = bnorm * B[0 + 1 * ldb];
    B[1 + 1 * ldb] = bnorm * B[1 + 1 * ldb];

    if (wi == ZERO) {
        alphar[0] = A[0 + 0 * lda];
        alphar[1] = A[1 + 1 * lda];
        alphai[0] = ZERO;
        alphai[1] = ZERO;
        beta[0] = B[0 + 0 * ldb];
        beta[1] = B[1 + 1 * ldb];
    } else {
        alphar[0] = anorm * wr1 / scale1 / bnorm;
        alphai[0] = anorm * wi / scale1 / bnorm;
        alphar[1] = alphar[0];
        alphai[1] = -alphai[0];
        beta[0] = ONE;
        beta[1] = ONE;
    }
}
