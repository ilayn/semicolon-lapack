/**
 * @file slaror.c
 * @brief SLAROR pre- or post-multiplies an M by N matrix A by a random
 *        orthogonal matrix U, overwriting A.
 *
 * Faithful port of LAPACK TESTING/MATGEN/slaror.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 *
 * U is generated using the method of G.W. Stewart
 * (SIAM J. Numer. Anal. 17, 1980, 403-409).
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"


/**
 * SLAROR pre- or post-multiplies an M by N matrix A by a random
 * orthogonal matrix U, overwriting A. A may optionally be initialized
 * to the identity matrix before multiplying by U.
 *
 * @param[in] side
 *     Specifies whether A is multiplied on the left or right by U.
 *     = 'L': Multiply A on the left (premultiply) by U
 *     = 'R': Multiply A on the right (postmultiply) by U'
 *     = 'C' or 'T': Multiply A on the left by U and the right by U'
 *
 * @param[in] init
 *     Specifies whether or not A should be initialized to the identity matrix.
 *     = 'I': Initialize A to (a section of) the identity matrix before applying U.
 *     = 'N': No initialization. Apply U to the input matrix A.
 *
 * @param[in] m
 *     The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *     The number of columns of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *     On entry, the array A.
 *     On exit, overwritten by U*A (if side='L'), or by A*U (if side='R'),
 *     or by U*A*U' (if side='C' or 'T').
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] X
 *     Workspace array of dimension (3*max(m, n)).
 *     2*m + n if side = 'L',
 *     2*n + m if side = 'R',
 *     3*n     if side = 'C' or 'T'.
 *
 * @param[out] info
 *     = 0: normal return
 *     < 0: if info = -k, the k-th argument had an illegal value
 *     = 1: if the random numbers generated are bad.
 *
 * @param[in,out] state
 *     RNG state array of 4 uint64_t elements, passed through from caller.
 */
void slaror(const char* side, const char* init,
            const INT m, const INT n,
            f32* A, const INT lda,
            f32* X, INT* info,
            uint64_t state[static 4])
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TOOSML = 1.0e-20f;

    INT irow, itype, ixfrm, j, jcol, kbeg, nxfrm;
    f32 factor, xnorm, xnorms;

    *info = 0;
    if (n == 0 || m == 0) {
        return;
    }

    /* Determine type of multiplication */
    itype = 0;
    if (side[0] == 'L' || side[0] == 'l') {
        itype = 1;
    } else if (side[0] == 'R' || side[0] == 'r') {
        itype = 2;
    } else if (side[0] == 'C' || side[0] == 'c' ||
               side[0] == 'T' || side[0] == 't') {
        itype = 3;
    }

    /* Check for argument errors */
    if (itype == 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0 || (itype == 3 && n != m)) {
        *info = -4;
    } else if (lda < m) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("SLAROR", -(*info));
        return;
    }

    if (itype == 1) {
        nxfrm = m;
    } else {
        nxfrm = n;
    }

    /* Initialize A to the identity matrix if desired */
    if (init[0] == 'I' || init[0] == 'i') {
        slaset("F", m, n, ZERO, ONE, A, lda);
    }

    /* Initialize X */
    for (j = 0; j < nxfrm; j++) {
        X[j] = ZERO;
    }

    /* Compute rotation by computing Householder transformations
       H(2), H(3), ..., H(nxfrm) */
    for (ixfrm = 2; ixfrm <= nxfrm; ixfrm++) {
        kbeg = nxfrm - ixfrm;  /* 0-based: nxfrm - ixfrm + 1 - 1 */

        /* Generate independent normal(0, 1) random numbers */
        for (j = kbeg; j < nxfrm; j++) {
            X[j] = rng_normal_f32(state);
        }

        /* Generate a Householder transformation from the random vector X */
        xnorm = cblas_snrm2(ixfrm, &X[kbeg], 1);
        xnorms = (X[kbeg] >= 0.0f) ? xnorm : -xnorm;
        X[kbeg + nxfrm] = (X[kbeg] >= 0.0f) ? -ONE : ONE;
        factor = xnorms * (xnorms + X[kbeg]);
        if (fabsf(factor) < TOOSML) {
            *info = 1;
            xerbla("SLAROR", *info);
            return;
        }
        factor = ONE / factor;
        X[kbeg] = X[kbeg] + xnorms;

        /* Apply Householder transformation to A */
        if (itype == 1 || itype == 3) {
            /* Apply H(k) from the left */
            cblas_sgemv(CblasColMajor, CblasTrans, ixfrm, n, ONE,
                        &A[kbeg + 0 * lda], lda, &X[kbeg], 1,
                        ZERO, &X[2 * nxfrm], 1);
            cblas_sger(CblasColMajor, ixfrm, n, -factor, &X[kbeg], 1,
                       &X[2 * nxfrm], 1, &A[kbeg + 0 * lda], lda);
        }

        if (itype == 2 || itype == 3) {
            /* Apply H(k) from the right */
            cblas_sgemv(CblasColMajor, CblasNoTrans, m, ixfrm, ONE,
                        &A[0 + kbeg * lda], lda, &X[kbeg], 1,
                        ZERO, &X[2 * nxfrm], 1);
            cblas_sger(CblasColMajor, m, ixfrm, -factor, &X[2 * nxfrm], 1,
                       &X[kbeg], 1, &A[0 + kbeg * lda], lda);
        }
    }

    X[2 * nxfrm - 1] = (rng_normal_f32(state) >= 0.0f) ? ONE : -ONE;

    /* Scale the matrix A by D */
    if (itype == 1 || itype == 3) {
        for (irow = 0; irow < m; irow++) {
            cblas_sscal(n, X[nxfrm + irow], &A[irow + 0 * lda], lda);
        }
    }

    if (itype == 2 || itype == 3) {
        for (jcol = 0; jcol < n; jcol++) {
            cblas_sscal(m, X[nxfrm + jcol], &A[0 + jcol * lda], 1);
        }
    }
}
