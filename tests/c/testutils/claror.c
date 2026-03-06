/**
 * @file claror.c
 * @brief CLAROR pre- or post-multiplies an M by N matrix A by a random
 *        unitary matrix U, overwriting A.
 *
 * Faithful port of LAPACK TESTING/MATGEN/claror.f
 *
 * U is generated using the method of G.W. Stewart
 * (SIAM J. Numer. Anal. 17, 1980, 403-409).
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/**
 * CLAROR pre- or post-multiplies an M by N matrix A by a random
 * unitary matrix U, overwriting A. A may optionally be initialized
 * to the identity matrix before multiplying by U.
 *
 * @param[in] side
 *     = 'L': Multiply A on the left (premultiply) by U
 *     = 'R': Multiply A on the right (postmultiply) by U*
 *     = 'C': Multiply A on the left by U and the right by U*
 *     = 'T': Multiply A on the left by U and the right by U'
 *
 * @param[in] init
 *     = 'I': Initialize A to the identity matrix before applying U.
 *     = 'N': No initialization.
 *
 * @param[in] m       Number of rows of A. m >= 0.
 * @param[in] n       Number of columns of A. n >= 0.
 * @param[in,out] A   Complex array, dimension (lda, n).
 * @param[in] lda     The leading dimension of A. lda >= max(1, m).
 * @param[out] X      Complex workspace of dimension (3*max(m, n)).
 * @param[out] info   = 0: normal return; = 1: bad random numbers.
 * @param[in,out] state  RNG state array.
 */
void claror(const char* side, const char* init,
            const INT m, const INT n,
            c64* A, const INT lda,
            c64* X, INT* info,
            uint64_t state[static 4])
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TOOSML = 1.0e-20f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT irow, itype, ixfrm, j, jcol, kbeg, nxfrm;
    f32 factor, xabs, xnorm;
    c64 csign, xnorms;

    *info = 0;
    if (n == 0 || m == 0) {
        return;
    }

    itype = 0;
    if (side[0] == 'L' || side[0] == 'l') {
        itype = 1;
    } else if (side[0] == 'R' || side[0] == 'r') {
        itype = 2;
    } else if (side[0] == 'C' || side[0] == 'c') {
        itype = 3;
    } else if (side[0] == 'T' || side[0] == 't') {
        itype = 4;
    }

    /* Check for argument errors */
    if (itype == 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0 || ((itype == 3 || itype == 4) && n != m)) {
        *info = -4;
    } else if (lda < m) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("CLAROR", -(*info));
        return;
    }

    if (itype == 1) {
        nxfrm = m;
    } else {
        nxfrm = n;
    }

    /* Initialize A to the identity matrix if desired */
    if (init[0] == 'I' || init[0] == 'i') {
        claset("F", m, n, CZERO, CONE, A, lda);
    }

    /* Initialize X */
    for (j = 0; j < nxfrm; j++) {
        X[j] = CZERO;
    }

    /* Compute rotation by computing Householder transformations
       H(2), H(3), ..., H(nxfrm) */
    for (ixfrm = 2; ixfrm <= nxfrm; ixfrm++) {
        kbeg = nxfrm - ixfrm;

        /* Generate independent normal(0, 1) random numbers */
        for (j = kbeg; j < nxfrm; j++) {
            X[j] = clarnd_rng(3, state);
        }

        /* Generate a Householder transformation from the random vector X */
        xnorm = cblas_scnrm2(ixfrm, &X[kbeg], 1);
        xabs = cabsf(X[kbeg]);
        if (xabs != ZERO) {
            csign = X[kbeg] / xabs;
        } else {
            csign = CONE;
        }
        xnorms = csign * xnorm;
        X[nxfrm + kbeg] = -csign;
        factor = xnorm * (xnorm + xabs);
        if (fabsf(factor) < TOOSML) {
            *info = 1;
            xerbla("CLAROR", -(*info));
            return;
        }
        factor = ONE / factor;
        X[kbeg] = X[kbeg] + xnorms;

        /* Apply Householder transformation to A */
        if (itype == 1 || itype == 3 || itype == 4) {
            /* Apply H(k) on the left of A */
            c64 neg_factor = CMPLXF(-factor, 0.0f);
            cblas_cgemv(CblasColMajor, CblasConjTrans, ixfrm, n, &CONE,
                        &A[kbeg], lda, &X[kbeg], 1, &CZERO, &X[2 * nxfrm], 1);
            cblas_cgerc(CblasColMajor, ixfrm, n, &neg_factor, &X[kbeg], 1,
                        &X[2 * nxfrm], 1, &A[kbeg], lda);
        }

        if (itype >= 2 && itype <= 4) {
            /* Apply H(k)* (or H(k)') on the right of A */
            if (itype == 4) {
                clacgv(ixfrm, &X[kbeg], 1);
            }

            c64 neg_factor = CMPLXF(-factor, 0.0f);
            cblas_cgemv(CblasColMajor, CblasNoTrans, m, ixfrm, &CONE,
                        &A[kbeg * lda], lda, &X[kbeg], 1, &CZERO,
                        &X[2 * nxfrm], 1);
            cblas_cgerc(CblasColMajor, m, ixfrm, &neg_factor,
                        &X[2 * nxfrm], 1, &X[kbeg], 1, &A[kbeg * lda], lda);
        }
    }

    X[0] = clarnd_rng(3, state);
    xabs = cabsf(X[0]);
    if (xabs != ZERO) {
        csign = X[0] / xabs;
    } else {
        csign = CONE;
    }
    X[2 * nxfrm - 1] = csign;

    /* Scale the matrix A by D */
    if (itype == 1 || itype == 3 || itype == 4) {
        for (irow = 0; irow < m; irow++) {
            c64 s = conjf(X[nxfrm + irow]);
            cblas_cscal(n, &s, &A[irow], lda);
        }
    }

    if (itype == 2 || itype == 3) {
        for (jcol = 0; jcol < n; jcol++) {
            cblas_cscal(m, &X[nxfrm + jcol], &A[jcol * lda], 1);
        }
    }

    if (itype == 4) {
        for (jcol = 0; jcol < n; jcol++) {
            c64 s = conjf(X[nxfrm + jcol]);
            cblas_cscal(m, &s, &A[jcol * lda], 1);
        }
    }
}
