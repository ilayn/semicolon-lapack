/**
 * @file zlaror.c
 * @brief ZLAROR pre- or post-multiplies an M by N matrix A by a random
 *        unitary matrix U, overwriting A.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlaror.f
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
 * ZLAROR pre- or post-multiplies an M by N matrix A by a random
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
void zlaror(const char* side, const char* init,
            const INT m, const INT n,
            c128* A, const INT lda,
            c128* X, INT* info,
            uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TOOSML = 1.0e-20;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT irow, itype, ixfrm, j, jcol, kbeg, nxfrm;
    f64 factor, xabs, xnorm;
    c128 csign, xnorms;

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
        xerbla("ZLAROR", -(*info));
        return;
    }

    if (itype == 1) {
        nxfrm = m;
    } else {
        nxfrm = n;
    }

    /* Initialize A to the identity matrix if desired */
    if (init[0] == 'I' || init[0] == 'i') {
        zlaset("F", m, n, CZERO, CONE, A, lda);
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
            X[j] = zlarnd_rng(3, state);
        }

        /* Generate a Householder transformation from the random vector X */
        xnorm = cblas_dznrm2(ixfrm, &X[kbeg], 1);
        xabs = cabs(X[kbeg]);
        if (xabs != ZERO) {
            csign = X[kbeg] / xabs;
        } else {
            csign = CONE;
        }
        xnorms = csign * xnorm;
        X[nxfrm + kbeg] = -csign;
        factor = xnorm * (xnorm + xabs);
        if (fabs(factor) < TOOSML) {
            *info = 1;
            xerbla("ZLAROR", -(*info));
            return;
        }
        factor = ONE / factor;
        X[kbeg] = X[kbeg] + xnorms;

        /* Apply Householder transformation to A */
        if (itype == 1 || itype == 3 || itype == 4) {
            /* Apply H(k) on the left of A */
            c128 neg_factor = CMPLX(-factor, 0.0);
            cblas_zgemv(CblasColMajor, CblasConjTrans, ixfrm, n, &CONE,
                        &A[kbeg], lda, &X[kbeg], 1, &CZERO, &X[2 * nxfrm], 1);
            cblas_zgerc(CblasColMajor, ixfrm, n, &neg_factor, &X[kbeg], 1,
                        &X[2 * nxfrm], 1, &A[kbeg], lda);
        }

        if (itype >= 2 && itype <= 4) {
            /* Apply H(k)* (or H(k)') on the right of A */
            if (itype == 4) {
                zlacgv(ixfrm, &X[kbeg], 1);
            }

            c128 neg_factor = CMPLX(-factor, 0.0);
            cblas_zgemv(CblasColMajor, CblasNoTrans, m, ixfrm, &CONE,
                        &A[kbeg * lda], lda, &X[kbeg], 1, &CZERO,
                        &X[2 * nxfrm], 1);
            cblas_zgerc(CblasColMajor, m, ixfrm, &neg_factor,
                        &X[2 * nxfrm], 1, &X[kbeg], 1, &A[kbeg * lda], lda);
        }
    }

    X[0] = zlarnd_rng(3, state);
    xabs = cabs(X[0]);
    if (xabs != ZERO) {
        csign = X[0] / xabs;
    } else {
        csign = CONE;
    }
    X[2 * nxfrm - 1] = csign;

    /* Scale the matrix A by D */
    if (itype == 1 || itype == 3 || itype == 4) {
        for (irow = 0; irow < m; irow++) {
            c128 s = conj(X[nxfrm + irow]);
            cblas_zscal(n, &s, &A[irow], lda);
        }
    }

    if (itype == 2 || itype == 3) {
        for (jcol = 0; jcol < n; jcol++) {
            cblas_zscal(m, &X[nxfrm + jcol], &A[jcol * lda], 1);
        }
    }

    if (itype == 4) {
        for (jcol = 0; jcol < n; jcol++) {
            c128 s = conj(X[nxfrm + jcol]);
            cblas_zscal(m, &s, &A[jcol * lda], 1);
        }
    }
}
