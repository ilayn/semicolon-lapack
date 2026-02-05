/**
 * @file dlasdq.c
 * @brief DLASDQ computes the SVD of a real bidiagonal matrix with diagonal d
 *        and off-diagonal e.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>

static const double ZERO = 0.0;

/**
 * DLASDQ computes the singular value decomposition (SVD) of a real
 * (upper or lower) bidiagonal matrix with diagonal D and offdiagonal
 * E, accumulating the transformations if desired. The singular values
 * S are overwritten on D.
 *
 * @param[in]     uplo    'U' or 'u': B is upper bidiagonal.
 *                        'L' or 'l': B is lower bidiagonal.
 * @param[in]     sqre    = 0: input matrix is N-by-N.
 *                        = 1: input matrix is N-by-(N+1) if uplo='U',
 *                             or (N+1)-by-N if uplo='L'.
 * @param[in]     n       Number of rows and columns. n >= 0.
 * @param[in]     ncvt    Number of columns of VT. ncvt >= 0.
 * @param[in]     nru     Number of rows of U. nru >= 0.
 * @param[in]     ncc     Number of columns of C. ncc >= 0.
 * @param[in,out] D       Array of dimension n. Diagonal entries on entry,
 *                        singular values in ascending order on exit.
 * @param[in,out] E       Array. Offdiagonal entries on entry, zeroed on exit.
 * @param[in,out] VT      Array (ldvt, ncvt). Premultiplied by P^T.
 * @param[in]     ldvt    Leading dimension of VT. ldvt >= 1, >= n if ncvt > 0.
 * @param[in,out] U       Array (ldu, n). Postmultiplied by Q.
 * @param[in]     ldu     Leading dimension of U. ldu >= max(1, nru).
 * @param[in,out] C       Array (ldc, ncc). Premultiplied by Q^T.
 * @param[in]     ldc     Leading dimension of C. ldc >= 1, >= n if ncc > 0.
 * @param[out]    work    Array of dimension 4*n.
 * @param[out]    info    = 0: success. < 0: illegal argument. > 0: not converged.
 */
void dlasdq(const char* uplo, const int sqre, const int n, const int ncvt,
            const int nru, const int ncc, double* const restrict D,
            double* const restrict E, double* const restrict VT, const int ldvt,
            double* const restrict U, const int ldu,
            double* const restrict C, const int ldc,
            double* const restrict work, int* info)
{
    int rotate;
    int i, isub, iuplo, j, np1, sqre1;
    double cs, r, smin, sn;

    /* Test the input parameters */
    *info = 0;
    iuplo = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        iuplo = 1;
    }
    if (uplo[0] == 'L' || uplo[0] == 'l') {
        iuplo = 2;
    }
    if (iuplo == 0) {
        *info = -1;
    } else if (sqre < 0 || sqre > 1) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ncvt < 0) {
        *info = -4;
    } else if (nru < 0) {
        *info = -5;
    } else if (ncc < 0) {
        *info = -6;
    } else if ((ncvt == 0 && ldvt < 1) || (ncvt > 0 && ldvt < (1 > n ? 1 : n))) {
        *info = -10;
    } else if (ldu < (1 > nru ? 1 : nru)) {
        *info = -12;
    } else if ((ncc == 0 && ldc < 1) || (ncc > 0 && ldc < (1 > n ? 1 : n))) {
        *info = -14;
    }
    if (*info != 0) {
        xerbla("DLASDQ", -(*info));
        return;
    }
    if (n == 0) {
        return;
    }

    /* ROTATE is true if any singular vectors desired, false otherwise */
    rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);
    np1 = n + 1;
    sqre1 = sqre;

    /* If matrix non-square upper bidiagonal, rotate to be lower
     * bidiagonal. The rotations are on the right. */
    if (iuplo == 1 && sqre1 == 1) {
        for (i = 0; i < n - 1; i++) {
            dlartg(D[i], E[i], &cs, &sn, &r);
            D[i] = r;
            E[i] = sn * D[i + 1];
            D[i + 1] = cs * D[i + 1];
            if (rotate) {
                work[i] = cs;
                work[n + i] = sn;
            }
        }
        dlartg(D[n - 1], E[n - 1], &cs, &sn, &r);
        D[n - 1] = r;
        E[n - 1] = ZERO;
        if (rotate) {
            work[n - 1] = cs;
            work[n + n - 1] = sn;
        }
        iuplo = 2;
        sqre1 = 0;

        /* Update singular vectors if desired */
        if (ncvt > 0) {
            dlasr("L", "V", "F", np1, ncvt, &work[0], &work[np1 - 1], VT, ldvt);
        }
    }

    /* If matrix lower bidiagonal, rotate to be upper bidiagonal
     * by applying Givens rotations on the left. */
    if (iuplo == 2) {
        for (i = 0; i < n - 1; i++) {
            dlartg(D[i], E[i], &cs, &sn, &r);
            D[i] = r;
            E[i] = sn * D[i + 1];
            D[i + 1] = cs * D[i + 1];
            if (rotate) {
                work[i] = cs;
                work[n + i] = sn;
            }
        }

        /* If matrix (N+1)-by-N lower bidiagonal, one additional rotation needed */
        if (sqre1 == 1) {
            dlartg(D[n - 1], E[n - 1], &cs, &sn, &r);
            D[n - 1] = r;
            if (rotate) {
                work[n - 1] = cs;
                work[n + n - 1] = sn;
            }
        }

        /* Update singular vectors if desired */
        if (nru > 0) {
            if (sqre1 == 0) {
                dlasr("R", "V", "F", nru, n, &work[0], &work[np1 - 1], U, ldu);
            } else {
                dlasr("R", "V", "F", nru, np1, &work[0], &work[np1 - 1], U, ldu);
            }
        }
        if (ncc > 0) {
            if (sqre1 == 0) {
                dlasr("L", "V", "F", n, ncc, &work[0], &work[np1 - 1], C, ldc);
            } else {
                dlasr("L", "V", "F", np1, ncc, &work[0], &work[np1 - 1], C, ldc);
            }
        }
    }

    /* Call DBDSQR to compute the SVD of the reduced real
     * N-by-N upper bidiagonal matrix. */
    dbdsqr("U", n, ncvt, nru, ncc, D, E, VT, ldvt, U, ldu, C, ldc, work, info);

    /* Sort the singular values into ascending order (insertion sort on
     * singular values, but only one transposition per singular vector) */
    for (i = 0; i < n; i++) {
        /* Scan for smallest D[i] */
        isub = i;
        smin = D[i];
        for (j = i + 1; j < n; j++) {
            if (D[j] < smin) {
                isub = j;
                smin = D[j];
            }
        }
        if (isub != i) {
            /* Swap singular values and vectors */
            D[isub] = D[i];
            D[i] = smin;
            if (ncvt > 0) {
                cblas_dswap(ncvt, &VT[isub + 0 * ldvt], ldvt, &VT[i + 0 * ldvt], ldvt);
            }
            if (nru > 0) {
                cblas_dswap(nru, &U[0 + isub * ldu], 1, &U[0 + i * ldu], 1);
            }
            if (ncc > 0) {
                cblas_dswap(ncc, &C[isub + 0 * ldc], ldc, &C[i + 0 * ldc], ldc);
            }
        }
    }
}
