/**
 * @file dorgtsqr.c
 * @brief DORGTSQR generates an M-by-N real matrix Q_out with orthonormal columns from DLATSQR output.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DORGTSQR generates an M-by-N real matrix Q_out with orthonormal columns,
 * which are the first N columns of a product of real orthogonal
 * matrices of order M which are returned by DLATSQR.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. m >= n >= 0.
 *
 * @param[in] mb
 *          The row block size used by DLATSQR. mb > n.
 *
 * @param[in] nb
 *          The column block size used by DLATSQR. nb >= 1.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, the elements below the diagonal represent the unit
 *          lower-trapezoidal blocked matrix V computed by DLATSQR.
 *          On exit, the M-by-N orthonormal matrix Q_out.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[in] T
 *          Double precision array, dimension (ldt, n * NIRB).
 *          The upper-triangular block reflectors from DLATSQR.
 *
 * @param[in] ldt
 *          The leading dimension of the array T. ldt >= max(1, min(nb, n)).
 *
 * @param[out] work
 *          Double precision workspace of size (max(1, lwork)).
 *          On exit, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work. lwork >= (m+nb)*n.
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void dorgtsqr(
    const INT m,
    const INT n,
    const INT mb,
    const INT nb,
    f64* restrict A,
    const INT lda,
    const f64* restrict T,
    const INT ldt,
    f64* restrict work,
    const INT lwork,
    INT* info)
{
    INT lquery;
    INT iinfo, ldc, lworkopt, lc, lw, nblocal, j;
    INT minval;

    lquery = (lwork == -1);
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || m < n) {
        *info = -2;
    } else if (mb <= n) {
        *info = -3;
    } else if (nb < 1) {
        *info = -4;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -6;
    } else {
        minval = (nb < n) ? nb : n;
        if (ldt < (1 > minval ? 1 : minval)) {
            *info = -8;
        } else {

            if (lwork < 2 && !lquery) {
                *info = -10;
            } else {

                nblocal = (nb < n) ? nb : n;

                ldc = m;
                lc = ldc * n;
                lw = n * nblocal;

                lworkopt = lc + lw;

                if (lwork < (1 > lworkopt ? 1 : lworkopt) && !lquery) {
                    *info = -10;
                }
            }

        }
    }

    if (*info != 0) {
        xerbla("DORGTSQR", -(*info));
        return;
    } else if (lquery) {
        work[0] = (f64)lworkopt;
        return;
    }

    minval = (m < n) ? m : n;
    if (minval == 0) {
        work[0] = (f64)lworkopt;
        return;
    }

    dlaset("F", m, n, 0.0, 1.0, work, ldc);

    dlamtsqr("L", "N", m, n, n, mb, nblocal, A, lda, T, ldt,
             work, ldc, &work[lc], lw, &iinfo);

    for (j = 0; j < n; j++) {
        cblas_dcopy(m, &work[j * ldc], 1, &A[0 + j * lda], 1);
    }

    work[0] = (f64)lworkopt;
}
