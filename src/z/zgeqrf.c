/**
 * @file zgeqrf.c
 * @brief ZGEQRF computes a QR factorization of a general M-by-N matrix
 *        using a blocked algorithm.
 */

#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZGEQRF computes a QR factorization of a complex m by n matrix A:
 *    A = Q * R.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m      The number of rows of A. m >= 0.
 * @param[in]     n      The number of columns of A. n >= 0.
 * @param[in,out] A      On entry, the m-by-n matrix A.
 *                       On exit, the elements on and above the diagonal contain
 *                       the min(m,n)-by-n upper trapezoidal matrix R; the elements
 *                       below the diagonal, with TAU, represent Q.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau    Array of dimension min(m, n).
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work. lwork >= max(1, n).
 *                       For optimal performance, lwork >= n*nb.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: success; < 0: -i means i-th argument was illegal.
 */
void zgeqrf(const int m, const int n,
            c128* restrict A, const int lda,
            c128* restrict tau,
            c128* restrict work, const int lwork,
            int* info)
{
    int k, nb, nbmin, nx, iws, ldwork;
    int i, ib, iinfo;
    int lquery;

    /* Parameter validation */
    *info = 0;
    k = m < n ? m : n;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }

    if (*info == 0) {
        if (k == 0) {
            work[0] = CMPLX(1.0, 0.0);
        } else {
            nb = lapack_get_nb("GEQRF");
            work[0] = CMPLX((f64)(n * nb), 0.0);
        }
        if (!lquery && lwork < (k > 0 ? (n > 1 ? n : 1) : 1)) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("ZGEQRF", -(*info));
        return;
    }

    if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (k == 0) {
        return;
    }

    nb = lapack_get_nb("GEQRF");
    nbmin = 2;
    nx = 0;
    iws = n;
    ldwork = n;

    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("GEQRF");
        if (nx < k) {
            /* Determine if workspace is large enough for blocked code */
            iws = n * nb;
            if (lwork < iws) {
                /* Not enough workspace: reduce nb */
                nb = lwork / n;
                nbmin = lapack_get_nbmin("GEQRF");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code */
        for (i = 0; i < k - nx; i += nb) {
            ib = (k - i) < nb ? (k - i) : nb;

            /* Factor block A(i:m-1, i:i+ib-1) */
            zgeqr2(m - i, ib, &A[i + i * lda], lda, &tau[i], work, &iinfo);

            if (i + ib < n) {
                /* Form the triangular factor of the block reflector
                 * H = H(i) H(i+1) . . . H(i+ib-1) */
                zlarft("F", "C", m - i, ib,
                       &A[i + i * lda], lda, &tau[i], work, ldwork);

                /* Apply H^H to A(i:m-1, i+ib:n-1) from the left */
                zlarfb("L", "C", "F", "C",
                       m - i, n - i - ib, ib,
                       &A[i + i * lda], lda,
                       work, ldwork,
                       &A[i + (i + ib) * lda], lda,
                       &work[ib], ldwork);
            }
        }
    } else {
        i = 0;
    }

    /* Use unblocked code for the remaining columns */
    if (i < k) {
        zgeqr2(m - i, n - i, &A[i + i * lda], lda, &tau[i], work, &iinfo);
    }

    work[0] = CMPLX((f64)iws, 0.0);
}
