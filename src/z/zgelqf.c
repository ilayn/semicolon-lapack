/**
 * @file zgelqf.c
 * @brief ZGELQF computes an LQ factorization of a general M-by-N matrix
 *        using a blocked algorithm.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZGELQF computes an LQ factorization of a complex m by n matrix A:
 *    A = L * Q.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m      The number of rows of A. m >= 0.
 * @param[in]     n      The number of columns of A. n >= 0.
 * @param[in,out] A      On entry, the m-by-n matrix A.
 *                       On exit, the elements on and below the diagonal contain
 *                       the m-by-min(m,n) lower trapezoidal matrix L; the elements
 *                       above the diagonal, with TAU, represent Q.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau    Array of dimension min(m, n).
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work. lwork >= max(1, m).
 *                       For optimal performance, lwork >= m*nb.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: success; < 0: -i means i-th argument was illegal.
 */
void zgelqf(const INT m, const INT n,
            c128* restrict A, const INT lda,
            c128* restrict tau,
            c128* restrict work, const INT lwork,
            INT* info)
{
    INT k, nb, nbmin, nx, iws, ldwork;
    INT i, ib, iinfo;
    INT lquery;

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
            work[0] = 1.0;
        } else {
            nb = lapack_get_nb("GELQF");
            work[0] = (c128)(m * nb);
        }
        if (!lquery && lwork < (k > 0 ? (m > 1 ? m : 1) : 1)) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("ZGELQF", -(*info));
        return;
    }

    if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (k == 0) {
        return;
    }

    nb = lapack_get_nb("GELQF");
    nbmin = 2;
    nx = 0;
    iws = m;
    ldwork = m;

    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("GELQF");
        if (nx < k) {
            iws = m * nb;
            if (lwork < iws) {
                nb = lwork / m;
                nbmin = lapack_get_nbmin("GELQF");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code */
        for (i = 0; i < k - nx; i += nb) {
            ib = (k - i) < nb ? (k - i) : nb;

            /* Factor block A(i:i+ib-1, i:n-1) */
            zgelq2(ib, n - i, &A[i + i * lda], lda, &tau[i], work, &iinfo);

            if (i + ib < m) {
                /* Form the triangular factor of the block reflector
                 * H = H(i) H(i+1) . . . H(i+ib-1) */
                zlarft("F", "R", n - i, ib,
                       &A[i + i * lda], lda, &tau[i], work, ldwork);

                /* Apply H to A(i+ib:m-1, i:n-1) from the right */
                zlarfb("R", "N", "F", "R",
                       m - i - ib, n - i, ib,
                       &A[i + i * lda], lda,
                       work, ldwork,
                       &A[(i + ib) + i * lda], lda,
                       &work[ib], ldwork);
            }
        }
    } else {
        i = 0;
    }

    /* Use unblocked code for the remaining rows */
    if (i < k) {
        zgelq2(m - i, n - i, &A[i + i * lda], lda, &tau[i], work, &iinfo);
    }

    work[0] = (c128)iws;
}
