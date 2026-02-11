/**
 * @file sgerqf.c
 * @brief SGERQF computes an RQ factorization of a general M-by-N matrix
 *        using a blocked algorithm.
 */

#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_single.h"

/**
 * SGERQF computes an RQ factorization of a real m by n matrix A:
 *    A = R * Q.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m      The number of rows of A. m >= 0.
 * @param[in]     n      The number of columns of A. n >= 0.
 * @param[in,out] A      On entry, the m-by-n matrix A.
 *                       On exit, if m <= n, the upper triangle of the subarray
 *                       A(0:m-1, n-m:n-1) contains the m-by-m upper triangular
 *                       matrix R; the remaining elements, with TAU, represent Q.
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
void sgerqf(const int m, const int n,
            float * const restrict A, const int lda,
            float * const restrict tau,
            float * const restrict work, const int lwork,
            int *info)
{
    int k, nb, nbmin, nx, iws, ldwork;
    int i, ib, iinfo;
    int ki, kk, mu, nu;
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
            work[0] = 1.0f;
        } else {
            nb = lapack_get_nb("GERQF");
            work[0] = (float)(m * nb);
        }
        if (!lquery && lwork < (k > 0 ? (m > 1 ? m : 1) : 1)) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("SGERQF", -(*info));
        return;
    }

    if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (k == 0) {
        return;
    }

    nb = lapack_get_nb("GERQF");
    nbmin = 2;
    nx = 1;
    iws = m;
    ldwork = m;

    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("GERQF");
        if (nx < k) {
            iws = m * nb;
            if (lwork < iws) {
                nb = lwork / m;
                nbmin = lapack_get_nbmin("GERQF");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code (backward processing) */
        ki = ((k - nx - 1) / nb) * nb;
        kk = k < ki + nb ? k : ki + nb;

        /* Process the last kk rows in blocks of nb, from bottom to top */
        for (i = k - kk + ki; i >= k - kk; i -= nb) {
            ib = (k - i) < nb ? (k - i) : nb;

            /* Factor block A(m-k+i:m-k+i+ib-1, 0:n-k+i+ib-1) */
            sgerq2(ib, n - k + i + ib,
                   &A[(m - k + i) + 0 * lda], lda,
                   &tau[i], work, &iinfo);

            if (m - k + i > 0) {
                /* Form the triangular factor of the block reflector
                 * H = H(i) H(i+1) . . . H(i+ib-1) */
                slarft("B", "R", n - k + i + ib, ib,
                       &A[(m - k + i) + 0 * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H to A(0:m-k+i-1, 0:n-k+i+ib-1) from the right */
                slarfb("R", "N", "B", "R",
                       m - k + i, n - k + i + ib, ib,
                       &A[(m - k + i) + 0 * lda], lda,
                       work, ldwork,
                       A, lda,
                       &work[ib], ldwork);
            }
        }

        /* Dimensions of the unblocked remainder */
        mu = m - kk;
        nu = n - kk;
    } else {
        mu = m;
        nu = n;
    }

    /* Use unblocked code for the remaining block */
    if (mu > 0 && nu > 0) {
        sgerq2(mu, nu, A, lda, tau, work, &iinfo);
    }

    work[0] = (float)iws;
}
