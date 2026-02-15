/**
 * @file cgeqlf.c
 * @brief CGEQLF computes a QL factorization of a general M-by-N matrix
 *        using a blocked algorithm.
 */

#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGEQLF computes a QL factorization of a complex m by n matrix A:
 *    A = Q * L.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m      The number of rows of A. m >= 0.
 * @param[in]     n      The number of columns of A. n >= 0.
 * @param[in,out] A      On entry, the m-by-n matrix A.
 *                       On exit, if m >= n, the lower triangle of the subarray
 *                       A(m-n:m-1, 0:n-1) contains the n-by-n lower triangular
 *                       matrix L; the remaining elements, with TAU, represent Q.
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
void cgeqlf(const int m, const int n,
            c64* restrict A, const int lda,
            c64* restrict tau,
            c64* restrict work, const int lwork,
            int* info)
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
            nb = lapack_get_nb("GEQLF");
            work[0] = (c64)(n * nb);
        }
        if (!lquery && lwork < (k > 0 ? (n > 1 ? n : 1) : 1)) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("CGEQLF", -(*info));
        return;
    }

    if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (k == 0) {
        return;
    }

    nb = lapack_get_nb("GEQLF");
    nbmin = 2;
    nx = 1;
    iws = n;
    ldwork = n;

    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("GEQLF");
        if (nx < k) {
            iws = n * nb;
            if (lwork < iws) {
                nb = lwork / n;
                nbmin = lapack_get_nbmin("GEQLF");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code initially.
         * The last kk columns are handled by the block method. */
        ki = ((k - nx - 1) / nb) * nb;
        kk = k < ki + nb ? k : ki + nb;

        for (i = k - kk + ki; i >= k - kk; i -= nb) {
            ib = (k - i) < nb ? (k - i) : nb;

            /* Compute the QL factorization of the current block
             * A(0:m-k+i+ib-1, n-k+i:n-k+i+ib-1) */
            cgeql2(m - k + i + ib, ib,
                   &A[0 + (n - k + i) * lda], lda,
                   &tau[i], work, &iinfo);

            if (n - k + i > 0) {
                /* Form the triangular factor of the block reflector
                 * H = H(i+ib-1) . . . H(i+1) H(i) */
                clarft("B", "C", m - k + i + ib, ib,
                       &A[0 + (n - k + i) * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H^H to A(0:m-k+i+ib-1, 0:n-k+i-1) from the left */
                clarfb("L", "C", "B", "C",
                       m - k + i + ib, n - k + i, ib,
                       &A[0 + (n - k + i) * lda], lda,
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

    /* Use unblocked code to factor the last or only block. */
    if (mu > 0 && nu > 0) {
        cgeql2(mu, nu, A, lda, tau, work, &iinfo);
    }

    work[0] = (c64)iws;
}
