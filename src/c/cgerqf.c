/**
 * @file cgerqf.c
 * @brief CGERQF computes an RQ factorization of a general M-by-N matrix
 *        using a blocked algorithm.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGERQF computes an RQ factorization of a complex m by n matrix A:
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
void cgerqf(const INT m, const INT n,
            c64* restrict A, const INT lda,
            c64* restrict tau,
            c64* restrict work, const INT lwork,
            INT* info)
{
    INT k, nb, nbmin, nx, iws, ldwork;
    INT i, ib, iinfo;
    INT ki, kk, mu, nu;
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
            work[0] = 1.0f;
        } else {
            nb = lapack_get_nb("GERQF");
            work[0] = (c64)(m * nb);
        }
        if (!lquery && lwork < (k > 0 ? (m > 1 ? m : 1) : 1)) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("CGERQF", -(*info));
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
            cgerq2(ib, n - k + i + ib,
                   &A[(m - k + i) + 0 * lda], lda,
                   &tau[i], work, &iinfo);

            if (m - k + i > 0) {
                /* Form the triangular factor of the block reflector
                 * H = H(i) H(i+1) . . . H(i+ib-1) */
                clarft("B", "R", n - k + i + ib, ib,
                       &A[(m - k + i) + 0 * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H to A(0:m-k+i-1, 0:n-k+i+ib-1) from the right */
                clarfb("R", "N", "B", "R",
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
        cgerq2(mu, nu, A, lda, tau, work, &iinfo);
    }

    work[0] = (c64)iws;
}
