/**
 * @file cungrq.c
 * @brief CUNGRQ generates all or part of the unitary matrix Q from
 *        an RQ factorization determined by CGERQF (blocked algorithm).
 */

#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CUNGRQ generates an M-by-N complex matrix Q with orthonormal rows,
 * which is defined as the last M rows of a product of K elementary
 * reflectors of order N
 *
 *    Q = H(0)**H H(1)**H . . . H(k-1)**H
 *
 * as returned by CGERQF.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m      The number of rows of Q. m >= 0.
 * @param[in]     n      The number of columns of Q. n >= m.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines Q. m >= k >= 0.
 * @param[in,out] A      On entry, the (m-k+i)-th row must contain the vector
 *                       which defines the elementary reflector H(i), for
 *                       i = 0,...,k-1, as returned by CGERQF.
 *                       On exit, the m-by-n matrix Q.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by CGERQF.
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work. lwork >= max(1, m).
 *                       For optimal performance, lwork >= m*nb.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cungrq(const int m, const int n, const int k,
            c64* restrict A, const int lda,
            const c64* restrict tau,
            c64* restrict work, const int lwork,
            int* info)
{
    int nb, nbmin, nx, iws, ldwork, lwkopt;
    int i, ib, ii, iinfo, j, l;
    int kk;
    int lquery;
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    /* Parameter validation */
    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < m) {
        *info = -2;
    } else if (k < 0 || k > m) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    }

    if (*info == 0) {
        if (m <= 0) {
            lwkopt = 1;
        } else {
            nb = lapack_get_nb("ORGRQ");
            lwkopt = m * nb;
        }
        work[0] = (c64)lwkopt;

        if (lwork < (m > 1 ? m : 1) && !lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        xerbla("CUNGRQ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m <= 0) {
        return;
    }

    nbmin = 2;
    nx = 0;
    iws = m;
    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("ORGRQ");
        if (nx < k) {
            ldwork = m;
            iws = ldwork * nb;
            if (lwork < iws) {
                nb = lwork / ldwork;
                nbmin = lapack_get_nbmin("ORGRQ");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code after the first block.
         * The last kk rows are handled by the block method. */
        kk = k < ((k - nx + nb - 1) / nb) * nb ? k : ((k - nx + nb - 1) / nb) * nb;

        /* Set A(0:m-kk-1, n-kk:n-1) to zero */
        for (j = n - kk; j < n; j++) {
            for (i = 0; i < m - kk; i++) {
                A[i + j * lda] = ZERO;
            }
        }
    } else {
        kk = 0;
    }

    /* Use unblocked code for the first or only block */
    cungr2(m - kk, n - kk, k - kk, A, lda, tau, work, &iinfo);

    if (kk > 0) {
        /* Use blocked code */
        ldwork = m;
        for (i = k - kk; i < k; i += nb) {
            ib = nb < k - i ? nb : k - i;
            ii = m - k + i;
            if (ii > 0) {
                /* Form the triangular factor of the block reflector
                 * H = H(i+ib-1) . . . H(i+1) H(i) */
                clarft("B", "R", n - k + i + ib, ib,
                       &A[ii + 0 * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H**H to A(0:ii-1, 0:n-k+i+ib-1) from the right */
                clarfb("R", "C", "B", "R",
                       ii, n - k + i + ib, ib,
                       &A[ii + 0 * lda], lda,
                       work, ldwork,
                       A, lda,
                       &work[ib], ldwork);
            }

            /* Apply H**H to columns 0:n-k+i+ib-1 of current block */
            cungr2(ib, n - k + i + ib, ib,
                   &A[ii + 0 * lda], lda,
                   &tau[i], work, &iinfo);

            /* Set columns n-k+i+ib:n-1 of current block to zero */
            for (l = n - k + i + ib; l < n; l++) {
                for (j = ii; j < ii + ib; j++) {
                    A[j + l * lda] = ZERO;
                }
            }
        }
    }

    work[0] = (c64)iws;
}
