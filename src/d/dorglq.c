/**
 * @file dorglq.c
 * @brief DORGLQ generates all or part of the orthogonal matrix Q from
 *        an LQ factorization determined by DGELQF (blocked algorithm).
 */

#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_double.h"

/**
 * DORGLQ generates an M-by-N real matrix Q with orthonormal rows,
 * which is defined as the first M rows of a product of K elementary
 * reflectors of order N
 *
 *    Q = H(k-1) . . . H(1) H(0)
 *
 * as returned by DGELQF.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m      The number of rows of Q. m >= 0.
 * @param[in]     n      The number of columns of Q. n >= m.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines Q. m >= k >= 0.
 * @param[in,out] A      On entry, the i-th row must contain the vector which
 *                       defines the elementary reflector H(i), for
 *                       i = 0,1,...,k-1, as returned by DGELQF.
 *                       On exit, the m-by-n matrix Q.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by DGELQF.
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work. lwork >= max(1, m).
 *                       For optimal performance, lwork >= m*nb.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value.
 */
void dorglq(const int m, const int n, const int k,
            double * const restrict A, const int lda,
            const double * const restrict tau,
            double * const restrict work, const int lwork,
            int *info)
{
    int nb, nbmin, nx, iws, ldwork, lwkopt;
    int i, ib, iinfo, j, l;
    int ki, kk;
    int lquery;
    const double ZERO = 0.0;

    /* Parameter validation */
    *info = 0;
    nb = lapack_get_nb("ORGLQ");
    lwkopt = (m > 1 ? m : 1) * nb;
    work[0] = (double)lwkopt;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < m) {
        *info = -2;
    } else if (k < 0 || k > m) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (lwork < (m > 1 ? m : 1) && !lquery) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("DORGLQ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m <= 0) {
        work[0] = 1.0;
        return;
    }

    nbmin = 2;
    nx = 0;
    iws = m;
    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("ORGLQ");
        if (nx < k) {
            ldwork = m;
            iws = ldwork * nb;
            if (lwork < iws) {
                nb = lwork / ldwork;
                nbmin = lapack_get_nbmin("ORGLQ");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code after the last block.
         * The first kk rows are handled by the block method. */
        ki = ((k - nx - 1) / nb) * nb;
        kk = k < ki + nb ? k : ki + nb;

        /* Set A(kk:m-1, 0:kk-1) to zero */
        for (j = 0; j < kk; j++) {
            for (i = kk; i < m; i++) {
                A[i + j * lda] = ZERO;
            }
        }
    } else {
        kk = 0;
    }

    /* Use unblocked code for the last or only block */
    if (kk < m) {
        dorgl2(m - kk, n - kk, k - kk,
               &A[kk + kk * lda], lda,
               &tau[kk], work, &iinfo);
    }

    if (kk > 0) {
        /* Use blocked code */
        ldwork = m;
        for (i = ki; i >= 0; i -= nb) {
            ib = nb < k - i ? nb : k - i;
            if (i + ib < m) {
                /* Form the triangular factor of the block reflector
                 * H = H(i) H(i+1) . . . H(i+ib-1) */
                dlarft("F", "R", n - i, ib,
                       &A[i + i * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H^T to A(i+ib:m-1, i:n-1) from the right */
                dlarfb("R", "T", "F", "R",
                       m - i - ib, n - i, ib,
                       &A[i + i * lda], lda,
                       work, ldwork,
                       &A[(i + ib) + i * lda], lda,
                       &work[ib], ldwork);
            }

            /* Apply H^T to columns i:n-1 of current block */
            dorgl2(ib, n - i, ib, &A[i + i * lda], lda,
                   &tau[i], work, &iinfo);

            /* Set columns 0:i-1 of current block to zero */
            for (j = 0; j < i; j++) {
                for (l = i; l < i + ib; l++) {
                    A[l + j * lda] = ZERO;
                }
            }
        }
    }

    work[0] = (double)iws;
}
