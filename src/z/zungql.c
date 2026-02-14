/**
 * @file zungql.c
 * @brief ZUNGQL generates all or part of the unitary matrix Q from
 *        a QL factorization determined by ZGEQLF (blocked algorithm).
 */

#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZUNGQL generates an M-by-N complex matrix Q with orthonormal columns,
 * which is defined as the last N columns of a product of K elementary
 * reflectors of order M
 *
 *    Q = H(k-1) . . . H(1) H(0)
 *
 * as returned by ZGEQLF.
 *
 * @param[in]     m      The number of rows of Q. m >= 0.
 * @param[in]     n      The number of columns of Q. m >= n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines Q. n >= k >= 0.
 * @param[in,out] A      On entry, the (n-k+i)-th column must contain the
 *                       vector which defines the elementary reflector H(i),
 *                       for i = 0,...,k-1, as returned by ZGEQLF.
 *                       On exit, the m-by-n matrix Q.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by ZGEQLF.
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work. lwork >= max(1, n).
 *                       For optimal performance, lwork >= n*nb.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zungql(const int m, const int n, const int k,
            double complex* const restrict A, const int lda,
            const double complex* const restrict tau,
            double complex* const restrict work, const int lwork,
            int* info)
{
    int nb, nbmin, nx, iws, ldwork, lwkopt;
    int i, ib, iinfo, j, l;
    int kk;
    int lquery;
    const double complex ZERO = CMPLX(0.0, 0.0);

    /* Parameter validation */
    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > m) {
        *info = -2;
    } else if (k < 0 || k > n) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    }

    if (*info == 0) {
        if (n == 0) {
            lwkopt = 1;
        } else {
            nb = lapack_get_nb("ORGQL");
            lwkopt = n * nb;
        }
        work[0] = (double complex)lwkopt;

        if (lwork < (n > 1 ? n : 1) && !lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        xerbla("ZUNGQL", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    nbmin = 2;
    nx = 0;
    iws = n;
    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("ORGQL");
        if (nx < k) {
            ldwork = n;
            iws = ldwork * nb;
            if (lwork < iws) {
                nb = lwork / ldwork;
                nbmin = lapack_get_nbmin("ORGQL");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code after the first block.
         * The last kk columns are handled by the block method. */
        kk = k < ((k - nx + nb - 1) / nb) * nb ? k : ((k - nx + nb - 1) / nb) * nb;

        /* Set A(m-kk:m-1, 0:n-kk-1) to zero */
        for (j = 0; j < n - kk; j++) {
            for (i = m - kk; i < m; i++) {
                A[i + j * lda] = ZERO;
            }
        }
    } else {
        kk = 0;
    }

    /* Use unblocked code for the first or only block */
    zung2l(m - kk, n - kk, k - kk, A, lda, tau, work, &iinfo);

    if (kk > 0) {
        /* Use blocked code */
        ldwork = n;
        for (i = k - kk; i < k; i += nb) {
            ib = nb < k - i ? nb : k - i;
            if (n - k + i > 0) {
                /* Form the triangular factor of the block reflector
                 * H = H(i+ib-1) . . . H(i+1) H(i) */
                zlarft("B", "C", m - k + i + ib, ib,
                       &A[0 + (n - k + i) * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H to A(0:m-k+i+ib-1, 0:n-k+i-1) from the left */
                zlarfb("L", "N", "B", "C",
                       m - k + i + ib, n - k + i, ib,
                       &A[0 + (n - k + i) * lda], lda,
                       work, ldwork,
                       A, lda,
                       &work[ib], ldwork);
            }

            /* Apply H to rows 0:m-k+i+ib-1 of current block */
            zung2l(m - k + i + ib, ib, ib,
                   &A[0 + (n - k + i) * lda], lda,
                   &tau[i], work, &iinfo);

            /* Set rows m-k+i+ib:m-1 of current block to zero */
            for (j = n - k + i; j < n - k + i + ib; j++) {
                for (l = m - k + i + ib; l < m; l++) {
                    A[l + j * lda] = ZERO;
                }
            }
        }
    }

    work[0] = (double complex)iws;
}
