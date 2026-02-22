/**
 * @file sorgqr.c
 * @brief SORGQR generates all or part of the orthogonal matrix Q from
 *        a QR factorization determined by SGEQRF (blocked algorithm).
 */

#include "semicolon_cblas.h"
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_single.h"

/**
 * SORGQR generates an M-by-N real matrix Q with orthonormal columns,
 * which is defined as the first N columns of a product of K elementary
 * reflectors of order M
 *
 *    Q = H(0) H(1) . . . H(k-1)
 *
 * as returned by SGEQRF.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m      The number of rows of Q. m >= 0.
 * @param[in]     n      The number of columns of Q. m >= n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines Q. n >= k >= 0.
 * @param[in,out] A      On entry, the i-th column must contain the vector
 *                       which defines the elementary reflector H(i), for
 *                       i = 0,1,...,k-1, as returned by SGEQRF.
 *                       On exit, the m-by-n matrix Q.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by SGEQRF.
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work. lwork >= max(1, n).
 *                       For optimal performance, lwork >= n*nb.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sorgqr(const INT m, const INT n, const INT k,
            f32* restrict A, const INT lda,
            const f32* restrict tau,
            f32* restrict work, const INT lwork,
            INT* info)
{
    INT nb, nbmin, nx, iws, ldwork, lwkopt;
    INT i, ib, iinfo, j, l;
    INT ki, kk;
    INT lquery;
    const f32 ZERO = 0.0f;

    /* Parameter validation */
    *info = 0;
    nb = lapack_get_nb("ORGQR");
    lwkopt = (n > 1 ? n : 1) * nb;
    work[0] = (f32)lwkopt;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > m) {
        *info = -2;
    } else if (k < 0 || k > n) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (lwork < (n > 1 ? n : 1) && !lquery) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("SORGQR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n <= 0) {
        work[0] = 1.0f;
        return;
    }

    nbmin = 2;
    nx = 0;
    iws = n;
    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("ORGQR");
        if (nx < k) {
            ldwork = n;
            iws = ldwork * nb;
            if (lwork < iws) {
                nb = lwork / ldwork;
                nbmin = lapack_get_nbmin("ORGQR");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code after the last block.
         * The first kk columns are handled by the block method. */
        ki = ((k - nx - 1) / nb) * nb;
        kk = k < ki + nb ? k : ki + nb;

        /* Set A(0:kk-1, kk:n-1) to zero */
        for (j = kk; j < n; j++) {
            for (i = 0; i < kk; i++) {
                A[i + j * lda] = ZERO;
            }
        }
    } else {
        kk = 0;
    }

    /* Use unblocked code for the last or only block */
    if (kk < n) {
        sorg2r(m - kk, n - kk, k - kk,
               &A[kk + kk * lda], lda,
               &tau[kk], work, &iinfo);
    }

    if (kk > 0) {
        /* Use blocked code */
        ldwork = n;
        for (i = ki; i >= 0; i -= nb) {
            ib = nb < k - i ? nb : k - i;
            if (i + ib < n) {
                /* Form the triangular factor of the block reflector
                 * H = H(i) H(i+1) . . . H(i+ib-1) */
                slarft("F", "C", m - i, ib,
                       &A[i + i * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H to A(i:m-1, i+ib:n-1) from the left */
                slarfb("L", "N", "F", "C",
                       m - i, n - i - ib, ib,
                       &A[i + i * lda], lda,
                       work, ldwork,
                       &A[i + (i + ib) * lda], lda,
                       &work[ib], ldwork);
            }

            /* Apply H to rows i:m-1 of current block */
            sorg2r(m - i, ib, ib, &A[i + i * lda], lda,
                   &tau[i], work, &iinfo);

            /* Set rows 0:i-1 of current block to zero */
            for (j = i; j < i + ib; j++) {
                for (l = 0; l < i; l++) {
                    A[l + j * lda] = ZERO;
                }
            }
        }
    }

    work[0] = (f32)iws;
}
