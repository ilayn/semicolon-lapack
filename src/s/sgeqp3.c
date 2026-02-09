/**
 * @file sgeqp3.c
 * @brief SGEQP3 computes a QR factorization with column pivoting of a
 *        matrix A: A*P = Q*R using Level 3 BLAS.
 */

#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_single.h"

/**
 * SGEQP3 computes a QR factorization with column pivoting of a
 * matrix A:  A*P = Q*R  using Level 3 BLAS.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *
 *    Q = H(1) H(2) . . . H(k), where k = min(m,n).
 *
 * Each H(i) has the form
 *
 *    H(i) = I - tau * v * v**T
 *
 * where tau is a real scalar, and v is a real vector
 * with v(0:i-2) = 0 and v(i-1) = 1; v(i:m-1) is stored on exit in
 * A(i:m-1, i-1), and tau in TAU(i-1).
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the M-by-N matrix A.
 *                      On exit, the upper triangle of the array contains the
 *                      min(M,N)-by-N upper trapezoidal matrix R; the elements
 *                      below the diagonal, together with the array TAU,
 *                      represent the orthogonal matrix Q as a product of
 *                      min(M,N) elementary reflectors.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[in,out] jpvt  Integer array, dimension (n).
 *                      On entry, if jpvt[j] != 0, the j-th column of A is
 *                      permuted to the front of A*P (a leading column);
 *                      if jpvt[j] = 0, the j-th column of A is a free column.
 *                      On exit, if jpvt[j] = k, then the j-th column of A*P
 *                      was the k-th column of A (0-based).
 * @param[out]    tau   Double precision array, dimension (min(m,n)).
 *                      The scalar factors of the elementary reflectors.
 * @param[out]    work  Double precision array, dimension (max(1, lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The dimension of the array work. lwork >= 3*n+1.
 *                      For optimal performance lwork >= 2*n + (n+1)*nb, where nb
 *                      is the optimal blocksize.
 *                      If lwork = -1, then a workspace query is assumed; the
 *                      routine only calculates the optimal size of the work
 *                      array, returns this value as the first entry of the work
 *                      array, and no error message related to lwork is issued
 *                      by xerbla.
 * @param[out]    info  = 0: successful exit.
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgeqp3(const int m, const int n,
            float * const restrict A, const int lda,
            int * const restrict jpvt,
            float * const restrict tau,
            float * const restrict work, const int lwork,
            int *info)
{
    int iws, lwkopt, minmn, minws, na, nb, nbmin, nfxd, nx;
    int sm, sn, sminmn, topbmn;
    int j, jb, fjb;
    int lquery;
    int iinfo;

    /* Parameter validation */
    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }

    if (*info == 0) {
        minmn = m < n ? m : n;
        if (minmn == 0) {
            iws = 1;
            lwkopt = 1;
        } else {
            iws = 3 * n + 1;
            /* ILAENV(1, 'SGEQRF', ' ', M, N, -1, -1) -> NB for GEQRF */
            nb = lapack_get_nb("GEQRF");
            lwkopt = 2 * n + (n + 1) * nb;
        }
        work[0] = (float)lwkopt;

        if ((lwork < iws) && !lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        xerbla("SGEQP3", -(*info));
        return;
    }

    if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmn == 0) {
        return;
    }

    /*
     * Move initial columns up front.
     * In 0-based indexing: nfxd counts how many fixed columns we have.
     * Fixed columns (jpvt[j] != 0) are swapped to positions 0..nfxd-1.
     */
    nfxd = 0;
    for (j = 0; j < n; j++) {
        if (jpvt[j] != 0) {
            if (j != nfxd) {
                cblas_sswap(m, &A[j * lda], 1, &A[nfxd * lda], 1);
                jpvt[j] = jpvt[nfxd];
                jpvt[nfxd] = j;  /* 0-based original column index */
            } else {
                jpvt[j] = j;  /* 0-based original column index */
            }
            nfxd++;
        } else {
            jpvt[j] = j;  /* 0-based original column index */
        }
    }

    /*
     * Factorize fixed columns
     * =======================
     *
     * Compute the QR factorization of fixed columns and update
     * remaining columns.
     */
    if (nfxd > 0) {
        na = m < nfxd ? m : nfxd;

        /* QR factorization of the fixed columns */
        sgeqrf(m, na, A, lda, tau, work, lwork, &iinfo);
        iws = iws > (int)work[0] ? iws : (int)work[0];

        if (na < n) {
            /* Apply Q^T to remaining columns: A(:, na:n-1) */
            sormqr("L", "T", m, n - na, na, A, lda, tau,
                   &A[na * lda], lda, work, lwork, &iinfo);
            iws = iws > (int)work[0] ? iws : (int)work[0];
        }
    }

    /*
     * Factorize free columns
     * ======================
     */
    if (nfxd < minmn) {

        sm = m - nfxd;
        sn = n - nfxd;
        sminmn = minmn - nfxd;

        /* Determine the block size. */
        /* ILAENV(1, 'SGEQRF', ' ', SM, SN, -1, -1) -> NB for GEQRF */
        nb = lapack_get_nb("GEQRF");
        nbmin = 2;
        nx = 0;

        if ((nb > 1) && (nb < sminmn)) {

            /* Determine when to cross over from blocked to unblocked code. */
            /* ILAENV(3, 'SGEQRF', ' ', SM, SN, -1, -1) -> NX for GEQRF */
            nx = lapack_get_nx("GEQRF");
            if (nx > 0) {
                nx = 0 > nx ? 0 : nx;
            }

            if (nx < sminmn) {

                /* Determine if workspace is large enough for blocked code. */
                minws = 2 * sn + (sn + 1) * nb;
                iws = iws > minws ? iws : minws;
                if (lwork < minws) {

                    /*
                     * Not enough workspace to use optimal NB: Reduce NB and
                     * determine the minimum value of NB.
                     */
                    nb = (lwork - 2 * sn) / (sn + 1);
                    /* ILAENV(2, 'SGEQRF', ' ', SM, SN, -1, -1) -> NBMIN for GEQRF */
                    nbmin = lapack_get_nbmin("GEQRF");
                    nbmin = 2 > nbmin ? 2 : nbmin;
                }
            }
        }

        /*
         * Initialize partial column norms. The first N elements of work
         * store the exact column norms.
         *
         * Workspace layout (0-based):
         *   work[0..n-1]    = vn1 (partial column norms)
         *   work[n..2n-1]   = vn2 (exact column norms)
         *   work[2n..]      = used by slaqps (auxv + F matrix)
         *
         * In the Fortran source, vn1 and vn2 are indexed from NFXD+1..N
         * (1-based). Here we use 0-based: work[nfxd..n-1] = vn1 for free
         * columns, work[n+nfxd..2n-1] = vn2 for free columns.
         */
        for (j = nfxd; j < n; j++) {
            work[j] = cblas_snrm2(sm, &A[nfxd + j * lda], 1);
            work[n + j] = work[j];
        }

        if ((nb >= nbmin) && (nb < sminmn) && (nx < sminmn)) {

            /*
             * Use blocked code initially.
             *
             * j is 0-based column index into the full matrix.
             * The Fortran GOTO 30 loop becomes a while loop.
             */
            j = nfxd;

            /* topbmn = minmn - nx (0-based: last column for blocked code) */
            topbmn = minmn - nx;

            while (j < topbmn) {
                jb = nb < (topbmn - j) ? nb : (topbmn - j);

                /*
                 * Factorize JB columns among columns j:n-1.
                 *
                 * slaqps arguments (translated from Fortran 1-based):
                 *   M      = m
                 *   N      = n - j       (number of remaining columns)
                 *   OFFSET = j           (rows already factorized)
                 *   NB     = jb          (block size to try)
                 *   KB     = fjb         (actual columns factorized, output)
                 *   A      = &A[j*lda]   (start at column j)
                 *   LDA    = lda
                 *   JPVT   = &jpvt[j]
                 *   TAU    = &tau[j]
                 *   VN1    = &work[j]
                 *   VN2    = &work[n+j]
                 *   AUXV   = &work[2*n]
                 *   F      = &work[2*n+jb]
                 *   LDF    = n - j
                 */
                slaqps(m, n - j, j, jb, &fjb,
                       &A[j * lda], lda,
                       &jpvt[j], &tau[j],
                       &work[j], &work[n + j],
                       &work[2 * n], &work[2 * n + jb],
                       n - j);

                j = j + fjb;
            }
        } else {
            j = nfxd;
        }

        /*
         * Use unblocked code to factor the last or only block.
         *
         * slaqp2 arguments (translated from Fortran 1-based):
         *   M      = m
         *   N      = n - j       (remaining columns)
         *   OFFSET = j           (rows already factorized)
         *   A      = &A[j*lda]  (start at column j)
         *   LDA    = lda
         *   JPVT   = &jpvt[j]
         *   TAU    = &tau[j]
         *   VN1    = &work[j]
         *   VN2    = &work[n+j]
         *   WORK   = &work[2*n]
         */
        if (j < minmn) {
            slaqp2(m, n - j, j,
                   &A[j * lda], lda,
                   &jpvt[j], &tau[j],
                   &work[j], &work[n + j],
                   &work[2 * n]);
        }
    }

    work[0] = (float)iws;
}
