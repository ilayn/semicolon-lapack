/**
 * @file dtzrzf.c
 * @brief DTZRZF reduces an upper trapezoidal matrix to upper triangular form.
 */

#include "semicolon_cblas.h"
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_double.h"

/**
 * DTZRZF reduces the M-by-N ( M<=N ) real upper trapezoidal matrix A
 * to upper triangular form by means of orthogonal transformations.
 *
 * The upper trapezoidal matrix A is factored as
 *
 *    A = ( R  0 ) * Z,
 *
 * where Z is an N-by-N orthogonal matrix and R is an M-by-M upper
 * triangular matrix.
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= m.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the leading M-by-N upper trapezoidal part of the
 *                      array A must contain the matrix to be factorized.
 *                      On exit, the leading M-by-M upper triangular part of A
 *                      contains the upper triangular matrix R, and elements M+1 to
 *                      N of the first M rows of A, with the array TAU, represent the
 *                      orthogonal matrix Z as a product of M elementary reflectors.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    tau   Double precision array, dimension (m).
 *                      The scalar factors of the elementary reflectors.
 * @param[out]    work  Double precision array, dimension (max(1, lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The dimension of the array work. lwork >= max(1, m).
 *                      For optimum performance lwork >= m*nb, where nb is
 *                      the optimal blocksize.
 *                      If lwork = -1, then a workspace query is assumed; the routine
 *                      only calculates the optimal size of the work array, returns
 *                      this value as the first entry of the work array, and no error
 *                      message related to lwork is issued by xerbla.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dtzrzf(const INT m, const INT n,
            f64* restrict A, const INT lda,
            f64* restrict tau,
            f64* restrict work, const INT lwork,
            INT* info)
{
    INT i, ib, iws, ki, kk, ldwork, lwkmin, lwkopt, mu, nb, nbmin, nx;
    INT lquery;

    /* Test the input arguments */
    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < m) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }

    if (*info == 0) {
        if (m == 0 || m == n) {
            lwkopt = 1;
            lwkmin = 1;
        } else {
            /* Determine the block size.
             * NB from ILAENV(1, 'DGERQF', ' ', M, N, -1, -1) */
            nb = lapack_get_nb("GERQF");
            lwkopt = m * nb;
            lwkmin = m > 1 ? m : 1;
        }
        work[0] = (f64)lwkopt;

        if (lwork < lwkmin && !lquery) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("DTZRZF", -(*info));
        return;
    }

    if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0) {
        return;
    }

    if (m == n) {
        for (i = 0; i < n; i++) {
            tau[i] = 0.0;
        }
        return;
    }

    nbmin = 2;
    nx = 1;

    if (nb > 1 && nb < m) {

        /* Determine when to cross over from blocked to unblocked code.
         * NX from ILAENV(3, 'DGERQF', ' ', M, N, -1, -1) */
        nx = lapack_get_nx("GERQF");
        if (nx < 0) nx = 0;

        if (nx < m) {

            /* Determine if workspace is large enough for blocked code. */
            ldwork = m;
            iws = ldwork * nb;
            if (lwork < iws) {

                /* Not enough workspace to use optimal NB:  reduce NB and
                 * determine the minimum value of NB.
                 * NBMIN from ILAENV(2, 'DGERQF', ' ', M, N, -1, -1) */
                nb = lwork / ldwork;
                nbmin = lapack_get_nbmin("GERQF");
                if (nbmin < 2) nbmin = 2;
            }
        }
    }

    if (nb >= nbmin && nb < m && nx < m) {

        /* Use blocked code initially.
         * The last kk rows are handled by the block method. */
        ki = ((m - nx - 1) / nb) * nb;
        kk = m < (ki + nb) ? m : (ki + nb);

        for (i = m - kk + ki; i >= m - kk; i -= nb) {
            ib = (m - i) < nb ? (m - i) : nb;

            /* Compute the TZ factorization of the current block
             * A(i:i+ib-1,i:n-1) */
            dlatrz(ib, n - i, n - m, &A[i + i * lda], lda, &tau[i], work);

            if (i > 0) {

                /* Form the triangular factor of the block reflector
                 * H = H(i+ib-1) . . . H(i+1) H(i) */
                dlarzt("B", "R", n - m, ib, &A[i + m * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H to A(0:i-1,i:n-1) from the right */
                dlarzb("R", "N", "B", "R",
                       i, n - i, ib, n - m,
                       &A[i + m * lda], lda,
                       work, ldwork,
                       &A[0 + i * lda], lda,
                       &work[ib], ldwork);
            }
        }

        mu = m - kk;
    } else {
        mu = m;
    }

    /* Use unblocked code to factor the last or only block */
    if (mu > 0) {
        dlatrz(mu, n, n - m, A, lda, tau, work);
    }

    work[0] = (f64)lwkopt;
}
