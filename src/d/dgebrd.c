/**
 * @file dgebrd.c
 * @brief DGEBRD reduces a general matrix to bidiagonal form using a blocked algorithm.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"
#include <cblas.h>

/**
 * DGEBRD reduces a general real M-by-N matrix A to upper or lower
 * bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
 *
 * If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
 *
 * @param[in]     m     The number of rows in the matrix A. m >= 0.
 * @param[in]     n     The number of columns in the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the M-by-N general matrix to be reduced.
 *                      On exit,
 *                      if m >= n, the diagonal and the first superdiagonal are
 *                        overwritten with the upper bidiagonal matrix B; the
 *                        elements below the diagonal, with the array TAUQ, represent
 *                        the orthogonal matrix Q as a product of elementary
 *                        reflectors, and the elements above the first superdiagonal,
 *                        with the array TAUP, represent the orthogonal matrix P as
 *                        a product of elementary reflectors;
 *                      if m < n, the diagonal and the first subdiagonal are
 *                        overwritten with the lower bidiagonal matrix B; the
 *                        elements below the first subdiagonal, with the array TAUQ,
 *                        represent the orthogonal matrix Q as a product of
 *                        elementary reflectors, and the elements above the diagonal,
 *                        with the array TAUP, represent the orthogonal matrix P as
 *                        a product of elementary reflectors.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    D     Double precision array, dimension (min(m,n)).
 *                      The diagonal elements of the bidiagonal matrix B:
 *                      D[i] = A[i,i].
 * @param[out]    E     Double precision array, dimension (min(m,n)-1).
 *                      The off-diagonal elements of the bidiagonal matrix B:
 *                      if m >= n, E[i] = A[i,i+1] for i = 0,1,...,n-2;
 *                      if m < n, E[i] = A[i+1,i] for i = 0,1,...,m-2.
 * @param[out]    tauq  Double precision array, dimension (min(m,n)).
 *                      The scalar factors of the elementary reflectors which
 *                      represent the orthogonal matrix Q.
 * @param[out]    taup  Double precision array, dimension (min(m,n)).
 *                      The scalar factors of the elementary reflectors which
 *                      represent the orthogonal matrix P.
 * @param[out]    work  Double precision array, dimension (max(1,lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The length of the array work.
 *                      lwork >= 1, if min(m,n) = 0, and lwork >= max(m,n), otherwise.
 *                      For optimum performance lwork >= (m+n)*NB, where NB
 *                      is the optimal blocksize.
 *                      If lwork = -1, then a workspace query is assumed; the routine
 *                      only calculates the optimal size of the work array, returns
 *                      this value as the first entry of the work array, and no error
 *                      message related to lwork is issued by xerbla.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dgebrd(const int m, const int n, double* const restrict A, const int lda,
            double* const restrict D, double* const restrict E,
            double* const restrict tauq, double* const restrict taup,
            double* const restrict work, const int lwork, int* info)
{
    int i, j, iinfo;
    int lquery, minmn, nb, nbmin, nx;
    int ldwrkx, ldwrky, lwkmin, lwkopt, ws;

    /* Test the input parameters */
    *info = 0;
    minmn = (m < n) ? m : n;
    if (minmn == 0) {
        lwkmin = 1;
        lwkopt = 1;
    } else {
        lwkmin = (m > n) ? m : n;
        nb = lapack_get_nb("GEBRD");
        if (nb < 1) nb = 1;
        lwkopt = (m + n) * nb;
    }
    work[0] = (double)lwkopt;

    lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < ((1 > m) ? 1 : m)) {
        *info = -4;
    } else if (lwork < lwkmin && !lquery) {
        *info = -10;
    }
    if (*info < 0) {
        xerbla("DGEBRD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmn == 0) {
        work[0] = 1.0;
        return;
    }

    ws = (m > n) ? m : n;
    ldwrkx = m;
    ldwrky = n;

    nb = lapack_get_nb("GEBRD");
    if (nb < 1) nb = 1;

    if (nb > 1 && nb < minmn) {
        /* Set the crossover point NX */
        nx = lapack_get_nx("GEBRD");
        if (nx < nb) nx = nb;

        /* Determine when to switch from blocked to unblocked code */
        if (nx < minmn) {
            ws = lwkopt;
            if (lwork < ws) {
                /* Not enough work space for the optimal NB, consider using
                 * a smaller block size */
                nbmin = lapack_get_nbmin("GEBRD");
                if (lwork >= (m + n) * nbmin) {
                    nb = lwork / (m + n);
                } else {
                    nb = 1;
                    nx = minmn;
                }
            }
        }
    } else {
        nx = minmn;
    }

    for (i = 0; i < minmn - nx; i += nb) {
        /*
         * Reduce rows and columns i:i+nb-1 to bidiagonal form and return
         * the matrices X and Y which are needed to update the unreduced
         * part of the matrix
         */
        dlabrd(m - i, n - i, nb, &A[i + i * lda], lda, &D[i], &E[i],
               &tauq[i], &taup[i], work, ldwrkx,
               &work[ldwrkx * nb], ldwrky);

        /*
         * Update the trailing submatrix A[i+nb:m-1, i+nb:n-1], using an update
         * of the form  A := A - V*Y**T - X*U**T
         */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    m - i - nb, n - i - nb, nb,
                    -1.0, &A[i + nb + i * lda], lda,
                    &work[ldwrkx * nb + nb], ldwrky,
                    1.0, &A[i + nb + (i + nb) * lda], lda);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - i - nb, n - i - nb, nb,
                    -1.0, &work[nb], ldwrkx,
                    &A[i + (i + nb) * lda], lda,
                    1.0, &A[i + nb + (i + nb) * lda], lda);

        /* Copy diagonal and off-diagonal elements of B back into A */
        if (m >= n) {
            for (j = i; j < i + nb; j++) {
                A[j + j * lda] = D[j];
                A[j + (j + 1) * lda] = E[j];
            }
        } else {
            for (j = i; j < i + nb; j++) {
                A[j + j * lda] = D[j];
                A[j + 1 + j * lda] = E[j];
            }
        }
    }

    /* Use unblocked code to reduce the remainder of the matrix */
    dgebd2(m - i, n - i, &A[i + i * lda], lda, &D[i], &E[i],
           &tauq[i], &taup[i], work, &iinfo);
    work[0] = (double)ws;
}
