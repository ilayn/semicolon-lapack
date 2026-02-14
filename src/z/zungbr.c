/**
 * @file zungbr.c
 * @brief ZUNGBR generates one of the unitary matrices Q or P**H
 *        determined by ZGEBRD.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZUNGBR generates one of the complex unitary matrices Q or P**H
 * determined by ZGEBRD when reducing a complex matrix A to bidiagonal
 * form: A = Q * B * P**H.  Q and P**H are defined as products of
 * elementary reflectors H(i) or G(i) respectively.
 *
 * If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
 * is of order M:
 * if m >= k, Q = H(1) H(2) . . . H(k) and ZUNGBR returns the first n
 * columns of Q, where m >= n >= k;
 * if m < k, Q = H(1) H(2) . . . H(m-1) and ZUNGBR returns Q as an
 * M-by-M matrix.
 *
 * If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**H
 * is of order N:
 * if k < n, P**H = G(k) . . . G(2) G(1) and ZUNGBR returns the first m
 * rows of P**H, where n >= m >= k;
 * if k >= n, P**H = G(n-1) . . . G(2) G(1) and ZUNGBR returns P**H as
 * an N-by-N matrix.
 *
 * @param[in]     vect  Specifies whether the matrix Q or the matrix P**H is
 *                      required:
 *                      = 'Q':  generate Q;
 *                      = 'P':  generate P**H.
 * @param[in]     m     The number of rows of the matrix Q or P**H to be
 *                      returned. m >= 0.
 * @param[in]     n     The number of columns of the matrix Q or P**H to be
 *                      returned. n >= 0.
 *                      If VECT = 'Q', m >= n >= min(m,k);
 *                      if VECT = 'P', n >= m >= min(n,k).
 * @param[in]     k     If VECT = 'Q', the number of columns in the original
 *                      m-by-k matrix reduced by ZGEBRD.
 *                      If VECT = 'P', the number of rows in the original
 *                      k-by-n matrix reduced by ZGEBRD.
 *                      k >= 0.
 * @param[in,out] A     Complex array, dimension (lda, n).
 *                      On entry, the vectors which define the elementary
 *                      reflectors, as returned by ZGEBRD.
 *                      On exit, the m-by-n matrix Q or P**H.
 * @param[in]     lda   The leading dimension of the array A. lda >= m.
 * @param[in]     tau   Complex array, dimension
 *                      (min(m,k)) if VECT = 'Q'
 *                      (min(n,k)) if VECT = 'P'
 *                      TAU(i) must contain the scalar factor of the elementary
 *                      reflector H(i) or G(i), which determines Q or P**H, as
 *                      returned by ZGEBRD in its array argument TAUQ or TAUP.
 * @param[out]    work  Complex array, dimension (max(1,lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The dimension of the array work. lwork >= max(1,min(m,n)).
 *                      For optimum performance lwork >= min(m,n)*NB, where NB
 *                      is the optimal blocksize.
 *
 *                      If lwork = -1, then a workspace query is assumed; the
 *                      routine only calculates the optimal size of the work
 *                      array, returns this value as the first entry of the work
 *                      array, and no error message related to lwork is issued
 *                      by xerbla.
 * @param[out]    info  = 0:  successful exit
 *                      < 0:  if info = -i, the i-th argument had an illegal value
 */
void zungbr(const char* vect, const int m, const int n, const int k,
            double complex* const restrict A, const int lda,
            const double complex* const restrict tau,
            double complex* const restrict work, const int lwork,
            int* info)
{
    const double complex ZERO = CMPLX(0.0, 0.0);
    const double complex ONE = CMPLX(1.0, 0.0);

    int i, iinfo, j, lwkopt, mn;
    int lquery, wantq;

    *info = 0;
    wantq = (vect[0] == 'Q' || vect[0] == 'q');
    mn = m < n ? m : n;
    lquery = (lwork == -1);

    if (!wantq && !(vect[0] == 'P' || vect[0] == 'p')) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0 ||
               (wantq && (n > m || n < (m < k ? m : k))) ||
               (!wantq && (m > n || m < (n < k ? n : k)))) {
        *info = -3;
    } else if (k < 0) {
        *info = -4;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -6;
    } else if (lwork < (1 > mn ? 1 : mn) && !lquery) {
        *info = -9;
    }

    if (*info == 0) {
        work[0] = CMPLX(1.0, 0.0);
        if (wantq) {
            if (m >= k) {
                zungqr(m, n, k, A, lda, tau, work, -1, &iinfo);
            } else {
                if (m > 1) {
                    zungqr(m - 1, m - 1, m - 1, A, lda, tau, work, -1,
                           &iinfo);
                }
            }
        } else {
            if (k < n) {
                zunglq(m, n, k, A, lda, tau, work, -1, &iinfo);
            } else {
                if (n > 1) {
                    zunglq(n - 1, n - 1, n - 1, A, lda, tau, work, -1,
                           &iinfo);
                }
            }
        }
        lwkopt = (int)creal(work[0]);
        lwkopt = lwkopt > mn ? lwkopt : mn;
    }

    if (*info != 0) {
        xerbla("ZUNGBR", -(*info));
        return;
    } else if (lquery) {
        work[0] = CMPLX((double)lwkopt, 0.0);
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        work[0] = CMPLX(1.0, 0.0);
        return;
    }

    if (wantq) {

        /* Form Q, determined by a call to ZGEBRD to reduce an m-by-k
           matrix */

        if (m >= k) {

            /* If m >= k, assume m >= n >= k */

            zungqr(m, n, k, A, lda, tau, work, lwork, &iinfo);

        } else {

            /* If m < k, assume m = n

               Shift the vectors which define the elementary reflectors one
               column to the right, and set the first row and column of Q
               to those of the unit matrix */

            for (j = m - 1; j >= 1; j--) {
                A[0 + j * lda] = ZERO;
                for (i = j + 1; i < m; i++) {
                    A[i + j * lda] = A[i + (j - 1) * lda];
                }
            }
            A[0 + 0 * lda] = ONE;
            for (i = 1; i < m; i++) {
                A[i + 0 * lda] = ZERO;
            }
            if (m > 1) {

                /* Form Q(2:m,2:m) */

                zungqr(m - 1, m - 1, m - 1, &A[1 + 1 * lda], lda, tau,
                       work, lwork, &iinfo);
            }
        }
    } else {

        /* Form P**H, determined by a call to ZGEBRD to reduce a k-by-n
           matrix */

        if (k < n) {

            /* If k < n, assume k <= m <= n */

            zunglq(m, n, k, A, lda, tau, work, lwork, &iinfo);

        } else {

            /* If k >= n, assume m = n

               Shift the vectors which define the elementary reflectors one
               row downward, and set the first row and column of P**H to
               those of the unit matrix */

            A[0 + 0 * lda] = ONE;
            for (i = 1; i < n; i++) {
                A[i + 0 * lda] = ZERO;
            }
            for (j = 1; j < n; j++) {
                for (i = j - 1; i >= 1; i--) {
                    A[i + j * lda] = A[(i - 1) + j * lda];
                }
                A[0 + j * lda] = ZERO;
            }
            if (n > 1) {

                /* Form P**H(2:n,2:n) */

                zunglq(n - 1, n - 1, n - 1, &A[1 + 1 * lda], lda, tau,
                       work, lwork, &iinfo);
            }
        }
    }
    work[0] = CMPLX((double)lwkopt, 0.0);
}
