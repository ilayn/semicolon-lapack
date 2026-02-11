/**
 * @file sorgbr.c
 * @brief SORGBR generates one of the orthogonal matrices Q or P**T determined by SGEBRD.
 */

#include "semicolon_lapack_single.h"

/**
 * SORGBR generates one of the real orthogonal matrices Q or P**T
 * determined by SGEBRD when reducing a real matrix A to bidiagonal
 * form: A = Q * B * P**T.  Q and P**T are defined as products of
 * elementary reflectors H(i) or G(i) respectively.
 *
 * If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
 * is of order M:
 * if m >= k, Q = H(1) H(2) . . . H(k) and SORGBR returns the first n
 * columns of Q, where m >= n >= k;
 * if m < k, Q = H(1) H(2) . . . H(m-1) and SORGBR returns Q as an
 * M-by-M matrix.
 *
 * If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**T
 * is of order N:
 * if k < n, P**T = G(k) . . . G(2) G(1) and SORGBR returns the first m
 * rows of P**T, where n >= m >= k;
 * if k >= n, P**T = G(n-1) . . . G(2) G(1) and SORGBR returns P**T as
 * an N-by-N matrix.
 *
 * @param[in]     vect  Specifies whether the matrix Q or the matrix P**T is
 *                      required, as defined in the transformation applied by SGEBRD:
 *                      = 'Q': generate Q;
 *                      = 'P': generate P**T.
 * @param[in]     m     The number of rows of the matrix Q or P**T to be returned.
 *                      m >= 0.
 * @param[in]     n     The number of columns of the matrix Q or P**T to be returned.
 *                      n >= 0.
 *                      If vect = 'Q', m >= n >= min(m,k);
 *                      if vect = 'P', n >= m >= min(n,k).
 * @param[in]     k     If vect = 'Q', the number of columns in the original M-by-K
 *                      matrix reduced by SGEBRD.
 *                      If vect = 'P', the number of rows in the original K-by-N
 *                      matrix reduced by SGEBRD.
 *                      k >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the vectors which define the elementary reflectors,
 *                      as returned by SGEBRD.
 *                      On exit, the M-by-N matrix Q or P**T.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,m).
 * @param[in]     tau   Double precision array, dimension
 *                      (min(m,k)) if vect = 'Q'
 *                      (min(n,k)) if vect = 'P'
 *                      tau[i] must contain the scalar factor of the elementary
 *                      reflector H(i) or G(i), which determines Q or P**T, as
 *                      returned by SGEBRD in its array argument TAUQ or TAUP.
 * @param[out]    work  Double precision array, dimension (max(1,lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The dimension of the array work. lwork >= max(1,min(m,n)).
 *                      For optimum performance lwork >= min(m,n)*NB, where NB
 *                      is the optimal blocksize.
 *                      If lwork = -1, then a workspace query is assumed; the routine
 *                      only calculates the optimal size of the work array, returns
 *                      this value as the first entry of the work array, and no error
 *                      message related to lwork is issued by xerbla.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void sorgbr(const char* vect, const int m, const int n, const int k,
            float* const restrict A, const int lda,
            const float* const restrict tau,
            float* const restrict work, const int lwork, int* info)
{
    int i, j, iinfo, lquery, mn, wantq, lwkopt;

    /* Test the input arguments */
    *info = 0;
    wantq = (vect[0] == 'Q' || vect[0] == 'q');
    mn = (m < n) ? m : n;
    lquery = (lwork == -1);

    if (!wantq && !(vect[0] == 'P' || vect[0] == 'p')) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0 ||
               (wantq && (n > m || n < ((m < k) ? m : k))) ||
               (!wantq && (m > n || m < ((n < k) ? n : k)))) {
        *info = -3;
    } else if (k < 0) {
        *info = -4;
    } else if (lda < ((1 > m) ? 1 : m)) {
        *info = -6;
    } else if (lwork < ((1 > mn) ? 1 : mn) && !lquery) {
        *info = -9;
    }

    if (*info == 0) {
        work[0] = 1.0f;
        if (wantq) {
            if (m >= k) {
                sorgqr(m, n, k, A, lda, tau, work, -1, &iinfo);
            } else {
                if (m > 1) {
                    sorgqr(m - 1, m - 1, m - 1, A, lda, tau, work, -1, &iinfo);
                }
            }
        } else {
            if (k < n) {
                sorglq(m, n, k, A, lda, tau, work, -1, &iinfo);
            } else {
                if (n > 1) {
                    sorglq(n - 1, n - 1, n - 1, A, lda, tau, work, -1, &iinfo);
                }
            }
        }
        lwkopt = (int)work[0];
        if (lwkopt < mn) lwkopt = mn;
    }

    if (*info != 0) {
        xerbla("SORGBR", -(*info));
        return;
    } else if (lquery) {
        work[0] = (float)lwkopt;
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        work[0] = 1.0f;
        return;
    }

    if (wantq) {
        /* Form Q, determined by a call to SGEBRD to reduce an m-by-k matrix */
        if (m >= k) {
            /* If m >= k, assume m >= n >= k */
            sorgqr(m, n, k, A, lda, tau, work, lwork, &iinfo);
        } else {
            /* If m < k, assume m = n
             * Shift the vectors which define the elementary reflectors one
             * column to the right, and set the first row and column of Q
             * to those of the unit matrix */
            for (j = m - 1; j >= 1; j--) {
                A[j * lda] = 0.0f;
                for (i = j; i < m; i++) {
                    A[i + j * lda] = A[i + (j - 1) * lda];
                }
            }
            A[0] = 1.0f;
            for (i = 1; i < m; i++) {
                A[i] = 0.0f;
            }
            if (m > 1) {
                /* Form Q(2:m, 2:m) */
                sorgqr(m - 1, m - 1, m - 1, &A[1 + lda], lda, tau, work, lwork, &iinfo);
            }
        }
    } else {
        /* Form P**T, determined by a call to SGEBRD to reduce a k-by-n matrix */
        if (k < n) {
            /* If k < n, assume k <= m <= n */
            sorglq(m, n, k, A, lda, tau, work, lwork, &iinfo);
        } else {
            /* If k >= n, assume m = n
             * Shift the vectors which define the elementary reflectors one
             * row downward, and set the first row and column of P**T to
             * those of the unit matrix */
            A[0] = 1.0f;
            for (i = 1; i < n; i++) {
                A[i] = 0.0f;
            }
            for (j = 1; j < n; j++) {
                for (i = j - 1; i >= 1; i--) {
                    A[i + j * lda] = A[i - 1 + j * lda];
                }
                A[j * lda] = 0.0f;
            }
            if (n > 1) {
                /* Form P**T(2:n, 2:n) */
                sorglq(n - 1, n - 1, n - 1, &A[1 + lda], lda, tau, work, lwork, &iinfo);
            }
        }
    }
    work[0] = (float)lwkopt;
}
