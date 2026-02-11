/**
 * @file sgebd2.c
 * @brief SGEBD2 reduces a general matrix to bidiagonal form using an unblocked algorithm.
 */

#include "semicolon_lapack_single.h"

/**
 * SGEBD2 reduces a real general m by n matrix A to upper or lower
 * bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
 *
 * If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
 *
 * @param[in]     m     The number of rows in the matrix A. m >= 0.
 * @param[in]     n     The number of columns in the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the m by n general matrix to be reduced.
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
 * @param[out]    work  Double precision array, dimension (max(m,n)).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgebd2(const int m, const int n, float* const restrict A, const int lda,
            float* const restrict D, float* const restrict E,
            float* const restrict tauq, float* const restrict taup,
            float* const restrict work, int* info)
{
    int i;

    /* Test the input parameters */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < ((1 > m) ? 1 : m)) {
        *info = -4;
    }
    if (*info < 0) {
        xerbla("SGEBD2", -(*info));
        return;
    }

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i = 0; i < n; i++) {
            /* Generate elementary reflector H(i) to annihilate A[i+1:m-1, i] */
            slarfg(m - i, &A[i + i * lda], &A[((i + 1) < m ? (i + 1) : (m - 1)) + i * lda], 1,
                   &tauq[i]);
            D[i] = A[i + i * lda];

            /* Apply H(i) to A[i:m-1, i+1:n-1] from the left */
            if (i < n - 1) {
                slarf1f("L", m - i, n - i - 1, &A[i + i * lda], 1, tauq[i],
                        &A[i + (i + 1) * lda], lda, work);
            }

            if (i < n - 1) {
                /* Generate elementary reflector G(i) to annihilate A[i, i+2:n-1] */
                slarfg(n - i - 1, &A[i + (i + 1) * lda],
                       &A[i + ((i + 2) < n ? (i + 2) : (n - 1)) * lda], lda, &taup[i]);
                E[i] = A[i + (i + 1) * lda];

                /* Apply G(i) to A[i+1:m-1, i+1:n-1] from the right */
                slarf1f("R", m - i - 1, n - i - 1, &A[i + (i + 1) * lda], lda,
                        taup[i], &A[i + 1 + (i + 1) * lda], lda, work);
            } else {
                taup[i] = 0.0f;
            }
        }
    } else {
        /* Reduce to lower bidiagonal form */
        for (i = 0; i < m; i++) {
            /* Generate elementary reflector G(i) to annihilate A[i, i+1:n-1] */
            slarfg(n - i, &A[i + i * lda], &A[i + ((i + 1) < n ? (i + 1) : (n - 1)) * lda], lda,
                   &taup[i]);
            D[i] = A[i + i * lda];

            /* Apply G(i) to A[i+1:m-1, i:n-1] from the right */
            if (i < m - 1) {
                slarf1f("R", m - i - 1, n - i, &A[i + i * lda], lda, taup[i],
                        &A[i + 1 + i * lda], lda, work);
            }

            if (i < m - 1) {
                /* Generate elementary reflector H(i) to annihilate A[i+2:m-1, i] */
                slarfg(m - i - 1, &A[i + 1 + i * lda],
                       &A[((i + 2) < m ? (i + 2) : (m - 1)) + i * lda], 1, &tauq[i]);
                E[i] = A[i + 1 + i * lda];

                /* Apply H(i) to A[i+1:m-1, i+1:n-1] from the left */
                slarf1f("L", m - i - 1, n - i - 1, &A[i + 1 + i * lda], 1,
                        tauq[i], &A[i + 1 + (i + 1) * lda], lda, work);
            } else {
                tauq[i] = 0.0f;
            }
        }
    }
}
