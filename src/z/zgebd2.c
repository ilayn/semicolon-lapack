/**
 * @file zgebd2.c
 * @brief ZGEBD2 reduces a general matrix to bidiagonal form using an unblocked algorithm.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZGEBD2 reduces a complex general m by n matrix A to upper or lower
 * real bidiagonal form B by a unitary transformation: Q**H * A * P = B.
 *
 * If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
 *
 * @param[in]     m     The number of rows in the matrix A. m >= 0.
 * @param[in]     n     The number of columns in the matrix A. n >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the m by n general matrix to be reduced.
 *                      On exit,
 *                      if m >= n, the diagonal and the first superdiagonal are
 *                        overwritten with the upper bidiagonal matrix B; the
 *                        elements below the diagonal, with the array TAUQ, represent
 *                        the unitary matrix Q as a product of elementary
 *                        reflectors, and the elements above the first superdiagonal,
 *                        with the array TAUP, represent the unitary matrix P as
 *                        a product of elementary reflectors;
 *                      if m < n, the diagonal and the first subdiagonal are
 *                        overwritten with the lower bidiagonal matrix B; the
 *                        elements below the first subdiagonal, with the array TAUQ,
 *                        represent the unitary matrix Q as a product of
 *                        elementary reflectors, and the elements above the diagonal,
 *                        with the array TAUP, represent the unitary matrix P as
 *                        a product of elementary reflectors.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    D     Double precision array, dimension (min(m,n)).
 *                      The diagonal elements of the bidiagonal matrix B:
 *                      D[i] = A[i,i].
 * @param[out]    E     Double precision array, dimension (min(m,n)-1).
 *                      The off-diagonal elements of the bidiagonal matrix B:
 *                      if m >= n, E[i] = A[i,i+1] for i = 0,1,...,n-2;
 *                      if m < n, E[i] = A[i+1,i] for i = 0,1,...,m-2.
 * @param[out]    tauq  Complex*16 array, dimension (min(m,n)).
 *                      The scalar factors of the elementary reflectors which
 *                      represent the unitary matrix Q.
 * @param[out]    taup  Complex*16 array, dimension (min(m,n)).
 *                      The scalar factors of the elementary reflectors which
 *                      represent the unitary matrix P.
 * @param[out]    work  Complex*16 array, dimension (max(m,n)).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zgebd2(const INT m, const INT n, c128* restrict A, const INT lda,
            f64* restrict D, f64* restrict E,
            c128* restrict tauq, c128* restrict taup,
            c128* restrict work, INT* info)
{
    const c128 ZERO = CMPLX(0.0, 0.0);
    INT i;
    c128 alpha;

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
        xerbla("ZGEBD2", -(*info));
        return;
    }

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i = 0; i < n; i++) {

            /* Generate elementary reflector H(i) to annihilate A[i+1:m-1, i] */
            alpha = A[i + i * lda];
            zlarfg(m - i, &alpha, &A[((i + 1) < m ? (i + 1) : (m - 1)) + i * lda], 1,
                   &tauq[i]);
            D[i] = creal(alpha);

            /* Apply H(i)**H to A[i:m-1, i+1:n-1] from the left */
            if (i < n - 1)
                zlarf1f("Left", m - i, n - i - 1, &A[i + i * lda], 1,
                        conj(tauq[i]), &A[i + (i + 1) * lda], lda, work);
            A[i + i * lda] = CMPLX(D[i], 0.0);

            if (i < n - 1) {

                /* Generate elementary reflector G(i) to annihilate A[i, i+2:n-1] */
                zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                alpha = A[i + (i + 1) * lda];
                zlarfg(n - i - 1, &alpha, &A[i + ((i + 2) < n ? (i + 2) : (n - 1)) * lda], lda,
                       &taup[i]);
                E[i] = creal(alpha);

                /* Apply G(i) to A[i+1:m-1, i+1:n-1] from the right */
                zlarf1f("Right", m - i - 1, n - i - 1, &A[i + (i + 1) * lda], lda,
                        taup[i], &A[i + 1 + (i + 1) * lda], lda, work);
                zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                A[i + (i + 1) * lda] = CMPLX(E[i], 0.0);
            } else {
                taup[i] = ZERO;
            }
        }
    } else {
        /* Reduce to lower bidiagonal form */
        for (i = 0; i < m; i++) {

            /* Generate elementary reflector G(i) to annihilate A[i, i+1:n-1] */
            zlacgv(n - i, &A[i + i * lda], lda);
            alpha = A[i + i * lda];
            zlarfg(n - i, &alpha, &A[i + ((i + 1) < n ? (i + 1) : (n - 1)) * lda], lda,
                   &taup[i]);
            D[i] = creal(alpha);

            /* Apply G(i) to A[i+1:m-1, i:n-1] from the right */
            if (i < m - 1)
                zlarf1f("Right", m - i - 1, n - i, &A[i + i * lda], lda, taup[i],
                        &A[i + 1 + i * lda], lda, work);
            zlacgv(n - i, &A[i + i * lda], lda);
            A[i + i * lda] = CMPLX(D[i], 0.0);

            if (i < m - 1) {

                /* Generate elementary reflector H(i) to annihilate A[i+2:m-1, i] */
                alpha = A[i + 1 + i * lda];
                zlarfg(m - i - 1, &alpha, &A[((i + 2) < m ? (i + 2) : (m - 1)) + i * lda], 1,
                       &tauq[i]);
                E[i] = creal(alpha);

                /* Apply H(i)**H to A[i+1:m-1, i+1:n-1] from the left */
                zlarf1f("Left", m - i - 1, n - i - 1, &A[i + 1 + i * lda], 1,
                        conj(tauq[i]), &A[i + 1 + (i + 1) * lda], lda,
                        work);
                A[i + 1 + i * lda] = CMPLX(E[i], 0.0);
            } else {
                tauq[i] = ZERO;
            }
        }
    }
}
