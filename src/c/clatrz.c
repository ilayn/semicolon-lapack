#include "semicolon_lapack_complex_single.h"
#include <complex.h>
/**
 * @file clatrz.c
 * @brief CLATRZ factors an upper trapezoidal matrix by means of unitary
 *        transformations.
 */

/**
 * CLATRZ factors the M-by-(M+L) complex upper trapezoidal matrix
 * [ A1 A2 ] = [ A(0:M-1,0:M-1) A(0:M-1,N-L:N-1) ] as ( R  0 ) * Z,
 * by means of unitary transformations. Z is an (M+L)-by-(M+L) unitary
 * matrix and, R and A1 are M-by-M upper triangular matrices.
 *
 * The factorization is obtained by Householder's method. The kth
 * transformation matrix, Z(k), which is used to introduce zeros into
 * the (m-k)th row of A, is given in the form
 *
 *    Z(k) = ( I     0   ),
 *           ( 0  T(k) )
 *
 * where
 *
 *    T(k) = I - tau*u(k)*u(k)**H,   u(k) = (   1    ),
 *                                            (   0    )
 *                                            ( z(k) )
 *
 * tau is a scalar and z(k) is an l element vector. tau and z(k)
 * are chosen to annihilate the elements of the kth row of A2.
 *
 * The scalar tau is returned in the kth element of TAU and the vector
 * u(k) in the kth row of A2, such that the elements of z(k) are
 * in a(k, n-l), ..., a(k, n-1). The elements of R are returned in
 * the upper triangular part of A1.
 *
 * Z is given by
 *
 *    Z = Z(0) * Z(1) * ... * Z(m-1).
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     l     The number of columns of the matrix A containing the
 *                      meaningful part of the Householder vectors.
 *                      N-M >= l >= 0.
 * @param[in,out] A     Complex array, dimension (lda, n).
 *                      On entry, the leading M-by-N upper trapezoidal part of
 *                      the array A must contain the matrix to be factorized.
 *                      On exit, the leading M-by-M upper triangular part of A
 *                      contains the upper triangular matrix R, and elements
 *                      N-L to N-1 of the first M rows of A, with the array
 *                      TAU, represent the unitary matrix Z as a product of
 *                      M elementary reflectors.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    tau   Complex array, dimension (m).
 *                      The scalar factors of the elementary reflectors.
 * @param[out]    work  Complex array, dimension (m).
 */
void clatrz(const INT m, const INT n, const INT l,
            c64* restrict A, const INT lda,
            c64* restrict tau,
            c64* restrict work)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    INT i;
    c64 alpha;

    /* Quick return if possible */
    if (m == 0) {
        return;
    }

    if (m == n) {
        for (i = 0; i < n; i++) {
            tau[i] = ZERO;
        }
        return;
    }

    for (i = m - 1; i >= 0; i--) {

        /* Generate elementary reflector H(i) to annihilate
         * [ A(i,i) A(i,n-l:n-1) ]. */

        clacgv(l, &A[i + (n - l) * lda], lda);
        alpha = conjf(A[i + i * lda]);
        clarfg(l + 1, &alpha, &A[i + (n - l) * lda], lda, &tau[i]);
        tau[i] = conjf(tau[i]);

        /* Apply H(i) to A(0:i-1, i:n-1) from the right. */

        if (i > 0) {
            c64 conjtau = conjf(tau[i]);
            clarz("R", i, n - i, l, &A[i + (n - l) * lda], lda, conjtau,
                  &A[i * lda], lda, work);
        }
        A[i + i * lda] = conjf(alpha);
    }
}
