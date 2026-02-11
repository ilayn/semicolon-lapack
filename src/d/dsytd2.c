/**
 * @file dsytd2.c
 * @brief DSYTD2 reduces a symmetric matrix to real symmetric tridiagonal
 *        form by an orthogonal similarity transformation (unblocked algorithm).
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSYTD2 reduces a real symmetric matrix A to symmetric tridiagonal
 * form T by an orthogonal similarity transformation: Q**T * A * Q = T.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the symmetric matrix A is stored:
 *                       = 'U': Upper triangular
 *                       = 'L': Lower triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the symmetric matrix A. If uplo = 'U', the
 *                       leading n-by-n upper triangular part of A contains the
 *                       upper triangular part of the matrix A, and the strictly
 *                       lower triangular part of A is not referenced. If
 *                       uplo = 'L', the leading n-by-n lower triangular part of
 *                       A contains the lower triangular part of the matrix A,
 *                       and the strictly upper triangular part of A is not
 *                       referenced.
 *                       On exit, if uplo = 'U', the diagonal and first
 *                       superdiagonal of A are overwritten by the corresponding
 *                       elements of the tridiagonal matrix T, and the elements
 *                       above the first superdiagonal, with the array tau,
 *                       represent the orthogonal matrix Q as a product of
 *                       elementary reflectors; if uplo = 'L', the diagonal and
 *                       first subdiagonal of A are overwritten by the
 *                       corresponding elements of the tridiagonal matrix T, and
 *                       the elements below the first subdiagonal, with the array
 *                       tau, represent the orthogonal matrix Q as a product of
 *                       elementary reflectors.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    D      Double precision array, dimension (n).
 *                       The diagonal elements of the tridiagonal matrix T:
 *                       D[i] = A[i + i*lda].
 * @param[out]    E      Double precision array, dimension (n-1).
 *                       The off-diagonal elements of the tridiagonal matrix T:
 *                       E[i] = A[i + (i+1)*lda] if uplo = 'U',
 *                       E[i] = A[(i+1) + i*lda] if uplo = 'L'.
 * @param[out]    tau    Double precision array, dimension (n-1).
 *                       The scalar factors of the elementary reflectors.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dsytd2(const char* uplo, const int n, double* const restrict A,
            const int lda, double* const restrict D, double* const restrict E,
            double* const restrict tau, int* info)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;
    const double HALF = 0.5;

    int upper;
    int i;
    double alpha, taui;
    CBLAS_UPLO cblas_uplo;

    /* Test the input parameters. */
    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("DSYTD2", -(*info));
        return;
    }

    /* Quick return if possible. */
    if (n <= 0) {
        return;
    }

    if (upper) {
        /* Reduce the upper triangle of A. */
        cblas_uplo = CblasUpper;

        for (i = n - 2; i >= 0; i--) {
            /* Generate elementary reflector H(i) = I - tau * v * v**T
             * to annihilate A(0:i-1, i+1). */
            dlarfg(i + 1, &A[i + (i + 1) * lda], &A[(i + 1) * lda], 1, &taui);
            E[i] = A[i + (i + 1) * lda];

            if (taui != ZERO) {
                /* Apply H(i) from both sides to A(0:i, 0:i) */
                A[i + (i + 1) * lda] = ONE;

                /* Compute x := tau * A * v, storing x in tau(0:i) */
                cblas_dsymv(CblasColMajor, cblas_uplo, i + 1, taui,
                            A, lda, &A[(i + 1) * lda], 1,
                            ZERO, tau, 1);

                /* Compute w := x - 1/2 * tau * (x**T * v) * v */
                alpha = -HALF * taui * cblas_ddot(i + 1, tau, 1,
                                                  &A[(i + 1) * lda], 1);
                cblas_daxpy(i + 1, alpha, &A[(i + 1) * lda], 1, tau, 1);

                /* Apply the transformation as a rank-2 update:
                 *   A := A - v * w**T - w * v**T */
                cblas_dsyr2(CblasColMajor, cblas_uplo, i + 1, -ONE,
                            &A[(i + 1) * lda], 1, tau, 1, A, lda);

                A[i + (i + 1) * lda] = E[i];
            }
            D[i + 1] = A[(i + 1) + (i + 1) * lda];
            tau[i] = taui;
        }
        D[0] = A[0];
    } else {
        /* Reduce the lower triangle of A. */
        cblas_uplo = CblasLower;

        for (i = 0; i <= n - 2; i++) {
            int ni = n - i - 1;
            int x_start = (i + 2 < n) ? (i + 2) : (n - 1);

            /* Generate elementary reflector H(i) = I - tau * v * v**T
             * to annihilate A(i+2:n-1, i). */
            dlarfg(ni, &A[(i + 1) + i * lda], &A[x_start + i * lda], 1, &taui);
            E[i] = A[(i + 1) + i * lda];

            if (taui != ZERO) {
                /* Apply H(i) from both sides to A(i+1:n-1, i+1:n-1) */
                A[(i + 1) + i * lda] = ONE;

                /* Compute x := tau * A * v, storing x in tau(i:i+ni-1) */
                cblas_dsymv(CblasColMajor, cblas_uplo, ni, taui,
                            &A[(i + 1) + (i + 1) * lda], lda,
                            &A[(i + 1) + i * lda], 1,
                            ZERO, &tau[i], 1);

                /* Compute w := x - 1/2 * tau * (x**T * v) * v */
                alpha = -HALF * taui * cblas_ddot(ni, &tau[i], 1,
                                                  &A[(i + 1) + i * lda], 1);
                cblas_daxpy(ni, alpha, &A[(i + 1) + i * lda], 1, &tau[i], 1);

                /* Apply the transformation as a rank-2 update:
                 *   A := A - v * w**T - w * v**T */
                cblas_dsyr2(CblasColMajor, cblas_uplo, ni, -ONE,
                            &A[(i + 1) + i * lda], 1, &tau[i], 1,
                            &A[(i + 1) + (i + 1) * lda], lda);

                A[(i + 1) + i * lda] = E[i];
            }
            D[i] = A[i + i * lda];
            tau[i] = taui;
        }
        D[n - 1] = A[(n - 1) + (n - 1) * lda];
    }
}
