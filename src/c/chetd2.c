/**
 * @file chetd2.c
 * @brief CHETD2 reduces a Hermitian matrix to real symmetric tridiagonal
 *        form by a unitary similarity transformation (unblocked algorithm).
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETD2 reduces a complex Hermitian matrix A to real symmetric
 * tridiagonal form T by a unitary similarity transformation:
 * Q**H * A * Q = T.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the Hermitian matrix A is stored:
 *                       = 'U': Upper triangular
 *                       = 'L': Lower triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the Hermitian matrix A. If uplo = 'U', the
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
 *                       represent the unitary matrix Q as a product of
 *                       elementary reflectors; if uplo = 'L', the diagonal and
 *                       first subdiagonal of A are overwritten by the
 *                       corresponding elements of the tridiagonal matrix T, and
 *                       the elements below the first subdiagonal, with the array
 *                       tau, represent the unitary matrix Q as a product of
 *                       elementary reflectors. See Further Details.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    D      Single precision array, dimension (n).
 *                       The diagonal elements of the tridiagonal matrix T:
 *                       D[i] = A[i + i*lda].
 * @param[out]    E      Single precision array, dimension (n-1).
 *                       The off-diagonal elements of the tridiagonal matrix T:
 *                       E[i] = A[i + (i+1)*lda] if uplo = 'U',
 *                       E[i] = A[(i+1) + i*lda] if uplo = 'L'.
 * @param[out]    tau    Complex*16 array, dimension (n-1).
 *                       The scalar factors of the elementary reflectors.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void chetd2(const char* uplo, const INT n, c64* restrict A,
            const INT lda, f32* restrict D, f32* restrict E,
            c64* restrict tau, INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 HALF = CMPLXF(0.5f, 0.0f);

    INT upper;
    INT i;
    c64 alpha, taui;
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
        xerbla("CHETD2", -(*info));
        return;
    }

    /* Quick return if possible. */
    if (n <= 0) {
        return;
    }

    if (upper) {
        /* Reduce the upper triangle of A. */
        cblas_uplo = CblasUpper;

        A[(n - 1) + (n - 1) * lda] = CMPLXF(crealf(A[(n - 1) + (n - 1) * lda]), 0.0f);
        for (i = n - 2; i >= 0; i--) {

            /* Generate elementary reflector H(i) = I - tau * v * v**H
             * to annihilate A(0:i-1, i+1). */
            alpha = A[i + (i + 1) * lda];
            clarfg(i + 1, &alpha, &A[(i + 1) * lda], 1, &taui);
            E[i] = crealf(alpha);

            if (taui != ZERO) {

                /* Apply H(i) from both sides to A(0:i, 0:i) */
                A[i + (i + 1) * lda] = ONE;

                /* Compute x := tau * A * v, storing x in tau(0:i) */
                cblas_chemv(CblasColMajor, cblas_uplo, i + 1, &taui,
                            A, lda, &A[(i + 1) * lda], 1,
                            &ZERO, tau, 1);

                /* Compute w := x - 1/2 * tau * (x**H * v) * v */
                c64 dotc;
                cblas_cdotc_sub(i + 1, tau, 1, &A[(i + 1) * lda], 1, &dotc);
                alpha = -HALF * taui * dotc;
                cblas_caxpy(i + 1, &alpha, &A[(i + 1) * lda], 1, tau, 1);

                /* Apply the transformation as a rank-2 update:
                 *   A := A - v * w**H - w * v**H */
                c64 neg_one = CMPLXF(-1.0f, 0.0f);
                cblas_cher2(CblasColMajor, cblas_uplo, i + 1, &neg_one,
                            &A[(i + 1) * lda], 1, tau, 1, A, lda);

            } else {
                A[i + i * lda] = CMPLXF(crealf(A[i + i * lda]), 0.0f);
            }
            A[i + (i + 1) * lda] = CMPLXF(E[i], 0.0f);
            D[i + 1] = crealf(A[(i + 1) + (i + 1) * lda]);
            tau[i] = taui;
        }
        D[0] = crealf(A[0]);
    } else {
        /* Reduce the lower triangle of A. */
        cblas_uplo = CblasLower;

        A[0] = CMPLXF(crealf(A[0]), 0.0f);
        for (i = 0; i <= n - 2; i++) {
            INT ni = n - i - 1;
            INT x_start = (i + 2 < n) ? (i + 2) : (n - 1);

            /* Generate elementary reflector H(i) = I - tau * v * v**H
             * to annihilate A(i+2:n-1, i). */
            alpha = A[(i + 1) + i * lda];
            clarfg(ni, &alpha, &A[x_start + i * lda], 1, &taui);
            E[i] = crealf(alpha);

            if (taui != ZERO) {

                /* Apply H(i) from both sides to A(i+1:n-1, i+1:n-1) */
                A[(i + 1) + i * lda] = ONE;

                /* Compute x := tau * A * v, storing x in tau(i:n-2) */
                cblas_chemv(CblasColMajor, cblas_uplo, ni, &taui,
                            &A[(i + 1) + (i + 1) * lda], lda,
                            &A[(i + 1) + i * lda], 1,
                            &ZERO, &tau[i], 1);

                /* Compute w := x - 1/2 * tau * (x**H * v) * v */
                c64 dotc;
                cblas_cdotc_sub(ni, &tau[i], 1, &A[(i + 1) + i * lda], 1, &dotc);
                alpha = -HALF * taui * dotc;
                cblas_caxpy(ni, &alpha, &A[(i + 1) + i * lda], 1, &tau[i], 1);

                /* Apply the transformation as a rank-2 update:
                 *   A := A - v * w**H - w * v**H */
                c64 neg_one = CMPLXF(-1.0f, 0.0f);
                cblas_cher2(CblasColMajor, cblas_uplo, ni, &neg_one,
                            &A[(i + 1) + i * lda], 1, &tau[i], 1,
                            &A[(i + 1) + (i + 1) * lda], lda);

            } else {
                A[(i + 1) + (i + 1) * lda] = CMPLXF(crealf(A[(i + 1) + (i + 1) * lda]), 0.0f);
            }
            A[(i + 1) + i * lda] = CMPLXF(E[i], 0.0f);
            D[i] = crealf(A[i + i * lda]);
            tau[i] = taui;
        }
        D[n - 1] = crealf(A[(n - 1) + (n - 1) * lda]);
    }
}
