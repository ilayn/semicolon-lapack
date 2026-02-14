/**
 * @file zlauu2.c
 * @brief ZLAUU2 computes the product U * U**H or L**H * L (unblocked algorithm).
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAUU2 computes the product U * U**H or L**H * L, where the triangular
 * factor U or L is stored in the upper or lower triangular part of
 * the array A.
 *
 * If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
 * overwriting the factor U in A.
 * If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
 * overwriting the factor L in A.
 *
 * This is the unblocked form of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo  Specifies whether the triangular factor stored in the
 *                      array A is upper or lower triangular:
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the triangular factor U or L. n >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the triangular factor U or L.
 *                      On exit, if UPLO = 'U', the upper triangle of A is
 *                      overwritten with the upper triangle of the product U * U**H;
 *                      if UPLO = 'L', the lower triangle of A is overwritten with
 *                      the lower triangle of the product L**H * L.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void zlauu2(
    const char* uplo,
    const int n,
    double complex* const restrict A,
    const int lda,
    int* info)
{
    const double complex ONE = CMPLX(1.0, 0.0);

    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("ZLAUU2", -(*info));
        return;
    }

    if (n == 0) return;

    if (upper) {
        /* Compute the product U * U**H. */
        for (int i = 0; i < n; i++) {
            double aii = creal(A[i + i * lda]);
            if (i < n - 1) {
                double complex dotc;
                cblas_zdotc_sub(n - i - 1, &A[i + (i + 1) * lda], lda,
                                &A[i + (i + 1) * lda], lda, &dotc);
                A[i + i * lda] = CMPLX(aii * aii + creal(dotc), 0.0);
                zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                double complex alpha = CMPLX(aii, 0.0);
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                            i, n - i - 1, &ONE,
                            &A[(i + 1) * lda], lda,
                            &A[i + (i + 1) * lda], lda,
                            &alpha, &A[i * lda], 1);
                zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
            } else {
                cblas_zdscal(i + 1, aii, &A[i * lda], 1);
            }
        }
    } else {
        /* Compute the product L**H * L. */
        for (int i = 0; i < n; i++) {
            double aii = creal(A[i + i * lda]);
            if (i < n - 1) {
                double complex dotc;
                cblas_zdotc_sub(n - i - 1, &A[(i + 1) + i * lda], 1,
                                &A[(i + 1) + i * lda], 1, &dotc);
                A[i + i * lda] = CMPLX(aii * aii + creal(dotc), 0.0);
                zlacgv(i, &A[i], lda);
                double complex alpha = CMPLX(aii, 0.0);
                cblas_zgemv(CblasColMajor, CblasConjTrans,
                            n - i - 1, i, &ONE,
                            &A[(i + 1)], lda,
                            &A[(i + 1) + i * lda], 1,
                            &alpha, &A[i], lda);
                zlacgv(i, &A[i], lda);
            } else {
                cblas_zdscal(i + 1, aii, &A[i], lda);
            }
        }
    }
}
