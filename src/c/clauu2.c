/**
 * @file clauu2.c
 * @brief CLAUU2 computes the product U * U**H or L**H * L (unblocked algorithm).
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CLAUU2 computes the product U * U**H or L**H * L, where the triangular
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
 *                      the lower triangle of the product L**H * L. Regardless,
 *                      the imaginary parts of the diagonal elements are set
 *                      to zero before exit.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void clauu2(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CLAUU2", -(*info));
        return;
    }

    if (n == 0) return;

    if (upper) {
        /* Compute the product U * U**H. */
        for (INT j = 0; j < n; j++) {
            c64 ajj = conjf(A[j + j * lda]);
            if (j == n - 1) {
                /* We have this case separate just to ensure we never
                 * access A out of bounds when we do A(J,J+1) = A(N,N+1) */
                cblas_cscal(n, &ajj, &A[(n - 1) * lda], 1);
            } else {
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                            j + 1, 1, n - j - 1, &ONE,
                            &A[(j + 1) * lda], lda,
                            &A[j + (j + 1) * lda], lda,
                            &ajj, &A[j * lda], lda);
            }
            /* Since in exact arithmetic the diagonal will be real on output
             * we explicitly cast the diagonal to real (zero out the
             * imaginary component) */
            A[j + j * lda] = CMPLXF(crealf(A[j + j * lda]), 0.0f);
        }
    } else {
        /* Compute the product L**H * L. */
        for (INT i = 0; i < n; i++) {
            c64 aii = conjf(A[i + i * lda]);
            if (i == n - 1) {
                /* We have this case separate just to ensure we never
                 * access A out of bounds when we do A(I+1,I) = A(N+1,N) */
                cblas_cscal(n, &aii, &A[n - 1], lda);
            } else {
                cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            1, i + 1, n - i - 1, &ONE,
                            &A[(i + 1) + i * lda], lda,
                            &A[i + 1], lda,
                            &aii, &A[i], lda);
            }
            /* Since in exact arithmetic the diagonal will be real on output
             * we explicitly cast the diagonal to real (zero out the
             * imaginary component) */
            A[i + i * lda] = CMPLXF(crealf(A[i + i * lda]), 0.0f);
        }
    }
}
