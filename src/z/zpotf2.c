/**
 * @file zpotf2.c
 * @brief ZPOTF2 computes the Cholesky factorization of a Hermitian positive
 *        definite matrix (unblocked algorithm).
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPOTF2 computes the Cholesky factorization of a complex Hermitian
 * positive definite matrix A.
 *
 * The factorization has the form
 *    A = U**H * U,  if UPLO = 'U', or
 *    A = L  * L**H, if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part of
 *                      the Hermitian matrix A is stored.
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A. If UPLO = 'U', the
 *                      leading n by n upper triangular part of A contains the
 *                      upper triangular part of the matrix A, and the strictly
 *                      lower triangular part of A is not referenced. If
 *                      UPLO = 'L', the leading n by n lower triangular part of
 *                      A contains the lower triangular part of the matrix A,
 *                      and the strictly upper triangular part of A is not
 *                      referenced.
 *                      On exit, if info = 0, the factor U or L from the
 *                      Cholesky factorization A = U**H*U or A = L*L**H.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, the leading principal minor of order k
 *                           is not positive, and the factorization could not be
 *                           completed.
 */
void zpotf2(
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);

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
        xerbla("ZPOTF2", -(*info));
        return;
    }

    if (n == 0) return;

    if (upper) {
        /* Compute the Cholesky factorization A = U**H *U. */
        for (INT j = 0; j < n; j++) {
            /* Compute U(J,J) and test for non-positive-definiteness. */
            c128 zdotc_result = CMPLX(0.0, 0.0);
            if (j > 0) {
                cblas_zdotc_sub(j, &A[j * lda], 1, &A[j * lda], 1,
                                &zdotc_result);
            }
            f64 ajj = creal(A[j + j * lda]) - creal(zdotc_result);
            if (ajj <= ZERO || disnan(ajj)) {
                A[j + j * lda] = CMPLX(ajj, 0.0);
                *info = j + 1;
                return;
            }
            ajj = sqrt(ajj);
            A[j + j * lda] = CMPLX(ajj, 0.0);

            /* Compute elements J+1:N of row J. */
            if (j < n - 1) {
                zlacgv(j, &A[j * lda], 1);
                cblas_zgemv(CblasColMajor, CblasTrans,
                            j, n - j - 1, &NEG_CONE,
                            &A[(j + 1) * lda], lda,
                            &A[j * lda], 1,
                            &CONE, &A[j + (j + 1) * lda], lda);
                zlacgv(j, &A[j * lda], 1);
                cblas_zdscal(n - j - 1, ONE / ajj,
                             &A[j + (j + 1) * lda], lda);
            }
        }
    } else {
        /* Compute the Cholesky factorization A = L*L**H. */
        for (INT j = 0; j < n; j++) {
            /* Compute L(J,J) and test for non-positive-definiteness. */
            c128 zdotc_result = CMPLX(0.0, 0.0);
            if (j > 0) {
                cblas_zdotc_sub(j, &A[j], lda, &A[j], lda,
                                &zdotc_result);
            }
            f64 ajj = creal(A[j + j * lda]) - creal(zdotc_result);
            if (ajj <= ZERO || disnan(ajj)) {
                A[j + j * lda] = CMPLX(ajj, 0.0);
                *info = j + 1;
                return;
            }
            ajj = sqrt(ajj);
            A[j + j * lda] = CMPLX(ajj, 0.0);

            /* Compute elements J+1:N of column J. */
            if (j < n - 1) {
                zlacgv(j, &A[j], lda);
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                            n - j - 1, j, &NEG_CONE,
                            &A[j + 1], lda,
                            &A[j], lda,
                            &CONE, &A[(j + 1) + j * lda], 1);
                zlacgv(j, &A[j], lda);
                cblas_zdscal(n - j - 1, ONE / ajj,
                             &A[(j + 1) + j * lda], 1);
            }
        }
    }
}
