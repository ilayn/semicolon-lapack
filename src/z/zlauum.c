/**
 * @file zlauum.c
 * @brief ZLAUUM computes the product U * U**H or L**H * L (blocked algorithm).
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"

/**
 * ZLAUUM computes the product U * U**H or L**H * L, where the triangular
 * factor U or L is stored in the upper or lower triangular part of
 * the array A.
 *
 * If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
 * overwriting the factor U in A.
 * If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
 * overwriting the factor L in A.
 *
 * This is the blocked form of the algorithm, calling Level 3 BLAS.
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
void zlauum(
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const f64 ONE = 1.0;

    // Test the input parameters
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
        xerbla("ZLAUUM", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    // Determine block size (NB=64 from ilaenv for LAUUM)
    INT nb = lapack_get_nb("LAUUM");

    if (nb <= 1 || nb >= n) {
        // Use unblocked code
        zlauu2(uplo, n, A, lda, info);
    } else {
        // Use blocked code
        if (upper) {
            // Compute the product U * U**H
            for (INT i = 0; i < n; i += nb) {
                INT ib = nb < (n - i) ? nb : (n - i);

                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans,
                            CblasNonUnit, i, ib, &CONE,
                            &A[i + i * lda], lda, &A[i * lda], lda);

                zlauu2("U", ib, &A[i + i * lda], lda, info);

                if (i + ib < n) {
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                i, ib, n - i - ib, &CONE,
                                &A[(i + ib) * lda], lda,
                                &A[i + (i + ib) * lda], lda,
                                &CONE, &A[i * lda], lda);

                    cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                ib, n - i - ib, ONE,
                                &A[i + (i + ib) * lda], lda,
                                ONE, &A[i + i * lda], lda);
                }
            }
        } else {
            // Compute the product L**H * L
            for (INT i = 0; i < n; i += nb) {
                INT ib = nb < (n - i) ? nb : (n - i);

                cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                            CblasNonUnit, ib, i, &CONE,
                            &A[i + i * lda], lda, &A[i], lda);

                zlauu2("L", ib, &A[i + i * lda], lda, info);

                if (i + ib < n) {
                    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                ib, i, n - i - ib, &CONE,
                                &A[(i + ib) + i * lda], lda,
                                &A[(i + ib)], lda,
                                &CONE, &A[i], lda);

                    cblas_zherk(CblasColMajor, CblasLower, CblasConjTrans,
                                ib, n - i - ib, ONE,
                                &A[(i + ib) + i * lda], lda,
                                ONE, &A[i + i * lda], lda);
                }
            }
        }
    }
}
