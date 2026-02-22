/**
 * @file slauum.c
 * @brief SLAUUM computes the product U * U**T or L**T * L (blocked algorithm).
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SLAUUM computes the product U * U**T or L**T * L, where the triangular
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
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the triangular factor U or L.
 *                      On exit, if UPLO = 'U', the upper triangle of A is
 *                      overwritten with the upper triangle of the product U * U**T;
 *                      if UPLO = 'L', the lower triangle of A is overwritten with
 *                      the lower triangle of the product L**T * L.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void slauum(
    const char* uplo,
    const INT n,
    f32* restrict A,
    const INT lda,
    INT* info)
{
    const f32 ONE = 1.0f;

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
        xerbla("SLAUUM", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    // Determine block size (NB=64 from ilaenv for LAUUM)
    INT nb = lapack_get_nb("LAUUM");

    if (nb <= 1 || nb >= n) {
        // Use unblocked code
        slauu2(uplo, n, A, lda, info);
    } else {
        // Use blocked code
        if (upper) {
            // Compute the product U * U**T
            for (INT i = 0; i < n; i += nb) {
                INT ib = nb < (n - i) ? nb : (n - i);

                // Fortran: DTRMM('Right', 'Upper', 'Transpose', 'Non-unit', I-1, IB, ONE, A(I,I), LDA, A(1,I), LDA)
                // 0-based: m = i, n = ib
                cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                            CblasNonUnit, i, ib, ONE,
                            &A[i + i * lda], lda, &A[i * lda], lda);

                // Factor the diagonal block
                slauu2("U", ib, &A[i + i * lda], lda, info);

                if (i + ib < n) {
                    // Fortran: DGEMM('N', 'T', I-1, IB, N-I-IB+1, ONE, A(1,I+IB), LDA, A(I,I+IB), LDA, ONE, A(1,I), LDA)
                    // 0-based: m = i, n = ib, k = n-i-ib
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                i, ib, n - i - ib, ONE,
                                &A[(i + ib) * lda], lda,
                                &A[i + (i + ib) * lda], lda,
                                ONE, &A[i * lda], lda);

                    // Fortran: DSYRK('Upper', 'N', IB, N-I-IB+1, ONE, A(I,I+IB), LDA, ONE, A(I,I), LDA)
                    // 0-based: n = ib, k = n-i-ib
                    cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                ib, n - i - ib, ONE,
                                &A[i + (i + ib) * lda], lda,
                                ONE, &A[i + i * lda], lda);
                }
            }
        } else {
            // Compute the product L**T * L
            for (INT i = 0; i < n; i += nb) {
                INT ib = nb < (n - i) ? nb : (n - i);

                // Fortran: DTRMM('Left', 'Lower', 'Transpose', 'Non-unit', IB, I-1, ONE, A(I,I), LDA, A(I,1), LDA)
                // 0-based: m = ib, n = i
                cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                            CblasNonUnit, ib, i, ONE,
                            &A[i + i * lda], lda, &A[i], lda);

                // Factor the diagonal block
                slauu2("L", ib, &A[i + i * lda], lda, info);

                if (i + ib < n) {
                    // Fortran: DGEMM('T', 'N', IB, I-1, N-I-IB+1, ONE, A(I+IB,I), LDA, A(I+IB,1), LDA, ONE, A(I,1), LDA)
                    // 0-based: m = ib, n = i, k = n-i-ib
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                ib, i, n - i - ib, ONE,
                                &A[(i + ib) + i * lda], lda,
                                &A[(i + ib)], lda,
                                ONE, &A[i], lda);

                    // Fortran: DSYRK('Lower', 'T', IB, N-I-IB+1, ONE, A(I+IB,I), LDA, ONE, A(I,I), LDA)
                    // 0-based: n = ib, k = n-i-ib
                    cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans,
                                ib, n - i - ib, ONE,
                                &A[(i + ib) + i * lda], lda,
                                ONE, &A[i + i * lda], lda);
                }
            }
        }
    }
}
