/**
 * @file claqhe.c
 * @brief CLAQHE scales a Hermitian matrix.
 */

#include <complex.h>
#include <float.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLAQHE equilibrates a Hermitian matrix A using the scaling factors
 * in the vector S.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular
 *                       - `'L'`: Lower triangular
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in,out] A      Single complex array of dimension `(lda,n)`.
 *                       On entry, the Hermitian matrix A. If `uplo='U'`, the
 *                       leading n by n upper triangular part of A contains the
 *                       upper triangular part of the matrix A, and the strictly
 *                       lower triangular part of A is not referenced. If
 *                       `uplo='L'`, the leading n by n lower triangular part of
 *                       A contains the lower triangular part of the matrix A,
 *                       and the strictly upper triangular part of A is not
 *                       referenced.
 *                       On exit, if `equed='Y'`, the equilibrated matrix:
 *                       diag(S) * A * diag(S).
 * @param[in]     lda    The leading dimension of A. `lda>=max(n,1)`.
 * @param[in]     S      Single precision array of dimension `n`.
 *                       The scale factors for A.
 * @param[in]     scond  Ratio of the smallest S(i) to the largest S(i).
 * @param[in]     amax   Absolute value of largest matrix entry.
 * @param[out]    equed
 *                       - `'N'`: No equilibration
 *                       - `'Y'`: Equilibration was done, i.e., A has been replaced
 *                              by diag(S) * A * diag(S).
 *
 * @par Internal Parameters:
 * `THRESH` decides whether to scale based on the ratio of the scaling
 * factors: scaling is performed when `scond<THRESH`. `LARGE` and `SMALL`
 * decide whether to scale based on the absolute size of the largest
 * matrix element: scaling is performed when `amax>LARGE` or `amax<SMALL`.
 */
void claqhe(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    const f32* restrict S,
    const f32 scond,
    const f32 amax,
    char* equed)
{
    const f32 THRESH = 0.1f;

    // Quick return if possible
    if (n <= 0) {
        *equed = 'N';
        return;
    }

    // Initialize LARGE and SMALL
    f32 eps = FLT_EPSILON * 0.5f;
    f32 sfmin = FLT_MIN;
    f32 small_val = 1.0f / FLT_MAX;
    if (small_val >= sfmin) {
        sfmin = small_val * (1.0f + eps);
    }
    f32 small = sfmin / eps;
    f32 large = 1.0f / small;

    if (scond >= THRESH && amax >= small && amax <= large) {
        *equed = 'N';
    } else {
        // Replace A by diag(S) * A * diag(S)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // Upper triangle of A is stored
            for (INT j = 0; j < n; j++) {
                f32 cj = S[j];
                for (INT i = 0; i < j; i++) {
                    A[i + j * lda] = cj * S[i] * A[i + j * lda];
                }
                A[j + j * lda] = CMPLXF(cj * cj * crealf(A[j + j * lda]), 0.0f);
            }
        } else {
            // Lower triangle of A is stored
            for (INT j = 0; j < n; j++) {
                f32 cj = S[j];
                A[j + j * lda] = CMPLXF(cj * cj * crealf(A[j + j * lda]), 0.0f);
                for (INT i = j + 1; i < n; i++) {
                    A[i + j * lda] = cj * S[i] * A[i + j * lda];
                }
            }
        }
        *equed = 'Y';
    }
}
