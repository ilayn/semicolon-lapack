/**
 * @file claqsy.c
 * @brief CLAQSY equilibrates a symmetric matrix using scaling factors.
 */

#include <complex.h>
#include <float.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLAQSY equilibrates a symmetric matrix A using the scaling factors
 * in the vector S.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the symmetric matrix A is stored.
 *                       = 'U': Upper triangular
 *                       = 'L': Lower triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the symmetric matrix A. If UPLO = 'U', the
 *                       leading n by n upper triangular part of A contains the
 *                       upper triangular part of the matrix A, and the strictly
 *                       lower triangular part of A is not referenced. If
 *                       UPLO = 'L', the leading n by n lower triangular part of
 *                       A contains the lower triangular part of the matrix A,
 *                       and the strictly upper triangular part of A is not
 *                       referenced.
 *                       On exit, if equed = 'Y', the equilibrated matrix:
 *                       diag(S) * A * diag(S).
 * @param[in]     lda    The leading dimension of the array A. lda >= max(n,1).
 * @param[in]     S      Single precision array, dimension (n).
 *                       The scale factors for A.
 * @param[in]     scond  Ratio of the smallest S(i) to the largest S(i).
 * @param[in]     amax   Absolute value of largest matrix entry.
 * @param[out]    equed  Specifies whether or not equilibration was done.
 *                       = 'N': No equilibration.
 *                       = 'Y': Equilibration was done, i.e., A has been replaced
 *                              by diag(S) * A * diag(S).
 */
void claqsy(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    const f32* restrict S,
    const f32 scond,
    const f32 amax,
    char* equed)
{
    // Internal parameters
    // THRESH is a threshold value used to decide if scaling should be done
    const f32 THRESH = 0.1f;

    // Quick return if possible
    if (n <= 0) {
        *equed = 'N';
        return;
    }

    // Initialize LARGE and SMALL
    // SMALL = SLAMCH('Safe minimum') / SLAMCH('Precision')
    // LARGE = 1 / SMALL
    f32 eps = FLT_EPSILON * 0.5f;
    f32 sfmin = FLT_MIN;
    f32 small_val = 1.0f / FLT_MAX;
    if (small_val >= sfmin) {
        sfmin = small_val * (1.0f + eps);
    }
    f32 small = sfmin / eps;
    f32 large = 1.0f / small;

    if (scond >= THRESH && amax >= small && amax <= large) {
        // No equilibration
        *equed = 'N';
    } else {
        // Replace A by diag(S) * A * diag(S)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // Upper triangle of A is stored
            for (INT j = 0; j < n; j++) {
                f32 cj = S[j];
                for (INT i = 0; i <= j; i++) {
                    A[i + j * lda] = cj * S[i] * A[i + j * lda];
                }
            }
        } else {
            // Lower triangle of A is stored
            for (INT j = 0; j < n; j++) {
                f32 cj = S[j];
                for (INT i = j; i < n; i++) {
                    A[i + j * lda] = cj * S[i] * A[i + j * lda];
                }
            }
        }
        *equed = 'Y';
    }
}
