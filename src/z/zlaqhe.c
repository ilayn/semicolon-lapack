/**
 * @file zlaqhe.c
 * @brief ZLAQHE scales a Hermitian matrix.
 */

#include <complex.h>
#include <float.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAQHE equilibrates a Hermitian matrix A using the scaling factors
 * in the vector S.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the Hermitian matrix A is stored.
 *                       = 'U': Upper triangular
 *                       = 'L': Lower triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the Hermitian matrix A. If UPLO = 'U', the
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
 * @param[in]     S      Double precision array, dimension (n).
 *                       The scale factors for A.
 * @param[in]     scond  Ratio of the smallest S(i) to the largest S(i).
 * @param[in]     amax   Absolute value of largest matrix entry.
 * @param[out]    equed  Specifies whether or not equilibration was done.
 *                       = 'N': No equilibration.
 *                       = 'Y': Equilibration was done, i.e., A has been replaced
 *                              by diag(S) * A * diag(S).
 */
void zlaqhe(
    const char* uplo,
    const int n,
    c128* const restrict A,
    const int lda,
    const f64* const restrict S,
    const f64 scond,
    const f64 amax,
    char* equed)
{
    const f64 THRESH = 0.1;

    // Quick return if possible
    if (n <= 0) {
        *equed = 'N';
        return;
    }

    // Initialize LARGE and SMALL
    f64 eps = DBL_EPSILON * 0.5;
    f64 sfmin = DBL_MIN;
    f64 small_val = 1.0 / DBL_MAX;
    if (small_val >= sfmin) {
        sfmin = small_val * (1.0 + eps);
    }
    f64 small = sfmin / eps;
    f64 large = 1.0 / small;

    if (scond >= THRESH && amax >= small && amax <= large) {
        *equed = 'N';
    } else {
        // Replace A by diag(S) * A * diag(S)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // Upper triangle of A is stored
            for (int j = 0; j < n; j++) {
                f64 cj = S[j];
                for (int i = 0; i < j; i++) {
                    A[i + j * lda] = cj * S[i] * A[i + j * lda];
                }
                A[j + j * lda] = CMPLX(cj * cj * creal(A[j + j * lda]), 0.0);
            }
        } else {
            // Lower triangle of A is stored
            for (int j = 0; j < n; j++) {
                f64 cj = S[j];
                A[j + j * lda] = CMPLX(cj * cj * creal(A[j + j * lda]), 0.0);
                for (int i = j + 1; i < n; i++) {
                    A[i + j * lda] = cj * S[i] * A[i + j * lda];
                }
            }
        }
        *equed = 'Y';
    }
}
