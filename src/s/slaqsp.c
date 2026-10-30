/**
 * @file slaqsp.c
 * @brief SLAQSP scales a symmetric matrix in packed storage, using scaling factors computed by sppequ.
 */

#include "semicolon_lapack_single.h"

/**
 * SLAQSP equilibrates a symmetric matrix A using the scaling factors
 * in the vector S.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular part of A is stored
 *                       - `'L'`: Lower triangular part of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in,out] AP     Array of dimension `n*(n+1)/2`.
 *                       On entry, the upper or lower triangle of the symmetric
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, the equilibrated matrix: diag(S) * A * diag(S),
 *                       in the same storage format as A.
 * @param[in]     S      Array of dimension `n`.
 *                       The scale factors for A.
 * @param[in]     scond  Ratio of the smallest S(i) to the largest S(i).
 * @param[in]     amax   Absolute value of largest matrix entry.
 * @param[out]    equed
 *                       - `'N'`: No equilibration
 *                       - `'Y'`: Equilibration was done, i.e., A has been replaced
 *                         by diag(S) * A * diag(S)
 */
void slaqsp(
    const char* uplo,
    const INT n,
    f32* restrict AP,
    const f32* restrict S,
    const f32 scond,
    const f32 amax,
    char* equed)
{
    const f32 ONE = 1.0f;
    const f32 THRESH = 0.1f;

    INT i, j, jc;
    f32 cj, large, small;

    // Quick return if possible
    if (n <= 0) {
        *equed = 'N';
        return;
    }

    // Initialize LARGE and SMALL
    small = slamch("S") / slamch("P");
    large = ONE / small;

    if (scond >= THRESH && amax >= small && amax <= large) {
        // No equilibration
        *equed = 'N';
    } else {
        // Replace A by diag(S) * A * diag(S)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // Upper triangle of A is stored
            jc = 0;
            for (j = 0; j < n; j++) {
                cj = S[j];
                for (i = 0; i <= j; i++) {
                    AP[jc + i] = cj * S[i] * AP[jc + i];
                }
                jc = jc + (j + 1);
            }
        } else {
            // Lower triangle of A is stored
            jc = 0;
            for (j = 0; j < n; j++) {
                cj = S[j];
                for (i = j; i < n; i++) {
                    AP[jc + i - j] = cj * S[i] * AP[jc + i - j];
                }
                jc = jc + (n - j);
            }
        }
        *equed = 'Y';
    }
}
