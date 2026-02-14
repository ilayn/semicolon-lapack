/**
 * @file dlaqsp.c
 * @brief DLAQSP scales a symmetric matrix in packed storage, using scaling factors computed by dppequ.
 */

#include "semicolon_lapack_double.h"

/**
 * DLAQSP equilibrates a symmetric matrix A using the scaling factors
 * in the vector S.
 *
 * @param[in]     uplo   = 'U': Upper triangular part of A is stored
 *                        = 'L': Lower triangular part of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the upper or lower triangle of the symmetric
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, the equilibrated matrix: diag(S) * A * diag(S),
 *                       in the same storage format as A.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     S      The scale factors for A. Array of dimension (n).
 * @param[in]     scond  Ratio of the smallest S(i) to the largest S(i).
 * @param[in]     amax   Absolute value of largest matrix entry.
 * @param[out]    equed  Specifies whether or not equilibration was done:
 *                       = 'N': No equilibration
 *                       = 'Y': Equilibration was done, i.e., A has been replaced
 *                              by diag(S) * A * diag(S).
 */
void dlaqsp(
    const char* uplo,
    const int n,
    f64* restrict AP,
    const f64* restrict S,
    const f64 scond,
    const f64 amax,
    char* equed)
{
    // dlaqsp.f lines 141-142: Parameters
    const f64 ONE = 1.0;
    const f64 THRESH = 0.1;  // dlaqsp.f line 142: THRESH = 0.1D+0

    // dlaqsp.f lines 145-146: Local Scalars
    int i, j, jc;
    f64 cj, large, small;

    // dlaqsp.f lines 157-160: Quick return if possible
    if (n <= 0) {
        *equed = 'N';
        return;
    }

    // dlaqsp.f lines 164-165: Initialize LARGE and SMALL
    small = dlamch("S") / dlamch("P");  // Safe minimum / Precision
    large = ONE / small;

    // dlaqsp.f lines 167-202: Check if equilibration is needed
    if (scond >= THRESH && amax >= small && amax <= large) {
        // dlaqsp.f lines 169-171: No equilibration
        *equed = 'N';
    } else {
        // dlaqsp.f lines 174-200: Replace A by diag(S) * A * diag(S)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // dlaqsp.f lines 178-187: Upper triangle of A is stored
            jc = 0;  // dlaqsp.f line 180: JC = 1 (0-based: jc = 0)
            for (j = 0; j < n; j++) {  // dlaqsp.f line 181: DO 20 J = 1, N
                cj = S[j];  // dlaqsp.f line 182
                for (i = 0; i <= j; i++) {  // dlaqsp.f line 183: DO 10 I = 1, J
                    AP[jc + i] = cj * S[i] * AP[jc + i];  // dlaqsp.f line 184
                }
                jc = jc + (j + 1);  // dlaqsp.f line 186: JC = JC + J
            }
        } else {
            // dlaqsp.f lines 190-199: Lower triangle of A is stored
            jc = 0;  // dlaqsp.f line 192: JC = 1 (0-based: jc = 0)
            for (j = 0; j < n; j++) {  // dlaqsp.f line 193: DO 40 J = 1, N
                cj = S[j];  // dlaqsp.f line 194
                for (i = j; i < n; i++) {  // dlaqsp.f line 195: DO 30 I = J, N
                    AP[jc + i - j] = cj * S[i] * AP[jc + i - j];  // dlaqsp.f line 196
                }
                jc = jc + (n - j);  // dlaqsp.f line 198: JC = JC + N - J + 1
            }
        }
        *equed = 'Y';  // dlaqsp.f line 201
    }
}
