/**
 * @file zlaqsp.c
 * @brief ZLAQSP scales a symmetric matrix in packed storage, using scaling factors computed by zppequ.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLAQSP equilibrates a symmetric matrix A using the scaling factors
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
void zlaqsp(
    const char* uplo,
    const int n,
    double complex* const restrict AP,
    const double* const restrict S,
    const double scond,
    const double amax,
    char* equed)
{
    // zlaqsp.f lines 143-144: Parameters
    const double ONE = 1.0;
    const double THRESH = 0.1;  // zlaqsp.f line 144: THRESH = 0.1D+0

    // zlaqsp.f lines 147-148: Local Scalars
    int i, j, jc;
    double cj, large, small;

    // zlaqsp.f lines 159-162: Quick return if possible
    if (n <= 0) {
        *equed = 'N';
        return;
    }

    // zlaqsp.f lines 166-167: Initialize LARGE and SMALL
    small = dlamch("S") / dlamch("P");  // Safe minimum / Precision
    large = ONE / small;

    // zlaqsp.f lines 169-204: Check if equilibration is needed
    if (scond >= THRESH && amax >= small && amax <= large) {
        // zlaqsp.f lines 173: No equilibration
        *equed = 'N';
    } else {
        // zlaqsp.f lines 176-202: Replace A by diag(S) * A * diag(S)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // zlaqsp.f lines 182-189: Upper triangle of A is stored
            jc = 0;  // zlaqsp.f line 182: JC = 1 (0-based: jc = 0)
            for (j = 0; j < n; j++) {  // zlaqsp.f line 183: DO 20 J = 1, N
                cj = S[j];  // zlaqsp.f line 184
                for (i = 0; i <= j; i++) {  // zlaqsp.f line 185: DO 10 I = 1, J
                    AP[jc + i] = cj * S[i] * AP[jc + i];  // zlaqsp.f line 186
                }
                jc = jc + (j + 1);  // zlaqsp.f line 188: JC = JC + J
            }
        } else {
            // zlaqsp.f lines 194-201: Lower triangle of A is stored
            jc = 0;  // zlaqsp.f line 194: JC = 1 (0-based: jc = 0)
            for (j = 0; j < n; j++) {  // zlaqsp.f line 195: DO 40 J = 1, N
                cj = S[j];  // zlaqsp.f line 196
                for (i = j; i < n; i++) {  // zlaqsp.f line 197: DO 30 I = J, N
                    AP[jc + i - j] = cj * S[i] * AP[jc + i - j];  // zlaqsp.f line 198
                }
                jc = jc + (n - j);  // zlaqsp.f line 200: JC = JC + N - J + 1
            }
        }
        *equed = 'Y';  // zlaqsp.f line 203
    }
}
