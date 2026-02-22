/**
 * @file zlaqhp.c
 * @brief ZLAQHP scales a Hermitian matrix stored in packed form.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLAQHP equilibrates a Hermitian matrix A using the scaling factors
 * in the vector S.
 *
 * @param[in]     uplo   = 'U': Upper triangular part of A is stored
 *                        = 'L': Lower triangular part of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the upper or lower triangle of the Hermitian
 *                       matrix A, packed columnwise in a linear array.
 *                       The j-th column of A is stored in the array AP as follows:
 *                       if uplo = 'U', AP(i + (j-1)*j/2) = A(i,j) for 0<=i<=j;
 *                       if uplo = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<n.
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
void zlaqhp(
    const char* uplo,
    const INT n,
    c128* restrict AP,
    const f64* restrict S,
    const f64 scond,
    const f64 amax,
    char* equed)
{
    const f64 ONE = 1.0;
    const f64 THRESH = 0.1;

    INT i, j, jc;
    f64 cj, large, small;

    if (n <= 0) {
        *equed = 'N';
        return;
    }

    small = dlamch("S") / dlamch("P");
    large = ONE / small;

    if (scond >= THRESH && amax >= small && amax <= large) {
        *equed = 'N';
    } else {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            jc = 0;
            for (j = 0; j < n; j++) {
                cj = S[j];
                for (i = 0; i < j; i++) {
                    AP[jc + i] = cj * S[i] * AP[jc + i];
                }
                AP[jc + j] = CMPLX(cj * cj * creal(AP[jc + j]), 0.0);
                jc = jc + (j + 1);
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                cj = S[j];
                AP[jc] = CMPLX(cj * cj * creal(AP[jc]), 0.0);
                for (i = j + 1; i < n; i++) {
                    AP[jc + i - j] = cj * S[i] * AP[jc + i - j];
                }
                jc = jc + (n - j);
            }
        }
        *equed = 'Y';
    }
}
