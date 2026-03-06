/**
 * @file cunt03.c
 * @brief CUNT03 compares two unitary matrices U and V to see if their
 *        corresponding rows or columns span the same spaces.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CUNT03 compares two unitary matrices U and V to see if their
 * corresponding rows or columns span the same spaces. The rows are
 * checked if RC = 'R', and the columns are checked if RC = 'C'.
 *
 * RESULT is the maximum of
 *
 *    | V*V' - I | / ( MV ulp ), if RC = 'R', or
 *
 *    | V'*V - I | / ( MV ulp ), if RC = 'C',
 *
 * and the maximum over rows (or columns) 1 to K of
 *
 *    | U(i) - S*V(i) |/ ( N ulp )
 *
 * where abs(S) = 1 (chosen to minimize the expression), U(i) is the
 * i-th row (column) of U, and V(i) is the i-th row (column) of V.
 *
 * @param[in]     rc     'R': compare rows; 'C': compare columns.
 * @param[in]     mu     Number of rows (rc='R') or columns (rc='C') of U.
 * @param[in]     mv     Number of rows (rc='R') or columns (rc='C') of V.
 * @param[in]     n      Number of columns (rc='R') or rows (rc='C') in U and V.
 * @param[in]     k      Number of rows or columns of U and V to compare.
 * @param[in]     U      Complex array, dimension (ldu, *).
 * @param[in]     ldu    Leading dimension of U.
 * @param[in]     V      Complex array, dimension (ldv, *).
 * @param[in]     ldv    Leading dimension of V.
 * @param[out]    work   Complex workspace, dimension (lwork).
 * @param[in]     lwork  Length of work array.
 * @param[out]    rwork  Double workspace, dimension (max(mv, n)).
 * @param[out]    result The computed test ratio.
 * @param[out]    info   0: success; < 0: illegal value.
 */
void cunt03(const char* rc, const INT mu, const INT mv, const INT n,
            const INT k, const c64* const restrict U, const INT ldu,
            const c64* const restrict V, const INT ldv,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork, f32* result, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT irc;
    INT i, j, lmx;
    f32 res1, res2, ulp;
    c64 s, su, sv;

    /* Check inputs */
    *info = 0;
    if (rc[0] == 'R' || rc[0] == 'r') {
        irc = 0;
    } else if (rc[0] == 'C' || rc[0] == 'c') {
        irc = 1;
    } else {
        irc = -1;
    }

    if (irc == -1) {
        *info = -1;
    } else if (mu < 0) {
        *info = -2;
    } else if (mv < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > (mu > mv ? mu : mv)) {
        *info = -5;
    } else if ((irc == 0 && ldu < (1 > mu ? 1 : mu)) ||
               (irc == 1 && ldu < (1 > n ? 1 : n))) {
        *info = -7;
    } else if ((irc == 0 && ldv < (1 > mv ? 1 : mv)) ||
               (irc == 1 && ldv < (1 > n ? 1 : n))) {
        *info = -9;
    }

    if (*info != 0) {
        return;
    }

    /* Initialize result */
    *result = ZERO;
    if (mu == 0 || mv == 0 || n == 0)
        return;

    /* Machine constants */
    ulp = slamch("P");

    if (irc == 0) {

        /* Compare rows */

        res1 = ZERO;
        for (i = 0; i < k; i++) {
            lmx = cblas_icamax(n, &U[i], ldu);
            if (crealf(V[i + lmx * ldv]) == 0.0f &&
                cimagf(V[i + lmx * ldv]) == 0.0f) {
                sv = ONE;
            } else {
                sv = cabsf(V[i + lmx * ldv]) / V[i + lmx * ldv];
            }
            if (crealf(U[i + lmx * ldu]) == 0.0f &&
                cimagf(U[i + lmx * ldu]) == 0.0f) {
                su = ONE;
            } else {
                su = cabsf(U[i + lmx * ldu]) / U[i + lmx * ldu];
            }
            s = sv / su;
            for (j = 0; j < n; j++) {
                res1 = fmaxf(res1, cabsf(U[i + j * ldu] - s * V[i + j * ldv]));
            }
        }
        res1 = res1 / ((f32)n * ulp);

        /* Compute orthogonality of rows of V. */

        cunt01("R", mv, n, V, ldv, work, lwork, rwork, &res2);

    } else {

        /* Compare columns */

        res1 = ZERO;
        for (i = 0; i < k; i++) {
            lmx = cblas_icamax(n, &U[i * ldu], 1);
            if (crealf(V[lmx + i * ldv]) == 0.0f &&
                cimagf(V[lmx + i * ldv]) == 0.0f) {
                sv = ONE;
            } else {
                sv = cabsf(V[lmx + i * ldv]) / V[lmx + i * ldv];
            }
            if (crealf(U[lmx + i * ldu]) == 0.0f &&
                cimagf(U[lmx + i * ldu]) == 0.0f) {
                su = ONE;
            } else {
                su = cabsf(U[lmx + i * ldu]) / U[lmx + i * ldu];
            }
            s = sv / su;
            for (j = 0; j < n; j++) {
                res1 = fmaxf(res1, cabsf(U[j + i * ldu] - s * V[j + i * ldv]));
            }
        }
        res1 = res1 / ((f32)n * ulp);

        /* Compute orthogonality of columns of V. */

        cunt01("C", n, mv, V, ldv, work, lwork, rwork, &res2);
    }

    f32 maxres = (res1 > res2) ? res1 : res2;
    *result = fminf(maxres, ONE / ulp);
}
