/**
 * @file sort03.c
 * @brief SORT03 compares two orthogonal matrices U and V to see if their
 *        corresponding rows or columns span the same spaces.
 *
 * Port of LAPACK's TESTING/EIG/sort03.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * SORT03 compares two orthogonal matrices U and V to see if their
 * corresponding rows or columns span the same spaces. The rows are
 * checked if RC = 'R', and the columns are checked if RC = 'C'.
 *
 * RESULT is the maximum of
 *
 *    | V*V' - I | / ( MV ulp ), if RC = 'R', or
 *    | V'*V - I | / ( MV ulp ), if RC = 'C',
 *
 * and the maximum over rows (or columns) 1 to K of
 *
 *    | U(i) - S*V(i) |/ ( N ulp )
 *
 * where S is +-1 (chosen to minimize the expression), U(i) is the i-th
 * row (column) of U, and V(i) is the i-th row (column) of V.
 *
 * @param[in]     rc     If rc = 'R' the rows of U and V are to be compared.
 *                       If rc = 'C' the columns of U and V are to be compared.
 * @param[in]     mu     The number of rows of U if rc = 'R', and the number of
 *                       columns if rc = 'C'. If mu = 0 SORT03 does nothing.
 * @param[in]     mv     The number of rows of V if rc = 'R', and the number of
 *                       columns if rc = 'C'. If mv = 0 SORT03 does nothing.
 * @param[in]     n      If rc = 'R', the number of columns in the matrices U and V,
 *                       and if rc = 'C', the number of rows in U and V.
 * @param[in]     k      The number of rows or columns of U and V to compare.
 * @param[in]     U      The first matrix to compare, dimension (ldu, *).
 * @param[in]     ldu    Leading dimension of U.
 * @param[in]     V      The second matrix to compare, dimension (ldv, *).
 * @param[in]     ldv    Leading dimension of V.
 * @param[out]    work   Workspace array, dimension (lwork).
 * @param[in]     lwork  Length of the array work.
 * @param[out]    result The computed test ratio.
 * @param[out]    info   0 indicates successful exit, < 0 indicates illegal value.
 */
void sort03(const char* rc, const INT mu, const INT mv, const INT n,
            const INT k, const f32* const restrict U, const INT ldu,
            const f32* const restrict V, const INT ldv,
            f32* const restrict work, const INT lwork,
            f32* result, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT irc;
    INT i, j, lmx;
    f32 res1, res2, s, ulp;

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
            /* Find index of max element in row i of U */
            lmx = cblas_isamax(n, &U[i], ldu);
            /* Determine sign */
            s = (U[i + lmx * ldu] >= ZERO ? ONE : -ONE) *
                (V[i + lmx * ldv] >= ZERO ? ONE : -ONE);

            /* Compute max | U(i,j) - s*V(i,j) | */
            for (j = 0; j < n; j++) {
                f32 diff = fabsf(U[i + j * ldu] - s * V[i + j * ldv]);
                if (diff > res1) {
                    res1 = diff;
                }
            }
        }
        res1 = res1 / ((f32)n * ulp);

        /* Compute orthogonality of rows of V. */
        sort01("R", mv, n, V, ldv, work, lwork, &res2);

    } else {
        /* Compare columns */
        res1 = ZERO;
        for (i = 0; i < k; i++) {
            /* Find index of max element in column i of U */
            lmx = cblas_isamax(n, &U[i * ldu], 1);
            /* Determine sign */
            s = (U[lmx + i * ldu] >= ZERO ? ONE : -ONE) *
                (V[lmx + i * ldv] >= ZERO ? ONE : -ONE);

            /* Compute max | U(j,i) - s*V(j,i) | */
            for (j = 0; j < n; j++) {
                f32 diff = fabsf(U[j + i * ldu] - s * V[j + i * ldv]);
                if (diff > res1) {
                    res1 = diff;
                }
            }
        }
        res1 = res1 / ((f32)n * ulp);

        /* Compute orthogonality of columns of V. */
        sort01("C", n, mv, V, ldv, work, lwork, &res2);
    }

    f32 maxres = (res1 > res2) ? res1 : res2;
    *result = fminf(maxres, ONE / ulp);
}
