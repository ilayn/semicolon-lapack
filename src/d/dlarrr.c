/**
 * @file dlarrr.c
 * @brief DLARRR performs tests to decide whether the symmetric tridiagonal
 *        matrix T warrants expensive computations which guarantee high
 *        relative accuracy in the eigenvalues.
 */

#include "internal_build_defs.h"
#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * Perform tests to decide whether the symmetric tridiagonal matrix T
 * warrants expensive computations which guarantee high relative accuracy
 * in the eigenvalues.
 *
 * @param[in]     n     The order of the matrix. n > 0.
 * @param[in]     D     Double precision array, dimension (n).
 *                      The n diagonal elements of the tridiagonal matrix T.
 * @param[in,out] E     Double precision array, dimension (n).
 *                      On entry, the first (n-1) entries contain the subdiagonal
 *                      elements of the tridiagonal matrix T; E(n-1) is set to zero.
 * @param[out]    info
 *                         - = 0: the matrix warrants computations preserving
 *                           relative accuracy.
 *                         - = 1: the matrix warrants computations guaranteeing
 *                           only absolute accuracy.
 */
void dlarrr(const INT n, const f64* D, f64* E, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 RELCOND = 0.999;

    /* Quick return if possible */
    if (n <= 0) {
        *info = 0;
        return;
    }

    /* As a default, do NOT go for relative-accuracy preserving computations. */
    *info = 1;

    f64 safmin = dlamch("S");
    f64 eps = dlamch("P");
    f64 smlnum = safmin / eps;
    f64 rmin = sqrt(smlnum);

    /* Tests for relative accuracy */

    /* Test for scaled diagonal dominance
       Scale the diagonal entries to one and check whether the sum of the
       off-diagonals is less than one

       The sdd relative error bounds have a 1/(1- 2*x) factor in them,
       x = max(OFFDIG + OFFDIG2), so when x is close to 1/2, no relative
       accuracy is promised.  In the notation of the code fragment below,
       1/(1 - (OFFDIG + OFFDIG2)) is the condition number.
       We don't think it is worth going into "sdd mode" unless the relative
       condition number is reasonable, not 1/macheps.
       The threshold should be compatible with other thresholds used in the
       code. We set  OFFDIG + OFFDIG2 <= .999 =: RELCOND, it corresponds
       to losing at most 3 decimal digits: 1 / (1 - (OFFDIG + OFFDIG2)) <= 1000
       instead of the current OFFDIG + OFFDIG2 < 1 */

    INT yesrel = 1;
    f64 offdig = ZERO;
    f64 offdig2;
    f64 tmp = sqrt(fabs(D[0]));
    if (tmp < rmin) yesrel = 0;
    if (yesrel) {
        for (INT i = 1; i < n; i++) {
            f64 tmp2 = sqrt(fabs(D[i]));
            if (tmp2 < rmin) { yesrel = 0; break; }
            offdig2 = fabs(E[i - 1]) / (tmp * tmp2);
            if (offdig + offdig2 >= RELCOND) { yesrel = 0; break; }
            tmp = tmp2;
            offdig = offdig2;
        }
    }

    if (yesrel) {
        *info = 0;
        return;
    }

    /* *** MORE TO BE IMPLEMENTED *** */

    /* Test if the lower bidiagonal matrix L from T = L D L^T
       (zero shift facto) is well conditioned */

    /* Test if the upper bidiagonal matrix U from T = U D U^T
       (zero shift facto) is well conditioned.
       In this case, the matrix needs to be flipped and, at the end
       of the eigenvector computation, the flip needs to be applied
       to the computed eigenvectors (and the support) */
}
