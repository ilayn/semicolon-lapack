/**
 * @file dlarmm.c
 * @brief DLARMM returns a scale factor to prevent overflow in matrix updates.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"

/**
 * DLARMM returns a factor s in (0, 1] such that the linear updates
 *
 *    (s * C) - A * (s * B)  and  (s * C) - (s * A) * B
 *
 * cannot overflow, where A, B, and C are matrices of conforming
 * dimensions.
 *
 * This is an auxiliary routine so there is no argument checking.
 *
 * Reference:
 *   C. C. Kjelgaard Mikkelsen and L. Karlsson, Blocked Algorithms for
 *   Robust Solution of Triangular Linear Systems. In: International
 *   Conference on Parallel Processing and Applied Mathematics, pages
 *   68--78. Springer, 2017.
 *
 * @param[in] anorm   The infinity norm of A. anorm >= 0.
 * @param[in] bnorm   The infinity norm of B. bnorm >= 0.
 * @param[in] cnorm   The infinity norm of C. cnorm >= 0.
 *
 * @return Scale factor s in (0, 1].
 */
f64 dlarmm(const f64 anorm, const f64 bnorm, const f64 cnorm)
{
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;
    const f64 FOUR = 4.0;

    f64 smlnum, bignum;

    /* Determine machine dependent parameters to control overflow */
    smlnum = dlamch("S") / dlamch("P");
    bignum = (ONE / smlnum) / FOUR;

    /* Compute a scale factor */
    if (bnorm <= ONE) {
        if (anorm * bnorm > bignum - cnorm) {
            return HALF;
        }
    } else {
        if (anorm > (bignum - cnorm) / bnorm) {
            return HALF / bnorm;
        }
    }

    return ONE;
}
