/**
 * @file slarmm.c
 * @brief SLARMM returns a scale factor to prevent overflow in matrix updates.
 */

#include "semicolon_lapack_single.h"

/**
 * SLARMM returns a factor s in (0, 1] such that the linear updates
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
float slarmm(const float anorm, const float bnorm, const float cnorm)
{
    const float ONE = 1.0f;
    const float HALF = 0.5f;
    const float FOUR = 4.0f;

    float smlnum, bignum;

    /* Determine machine dependent parameters to control overflow */
    smlnum = slamch("S") / slamch("P");
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
