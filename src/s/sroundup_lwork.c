/**
 * @file sroundup_lwork.c
 * @brief SROUNDUP_LWORK deals with a subtle bug with returning LWORK as a Float.
 */

#include <float.h>
#include "semicolon_lapack_single.h"

/**
 * SROUNDUP_LWORK deals with a subtle bug with returning LWORK as a Float.
 * This routine guarantees it is rounded up instead of down by
 * multiplying LWORK by 1+eps when it is necessary, where eps is the relative
 * machine precision.
 * E.g.,
 * @rst
 * .. code-block:: text
 *
 *        float( 16777217            ) == 16777216
 *        float( 16777217 ) * (1.+eps) == 16777218
 * @endrst
 *
 * This routine was inspired in the method `magma_zmake_lwork` from MAGMA.
 * See https://bitbucket.org/icl/magma/src/master/control/magma_zauxiliary.cpp
 *
 * @param[in] lwork Workspace size.
 *
 * @return `sroundup_lwork >= lwork`.
 *         `sroundup_lwork` is guaranteed to have zero decimal part.
 */
f32 sroundup_lwork(const INT lwork)
{
    f32 roundup = (f32)lwork;

    if ((INT)roundup < lwork) {
        /* Force round up of LWORK */
        roundup = roundup * (1.0f + FLT_EPSILON);
    }

    return roundup;
}
