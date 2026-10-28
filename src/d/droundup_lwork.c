/**
 * @file droundup_lwork.c
 * @brief DROUNDUP_LWORK deals with a subtle bug with returning LWORK as a Float.
 */

#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DROUNDUP_LWORK deals with a subtle bug with returning LWORK as a Float.
 * This routine guarantees it is rounded up instead of down by
 * multiplying LWORK by 1+eps when it is necessary, where eps is the relative
 * machine precision.
 * E.g.,
 * @rst
 * .. code-block:: text
 *
 *        float( 9007199254740993            ) == 9007199254740992
 *        float( 9007199254740993 ) * (1.+eps) == 9007199254740994
 * @endrst
 *
 * This routine was inspired in the method `magma_zmake_lwork` from MAGMA.
 * See https://bitbucket.org/icl/magma/src/master/control/magma_zauxiliary.cpp
 *
 * @param[in] lwork Workspace size.
 *
 * @return `droundup_lwork >= lwork`.
 *         `droundup_lwork` is guaranteed to have zero decimal part.
 */
f64 droundup_lwork(const INT lwork)
{
    f64 roundup = (f64)lwork;

    if ((INT)roundup < lwork) {
        /* Force round up of LWORK */
        roundup = roundup * (1.0 + DBL_EPSILON);
    }

    return roundup;
}
