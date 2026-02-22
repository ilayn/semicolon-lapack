/**
 * @file ilaenv2stage.c
 * @brief ILAENV2STAGE chooses problem-dependent parameters for 2-stage algorithms.
 */

#include "semicolon_lapack_auxiliary.h"

INT ilaenv2stage(const INT ispec, const char* name, const char* opts,
                 const INT n1, const INT n2, const INT n3, const INT n4)
{
    INT iispec;

    if (ispec < 1 || ispec > 5) {
        return -1;
    }

    iispec = 16 + ispec;
    return iparam2stage(iispec, name, opts, n1, n2, n3, n4);
}
