/**
 * @file ilaenv2stage.c
 * @brief ILAENV2STAGE chooses problem-dependent parameters for 2-stage algorithms.
 */

#include "semicolon_lapack_auxiliary.h"

int ilaenv2stage(const int ispec, const char* name, const char* opts,
                 const int n1, const int n2, const int n3, const int n4)
{
    int iispec;

    if (ispec < 1 || ispec > 5) {
        return -1;
    }

    iispec = 16 + ispec;
    return iparam2stage(iispec, name, opts, n1, n2, n3, n4);
}
