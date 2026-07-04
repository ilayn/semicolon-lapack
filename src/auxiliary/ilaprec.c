/**
 * @file ilaprec.c
 * @brief ILAPREC translates a character PREC to the corresponding BLAST integer constant.
 */

#include "semicolon_lapack_auxiliary.h"

/**
 * This subroutine translated from a character string specifying an
 * intermediate precision to the relevant BLAST-specified integer
 * constant.
 *
 * ILAPREC returns an INTEGER.  If ILAPREC < 0, then the input is not a
 * character indicating a supported intermediate precision.  Otherwise
 * ILAPREC returns the constant value corresponding to PREC.
 */
INT ilaprec(const char* prec)
{
    const INT BLAS_PREC_SINGLE = 211;
    const INT BLAS_PREC_DOUBLE = 212;
    const INT BLAS_PREC_INDIGENOUS = 213;
    const INT BLAS_PREC_EXTRA = 214;

    if (prec[0] == 'S' || prec[0] == 's') {
        return BLAS_PREC_SINGLE;
    } else if (prec[0] == 'D' || prec[0] == 'd') {
        return BLAS_PREC_DOUBLE;
    } else if (prec[0] == 'I' || prec[0] == 'i') {
        return BLAS_PREC_INDIGENOUS;
    } else if (prec[0] == 'X' || prec[0] == 'x' || prec[0] == 'E' || prec[0] == 'e') {
        return BLAS_PREC_EXTRA;
    } else {
        return -1;
    }
}
