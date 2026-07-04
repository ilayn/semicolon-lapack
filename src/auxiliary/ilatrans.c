/**
 * @file ilatrans.c
 * @brief ILATRANS translates a character TRANS to the corresponding BLAST integer constant.
 */

#include "semicolon_lapack_auxiliary.h"

/**
 * This subroutine translates from a character string specifying a
 * transposition operation to the relevant BLAST-specified integer
 * constant.
 *
 * ILATRANS returns an INTEGER.  If ILATRANS < 0, then the input is not
 * a character indicating a transposition operator.  Otherwise ILATRANS
 * returns the constant value corresponding to TRANS.
 */
INT ilatrans(const char* trans)
{
    const INT BLAS_NO_TRANS = 111;
    const INT BLAS_TRANS = 112;
    const INT BLAS_CONJ_TRANS = 113;

    if (trans[0] == 'N' || trans[0] == 'n') {
        return BLAS_NO_TRANS;
    } else if (trans[0] == 'T' || trans[0] == 't') {
        return BLAS_TRANS;
    } else if (trans[0] == 'C' || trans[0] == 'c') {
        return BLAS_CONJ_TRANS;
    } else {
        return -1;
    }
}
