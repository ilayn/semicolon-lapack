/**
 * @file chla_transtype.c
 * @brief CHLA_TRANSTYPE translates a BLAST integer constant to the character transposition type.
 */

#include "semicolon_lapack_auxiliary.h"

/**
 * This subroutine translates from a BLAST-specified integer constant to
 * the character string specifying a transposition operation.
 *
 * CHLA_TRANSTYPE returns an CHARACTER*1.  If CHLA_TRANSTYPE is 'X',
 * then input is not an integer indicating a transposition operator.
 * Otherwise CHLA_TRANSTYPE returns the constant value corresponding to
 * TRANS.
 */
char chla_transtype(const INT trans)
{
    const INT BLAS_NO_TRANS = 111;
    const INT BLAS_TRANS = 112;
    const INT BLAS_CONJ_TRANS = 113;

    if (trans == BLAS_NO_TRANS) {
        return 'N';
    } else if (trans == BLAS_TRANS) {
        return 'T';
    } else if (trans == BLAS_CONJ_TRANS) {
        return 'C';
    } else {
        return 'X';
    }
}
