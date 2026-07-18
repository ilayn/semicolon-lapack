/**
 * @file chla_transtype.c
 * @brief CHLA_TRANSTYPE translates a BLAST integer constant to the character transposition type.
 */

#include "semicolon_lapack_auxiliary.h"

/**
 * This subroutine translates from a BLAST-specified integer constant to
 * the character string specifying a transposition operation.
 *
 * @param[in] trans The BLAST-specified integer constant identifying the
 *                  transposition operation.
 *
 * @return A character. If the returned value is `'X'`, then the input is not an
 *         integer indicating a transposition operator. Otherwise the constant
 *         value corresponding to `trans` is returned:
 *         `'N'` for `trans=111` (BLAS_NO_TRANS),
 *         `'T'` for `trans=112` (BLAS_TRANS),
 *         `'C'` for `trans=113` (BLAS_CONJ_TRANS).
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
