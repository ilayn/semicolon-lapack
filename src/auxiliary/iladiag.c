/**
 * @file iladiag.c
 * @brief ILADIAG translates a character DIAG to the corresponding BLAST integer constant.
 */

#include "semicolon_lapack_auxiliary.h"

/**
 * This subroutine translated from a character string specifying if a
 * matrix has unit diagonal or not to the relevant BLAST-specified
 * integer constant.
 *
 * @param[in] diag `diag='N'`: the matrix has a non-unit diagonal.
 *                 `diag='U'`: the matrix has a unit diagonal.
 *
 * @return An integer. If the returned value `<0`, then the input is not a
 *         character indicating a unit or non-unit diagonal. Otherwise the
 *         constant value corresponding to `diag` is returned:
 *         `131` (BLAS_NON_UNIT_DIAG) for `diag='N'`,
 *         `132` (BLAS_UNIT_DIAG) for `diag='U'`.
 */
INT iladiag(const char* diag)
{
    const INT BLAS_NON_UNIT_DIAG = 131;
    const INT BLAS_UNIT_DIAG = 132;

    if (diag[0] == 'N' || diag[0] == 'n') {
        return BLAS_NON_UNIT_DIAG;
    } else if (diag[0] == 'U' || diag[0] == 'u') {
        return BLAS_UNIT_DIAG;
    } else {
        return -1;
    }
}
