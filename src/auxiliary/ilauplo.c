/**
 * @file ilauplo.c
 * @brief ILAUPLO translates a character UPLO to the corresponding BLAST integer constant.
 */

#include "semicolon_lapack_auxiliary.h"

/**
 * This subroutine translated from a character string specifying a
 * upper- or lower-triangular matrix to the relevant BLAST-specified
 * integer constant.
 *
 * ILAUPLO returns an INTEGER.  If ILAUPLO < 0, then the input is not
 * a character indicating an upper- or lower-triangular matrix.
 * Otherwise ILAUPLO returns the constant value corresponding to UPLO.
 */
INT ilauplo(const char* uplo)
{
    const INT BLAS_UPPER = 121;
    const INT BLAS_LOWER = 122;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        return BLAS_UPPER;
    } else if (uplo[0] == 'L' || uplo[0] == 'l') {
        return BLAS_LOWER;
    } else {
        return -1;
    }
}
