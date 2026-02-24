/**
 * @file slatb4.c
 * @brief SLATB4 sets parameters for the matrix generator based on the type of
 *        matrix to be generated.
 */

#include <math.h>
#include "verify.h"
#include <string.h>

// Static variables to cache machine-dependent values
static INT first = 1;
static f32 eps, small, large, badc1, badc2;

/**
 * SLATB4 sets parameters for the matrix generator based on the type of
 * matrix to be generated.
 *
 * @param[in]  path    The LAPACK path name (e.g., "SGE" for general matrices).
 * @param[in]  imat    An integer key describing which matrix to generate.
 * @param[in]  m       The number of rows in the matrix to be generated.
 * @param[in]  n       The number of columns in the matrix to be generated.
 * @param[out] type    The type of matrix: 'S' symmetric, 'P' positive definite,
 *                     'N' nonsymmetric.
 * @param[out] kl      The lower bandwidth of the matrix.
 * @param[out] ku      The upper bandwidth of the matrix.
 * @param[out] anorm   The desired norm of the matrix.
 * @param[out] mode    A key indicating how to choose eigenvalues.
 * @param[out] cndnum  The desired condition number.
 * @param[out] dist    The type of distribution for the random number generator.
 */
void slatb4(
    const char *path,
    const INT imat,
    const INT m,
    const INT n,
    char *type,
    INT* kl,
    INT* ku,
    f32 *anorm,
    INT* mode,
    f32 *cndnum,
    char *dist)
{
    const f32 SHRINK = 0.25f;
    const f32 TENTH = 0.1f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    char c2[3];

    // Initialize constants on first call
    if (first) {
        first = 0;
        eps = slamch("P");  // Precision = eps * base
        badc2 = TENTH / eps;
        badc1 = sqrtf(badc2);
        small = slamch("S");  // Safe minimum
        large = ONE / small;
        small = SHRINK * (small / eps);
        large = ONE / small;
    }

    // Extract the 2-character matrix type from path (characters 2-3, 0-indexed 1-2)
    c2[0] = path[1];
    c2[1] = path[2];
    c2[2] = '\0';

    // Set default distribution
    *dist = 'S';  // Symmetric distribution
    *mode = 3;    // Default mode

    if (strcmp(c2, "GE") == 0) {
        // xGE: General M x N matrix
        *type = 'N';  // Nonsymmetric

        // Set bandwidths based on matrix type
        if (imat == 1) {
            // Diagonal
            *kl = 0;
            *ku = 0;
        } else if (imat == 2) {
            // Upper triangular
            *kl = 0;
            *ku = (n > 1) ? n - 1 : 0;
        } else if (imat == 3) {
            // Lower triangular
            *kl = (m > 1) ? m - 1 : 0;
            *ku = 0;
        } else {
            // Full matrix
            *kl = (m > 1) ? m - 1 : 0;
            *ku = (n > 1) ? n - 1 : 0;
        }

        // Set condition number
        if (imat == 8) {
            *cndnum = badc1;  // sqrt(0.1/eps) ≈ 3e7
        } else if (imat == 9) {
            *cndnum = badc2;  // 0.1/eps ≈ 9e15
        } else {
            *cndnum = TWO;    // Well-conditioned
        }

        // Set norm
        if (imat == 10) {
            *anorm = small;   // Near underflow
        } else if (imat == 11) {
            *anorm = large;   // Near overflow
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "GB") == 0) {
        // xGB: General banded matrix
        *type = 'N';

        // Condition number
        if (imat == 5) {
            *cndnum = badc1;
        } else if (imat == 6) {
            *cndnum = TENTH * badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (imat == 7) {
            *anorm = small;
        } else if (imat == 8) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "GT") == 0) {
        // xGT: General tridiagonal
        *type = 'N';

        // Bandwidths
        if (imat == 1) {
            *kl = 0;
        } else {
            *kl = 1;
        }
        *ku = *kl;

        // Condition number
        if (imat == 3) {
            *cndnum = badc1;
        } else if (imat == 4) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (imat == 5 || imat == 11) {
            *anorm = small;
        } else if (imat == 6 || imat == 12) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "PO") == 0 || strcmp(c2, "PP") == 0) {
        // xPO, xPP: Symmetric positive definite
        *type = c2[0];

        // Bandwidths
        if (imat == 1) {
            *kl = 0;
        } else {
            *kl = (n > 1) ? n - 1 : 0;
        }
        *ku = *kl;

        // Condition number
        if (imat == 6) {
            *cndnum = badc1;
        } else if (imat == 7) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (imat == 8) {
            *anorm = small;
        } else if (imat == 9) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "PT") == 0) {
        // xPT: Symmetric positive definite tridiagonal
        *type = 'P';

        // Bandwidths
        if (imat == 1) {
            *kl = 0;
        } else {
            *kl = 1;
        }
        *ku = *kl;

        // Condition number
        if (imat == 3) {
            *cndnum = badc1;
        } else if (imat == 4) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (imat == 5 || imat == 11) {
            *anorm = small;
        } else if (imat == 6 || imat == 12) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "PB") == 0) {
        // xPB: Symmetric positive definite band matrix
        *type = 'P';

        // Condition number
        if (imat == 5) {
            *cndnum = badc1;
        } else if (imat == 6) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (imat == 7) {
            *anorm = small;
        } else if (imat == 8) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "SY") == 0 || strcmp(c2, "SP") == 0) {
        // xSY, xSP: Symmetric
        *type = c2[0];

        // Bandwidths
        if (imat == 1) {
            *kl = 0;
        } else {
            *kl = (n > 1) ? n - 1 : 0;
        }
        *ku = *kl;

        // Condition number
        if (imat == 7) {
            *cndnum = badc1;
        } else if (imat == 8) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (imat == 9) {
            *anorm = small;
        } else if (imat == 10) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "QR") == 0 || strcmp(c2, "LQ") == 0 ||
               strcmp(c2, "QL") == 0 || strcmp(c2, "RQ") == 0) {
        // xQR, xLQ, xQL, xRQ: General M x N matrix for orthogonal factorizations
        *type = 'N';

        // Bandwidths
        if (imat == 1) {
            *kl = 0;
            *ku = 0;
        } else if (imat == 2) {
            *kl = 0;
            *ku = (n > 1) ? n - 1 : 0;
        } else if (imat == 3) {
            *kl = (m > 1) ? m - 1 : 0;
            *ku = 0;
        } else {
            *kl = (m > 1) ? m - 1 : 0;
            *ku = (n > 1) ? n - 1 : 0;
        }

        // Condition number
        if (imat == 5) {
            *cndnum = badc1;
        } else if (imat == 6) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (imat == 7) {
            *anorm = small;
        } else if (imat == 8) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "QK") == 0) {
        // xQK: Truncated QR with pivoting (SGEQP3RK)
        // General M x N matrix with 19 matrix types
        *type = 'N';
        *dist = 'S';

        // Bandwidths
        if (imat == 2) {
            // Diagonal
            *kl = 0;
            *ku = 0;
            *cndnum = TWO;
            *anorm = ONE;
            *mode = 3;
        } else if (imat == 3) {
            // Upper triangular
            *kl = 0;
            *ku = (n > 1) ? n - 1 : 0;
            *cndnum = TWO;
            *anorm = ONE;
            *mode = 3;
        } else if (imat == 4) {
            // Lower triangular
            *kl = (m > 1) ? m - 1 : 0;
            *ku = 0;
            *cndnum = TWO;
            *anorm = ONE;
            *mode = 3;
        } else {
            // 5-19: Rectangular matrix
            *kl = (m > 1) ? m - 1 : 0;
            *ku = (n > 1) ? n - 1 : 0;

            if (imat >= 5 && imat <= 14) {
                // 5-14: Random, CNDNUM = 2
                *cndnum = TWO;
                *anorm = ONE;
                *mode = 3;
            } else if (imat == 15) {
                // 15: Random, CNDNUM = sqrt(0.1/EPS)
                *cndnum = badc1;
                *anorm = ONE;
                *mode = 3;
            } else if (imat == 16) {
                // 16: Random, CNDNUM = 0.1/EPS
                *cndnum = badc2;
                *anorm = ONE;
                *mode = 3;
            } else if (imat == 17) {
                // 17: Random, CNDNUM = 0.1/EPS, one small singular value
                *cndnum = badc2;
                *anorm = ONE;
                *mode = 2;
            } else if (imat == 18) {
                // 18: Random, scaled near underflow
                *cndnum = TWO;
                *anorm = small;
                *mode = 3;
            } else if (imat == 19) {
                // 19: Random, scaled near overflow
                *cndnum = TWO;
                *anorm = large;
                *mode = 3;
            } else {
                // Default for other imat values (including 1 which is zero matrix)
                *cndnum = TWO;
                *anorm = ONE;
                *mode = 3;
            }
        }

    } else if (strcmp(c2, "TB") == 0) {
        // xTB: Triangular banded matrix
        *type = 'N';

        INT mat = (imat < 0) ? -imat : imat;

        // Condition number
        if (mat == 2 || mat == 8) {
            *cndnum = badc1;
        } else if (mat == 3 || mat == 9) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (mat == 4) {
            *anorm = small;
        } else if (mat == 5) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "TR") == 0 || strcmp(c2, "TP") == 0) {
        // xTR, xTP: Triangular
        *type = 'N';

        INT mat = (imat < 0) ? -imat : imat;

        // Bandwidths
        if (mat == 1 || mat == 7) {
            *kl = 0;
            *ku = 0;
        } else if (imat < 0) {
            *kl = (n > 1) ? n - 1 : 0;
            *ku = 0;
        } else {
            *kl = 0;
            *ku = (n > 1) ? n - 1 : 0;
        }

        // Condition number
        if (mat == 3 || mat == 9) {
            *cndnum = badc1;
        } else if (mat == 4 || mat == 10) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        // Norm
        if (mat == 5) {
            *anorm = small;
        } else if (mat == 6) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else {
        // Default: General matrix with default parameters
        *type = 'N';
        *kl = (m > 1) ? m - 1 : 0;
        *ku = (n > 1) ? n - 1 : 0;
        *cndnum = TWO;
        *anorm = ONE;
    }

    // For small matrices, use condition number of 1
    if (n <= 1) {
        *cndnum = ONE;
    }
}
