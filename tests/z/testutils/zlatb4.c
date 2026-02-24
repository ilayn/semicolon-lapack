/**
 * @file zlatb4.c
 * @brief ZLATB4 sets parameters for the matrix generator based on the type of
 *        matrix to be generated.
 */

#include <math.h>
#include "verify.h"
#include <string.h>

static INT first = 1;
static f64 eps, small, large, badc1, badc2;

/**
 * ZLATB4 sets parameters for the matrix generator based on the type of
 * matrix to be generated.
 *
 * @param[in]  path    The LAPACK path name (e.g., "ZGE" for general matrices).
 * @param[in]  imat    An integer key describing which matrix to generate.
 * @param[in]  m       The number of rows in the matrix to be generated.
 * @param[in]  n       The number of columns in the matrix to be generated.
 * @param[out] type    The type of matrix: 'S' symmetric, 'H' Hermitian,
 *                     'P' positive definite, 'N' nonsymmetric.
 * @param[out] kl      The lower bandwidth of the matrix.
 * @param[out] ku      The upper bandwidth of the matrix.
 * @param[out] anorm   The desired norm of the matrix.
 * @param[out] mode    A key indicating how to choose eigenvalues.
 * @param[out] cndnum  The desired condition number.
 * @param[out] dist    The type of distribution for the random number generator.
 */
void zlatb4(
    const char* path,
    const INT imat,
    const INT m,
    const INT n,
    char* type,
    INT* kl,
    INT* ku,
    f64* anorm,
    INT* mode,
    f64* cndnum,
    char* dist)
{
    const f64 SHRINK = 0.25;
    const f64 TENTH = 0.1;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    char c2[3];

    if (first) {
        first = 0;
        eps = dlamch("P");
        badc2 = TENTH / eps;
        badc1 = sqrt(badc2);
        small = dlamch("S");
        large = ONE / small;
        small = SHRINK * (small / eps);
        large = ONE / small;
    }

    c2[0] = path[1];
    c2[1] = path[2];
    c2[2] = '\0';

    *dist = 'S';
    *mode = 3;

    if (strcmp(c2, "QR") == 0 || strcmp(c2, "LQ") == 0 ||
        strcmp(c2, "QL") == 0 || strcmp(c2, "RQ") == 0) {

        *type = 'N';

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

        if (imat == 5) {
            *cndnum = badc1;
        } else if (imat == 6) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (imat == 7) {
            *anorm = small;
        } else if (imat == 8) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "QK") == 0) {

        *type = 'N';
        *dist = 'S';

        if (imat == 2) {
            *kl = 0;
            *ku = 0;
            *cndnum = TWO;
            *anorm = ONE;
            *mode = 3;
        } else if (imat == 3) {
            *kl = 0;
            *ku = (n > 1) ? n - 1 : 0;
            *cndnum = TWO;
            *anorm = ONE;
            *mode = 3;
        } else if (imat == 4) {
            *kl = (m > 1) ? m - 1 : 0;
            *ku = 0;
            *cndnum = TWO;
            *anorm = ONE;
            *mode = 3;
        } else {
            *kl = (m > 1) ? m - 1 : 0;
            *ku = (n > 1) ? n - 1 : 0;

            if (imat >= 5 && imat <= 14) {
                *cndnum = TWO;
                *anorm = ONE;
                *mode = 3;
            } else if (imat == 15) {
                *cndnum = badc1;
                *anorm = ONE;
                *mode = 3;
            } else if (imat == 16) {
                *cndnum = badc2;
                *anorm = ONE;
                *mode = 3;
            } else if (imat == 17) {
                *cndnum = badc2;
                *anorm = ONE;
                *mode = 2;
            } else if (imat == 18) {
                *cndnum = TWO;
                *anorm = small;
                *mode = 3;
            } else if (imat == 19) {
                *cndnum = TWO;
                *anorm = large;
                *mode = 3;
            } else {
                *cndnum = TWO;
                *anorm = ONE;
                *mode = 3;
            }
        }

    } else if (strcmp(c2, "GE") == 0) {

        *type = 'N';

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

        if (imat == 8) {
            *cndnum = badc1;
        } else if (imat == 9) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (imat == 10) {
            *anorm = small;
        } else if (imat == 11) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "GB") == 0) {

        *type = 'N';

        if (imat == 5) {
            *cndnum = badc1;
        } else if (imat == 6) {
            *cndnum = TENTH * badc2;
        } else {
            *cndnum = TWO;
        }

        if (imat == 7) {
            *anorm = small;
        } else if (imat == 8) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "GT") == 0) {

        *type = 'N';

        if (imat == 1) {
            *kl = 0;
        } else {
            *kl = 1;
        }
        *ku = *kl;

        if (imat == 3) {
            *cndnum = badc1;
        } else if (imat == 4) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (imat == 5 || imat == 11) {
            *anorm = small;
        } else if (imat == 6 || imat == 12) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "PO") == 0 || strcmp(c2, "PP") == 0) {

        *type = c2[0];

        if (imat == 1) {
            *kl = 0;
        } else {
            *kl = (n > 1) ? n - 1 : 0;
        }
        *ku = *kl;

        if (imat == 6) {
            *cndnum = badc1;
        } else if (imat == 7) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (imat == 8) {
            *anorm = small;
        } else if (imat == 9) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "HE") == 0 || strcmp(c2, "HP") == 0 ||
               strcmp(c2, "SY") == 0 || strcmp(c2, "SP") == 0) {

        *type = c2[0];

        if (imat == 1) {
            *kl = 0;
        } else {
            *kl = (n > 1) ? n - 1 : 0;
        }
        *ku = *kl;

        if (imat == 7) {
            *cndnum = badc1;
        } else if (imat == 8) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (imat == 9) {
            *anorm = small;
        } else if (imat == 10) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "PB") == 0) {

        *type = 'P';

        if (imat == 5) {
            *cndnum = badc1;
        } else if (imat == 6) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (imat == 7) {
            *anorm = small;
        } else if (imat == 8) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "PT") == 0) {

        *type = 'P';
        if (imat == 1) {
            *kl = 0;
        } else {
            *kl = 1;
        }
        *ku = *kl;

        if (imat == 3) {
            *cndnum = badc1;
        } else if (imat == 4) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (imat == 5 || imat == 11) {
            *anorm = small;
        } else if (imat == 6 || imat == 12) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "TR") == 0 || strcmp(c2, "TP") == 0) {

        *type = 'N';

        INT mat = (imat < 0) ? -imat : imat;

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

        if (mat == 3 || mat == 9) {
            *cndnum = badc1;
        } else if (mat == 4 || mat == 10) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (mat == 5) {
            *anorm = small;
        } else if (mat == 6) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else if (strcmp(c2, "TB") == 0) {

        *type = 'N';

        INT mat = (imat < 0) ? -imat : imat;

        if (mat == 2 || mat == 8) {
            *cndnum = badc1;
        } else if (mat == 3 || mat == 9) {
            *cndnum = badc2;
        } else {
            *cndnum = TWO;
        }

        if (mat == 4) {
            *anorm = small;
        } else if (mat == 5) {
            *anorm = large;
        } else {
            *anorm = ONE;
        }

    } else {
        *type = 'N';
        *kl = (m > 1) ? m - 1 : 0;
        *ku = (n > 1) ? n - 1 : 0;
        *cndnum = TWO;
        *anorm = ONE;
    }

    if (n <= 1) {
        *cndnum = ONE;
    }
}
