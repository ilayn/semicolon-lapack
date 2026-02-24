/**
 * @file dlatb5.c
 * @brief DLATB5 sets parameters for the matrix generator based on the type of
 *        matrix to be generated.
 */

#include <math.h>
#include "verify.h"

static INT first = 1;
static f64 eps, small, large, badc1, badc2;

/**
 * DLATB5 sets parameters for the matrix generator based on the type of
 * matrix to be generated.
 *
 * @param[in]  path    The LAPACK path name.
 * @param[in]  imat    An integer key describing which matrix to generate for
 *                     this path.
 * @param[in]  n       The number of rows and columns in the matrix to be
 *                     generated.
 * @param[out] type    The type of the matrix to be generated:
 *                     = 'S':  symmetric matrix
 *                     = 'P':  symmetric positive (semi)definite matrix
 *                     = 'N':  nonsymmetric matrix
 * @param[out] kl      The lower band width of the matrix to be generated.
 * @param[out] ku      The upper band width of the matrix to be generated.
 * @param[out] anorm   The desired norm of the matrix to be generated. The
 *                     diagonal matrix of singular values or eigenvalues is
 *                     scaled by this value.
 * @param[out] mode    A key indicating how to choose the vector of eigenvalues.
 * @param[out] cndnum  The desired condition number.
 * @param[out] dist    The type of distribution to be used by the random number
 *                     generator.
 */
void dlatb5(
    const char* path,
    const INT imat,
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

    *dist = 'S';
    *mode = 3;

    *type = path[1];

    if (imat == 1) {
        *kl = 0;
    } else {
        *kl = (n > 1) ? n - 1 : 0;
    }
    *ku = *kl;

    if (imat == 3) {
        *cndnum = 1.0e12;
        *mode = 2;
    } else if (imat == 4) {
        *cndnum = 1.0e12;
        *mode = 1;
    } else if (imat == 5) {
        *cndnum = 1.0e12;
        *mode = 3;
    } else if (imat == 6) {
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

    if (n <= 1) {
        *cndnum = ONE;
    }
}
