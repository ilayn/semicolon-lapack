/**
 * @file slatb5.c
 * @brief SLATB5 sets parameters for the matrix generator based on the type of
 *        matrix to be generated.
 */

#include <math.h>
#include "verify.h"

extern f32 slamch(const char* cmach);

static int first = 1;
static f32 eps, small, large, badc1, badc2;

/**
 * SLATB5 sets parameters for the matrix generator based on the type of
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
void slatb5(
    const char* path,
    const int imat,
    const int n,
    char* type,
    int* kl,
    int* ku,
    f32* anorm,
    int* mode,
    f32* cndnum,
    char* dist)
{
    const f32 SHRINK = 0.25f;
    const f32 TENTH = 0.1f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    if (first) {
        first = 0;
        eps = slamch("P");
        badc2 = TENTH / eps;
        badc1 = sqrtf(badc2);
        small = slamch("S");
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
        *cndnum = 1.0e4f;
        *mode = 2;
    } else if (imat == 4) {
        *cndnum = 1.0e4f;
        *mode = 1;
    } else if (imat == 5) {
        *cndnum = 1.0e4f;
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
