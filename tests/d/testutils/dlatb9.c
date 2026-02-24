/**
 * @file dlatb9.c
 * @brief DLATB9 sets parameters for the matrix generator based on the type of
 *        matrix to be generated.
 */

#include <math.h>
#include "verify.h"

static INT first = 1;
static f64 eps, small, large, badc1, badc2;

/**
 * DLATB9 sets parameters for the matrix generator based on the type of
 * matrix to be generated.
 *
 * @param[in]  path    The LAPACK path name (3 characters: GLM, GQR, GRQ, GSV, LSE).
 * @param[in]  imat    An integer key describing which matrix to generate.
 * @param[in]  m       The number of rows/columns dimension M.
 * @param[in]  p       The number of rows/columns dimension P.
 * @param[in]  n       The number of rows/columns dimension N.
 * @param[out] type    The type of the matrix to be generated.
 * @param[out] kla     The lower band width of matrix A.
 * @param[out] kua     The upper band width of matrix A.
 * @param[out] klb     The lower band width of matrix B.
 * @param[out] kub     The upper band width of matrix B.
 * @param[out] anorm   The desired norm of matrix A.
 * @param[out] bnorm   The desired norm of matrix B.
 * @param[out] modea   A key indicating how to choose the eigenvalues of A.
 * @param[out] modeb   A key indicating how to choose the eigenvalues of B.
 * @param[out] cndnma  The desired condition number of A.
 * @param[out] cndnmb  The desired condition number of B.
 * @param[out] dista   The distribution for matrix A.
 * @param[out] distb   The distribution for matrix B.
 */
void dlatb9(
    const char* path,
    const INT imat,
    const INT m,
    const INT p,
    const INT n,
    char* type,
    INT* kla,
    INT* kua,
    INT* klb,
    INT* kub,
    f64* anorm,
    f64* bnorm,
    INT* modea,
    INT* modeb,
    f64* cndnma,
    f64* cndnmb,
    char* dista,
    char* distb)
{
    const f64 SHRINK = 0.25;
    const f64 TENTH = 0.1;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;

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

    /* Set some parameters we don't plan to change. */
    *type = 'N';
    *dista = 'S';
    *distb = 'S';
    *modea = 3;
    *modeb = 4;

    /* Set the lower and upper bandwidths. */

    if ((path[0] == 'G' && path[1] == 'R' && path[2] == 'Q') ||
        (path[0] == 'L' && path[1] == 'S' && path[2] == 'E') ||
        (path[0] == 'G' && path[1] == 'S' && path[2] == 'V')) {

        /* A: M by N, B: P by N */

        if (imat == 1) {
            /* A: diagonal, B: upper triangular */
            *kla = 0;
            *kua = 0;
            *klb = 0;
            *kub = (n - 1 > 0) ? n - 1 : 0;
        } else if (imat == 2) {
            /* A: upper triangular, B: upper triangular */
            *kla = 0;
            *kua = (n - 1 > 0) ? n - 1 : 0;
            *klb = 0;
            *kub = (n - 1 > 0) ? n - 1 : 0;
        } else if (imat == 3) {
            /* A: lower triangular, B: upper triangular */
            *kla = (m - 1 > 0) ? m - 1 : 0;
            *kua = 0;
            *klb = 0;
            *kub = (n - 1 > 0) ? n - 1 : 0;
        } else {
            /* A: general dense, B: general dense */
            *kla = (m - 1 > 0) ? m - 1 : 0;
            *kua = (n - 1 > 0) ? n - 1 : 0;
            *klb = (p - 1 > 0) ? p - 1 : 0;
            *kub = (n - 1 > 0) ? n - 1 : 0;
        }

    } else if ((path[0] == 'G' && path[1] == 'Q' && path[2] == 'R') ||
               (path[0] == 'G' && path[1] == 'L' && path[2] == 'M')) {

        /* A: N by M, B: N by P */

        if (imat == 1) {
            /* A: diagonal, B: lower triangular */
            *kla = 0;
            *kua = 0;
            *klb = (n - 1 > 0) ? n - 1 : 0;
            *kub = 0;
        } else if (imat == 2) {
            /* A: lower triangular, B: diagonal */
            *kla = (n - 1 > 0) ? n - 1 : 0;
            *kua = 0;
            *klb = 0;
            *kub = 0;
        } else if (imat == 3) {
            /* A: lower triangular, B: upper triangular */
            *kla = (n - 1 > 0) ? n - 1 : 0;
            *kua = 0;
            *klb = 0;
            *kub = (p - 1 > 0) ? p - 1 : 0;
        } else {
            /* A: general dense, B: general dense */
            *kla = (n - 1 > 0) ? n - 1 : 0;
            *kua = (m - 1 > 0) ? m - 1 : 0;
            *klb = (n - 1 > 0) ? n - 1 : 0;
            *kub = (p - 1 > 0) ? p - 1 : 0;
        }
    }

    /* Set the condition number and norm. */

    *cndnma = TEN * TEN;
    *cndnmb = TEN;
    if ((path[0] == 'G' && path[1] == 'Q' && path[2] == 'R') ||
        (path[0] == 'G' && path[1] == 'R' && path[2] == 'Q') ||
        (path[0] == 'G' && path[1] == 'S' && path[2] == 'V')) {
        if (imat == 5) {
            *cndnma = badc1;
            *cndnmb = badc1;
        } else if (imat == 6) {
            *cndnma = badc2;
            *cndnmb = badc2;
        } else if (imat == 7) {
            *cndnma = badc1;
            *cndnmb = badc2;
        } else if (imat == 8) {
            *cndnma = badc2;
            *cndnmb = badc1;
        }
    }

    *anorm = TEN;
    *bnorm = TEN * TEN * TEN;
    if ((path[0] == 'G' && path[1] == 'Q' && path[2] == 'R') ||
        (path[0] == 'G' && path[1] == 'R' && path[2] == 'Q')) {
        if (imat == 7) {
            *anorm = small;
            *bnorm = large;
        } else if (imat == 8) {
            *anorm = large;
            *bnorm = small;
        }
    }

    if (n <= 1) {
        *cndnma = ONE;
        *cndnmb = ONE;
    }
}
