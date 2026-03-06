/**
 * @file clctsx.c
 * @brief CLCTSX selects eigenvalues for generalized Schur form reordering tests.
 *
 * Stateful selection function used by ZDRGSX test driver.
 * Port of LAPACK TESTING/EIG/clctsx.f
 */

#include "verify.h"

static INT zlctsx_m = 0;
static INT zlctsx_n = 0;
static INT zlctsx_mplusn = 0;
static INT zlctsx_i = 0;
static INT zlctsx_fs = 1;

void clctsx_reset(INT m, INT n, INT mplusn)
{
    zlctsx_m = m;
    zlctsx_n = n;
    zlctsx_mplusn = mplusn;
    zlctsx_i = 0;
    zlctsx_fs = 1;
}

INT clctsx(const c64* alpha, const c64* beta)
{
    (void)alpha;
    (void)beta;

    INT result;

    if (zlctsx_fs) {
        zlctsx_i = zlctsx_i + 1;
        if (zlctsx_i <= zlctsx_m) {
            result = 0;
        } else {
            result = 1;
        }
        if (zlctsx_i == zlctsx_mplusn) {
            zlctsx_fs = 0;
            zlctsx_i = 0;
        }
    } else {
        zlctsx_i = zlctsx_i + 1;
        if (zlctsx_i <= zlctsx_n) {
            result = 1;
        } else {
            result = 0;
        }
        if (zlctsx_i == zlctsx_mplusn) {
            zlctsx_fs = 1;
            zlctsx_i = 0;
        }
    }

    return result;
}
