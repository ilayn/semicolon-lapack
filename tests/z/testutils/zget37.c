/**
 * @file zget37.c
 * @brief ZGET37 tests ZTRSNA, a routine for estimating condition numbers of
 *        eigenvalues and/or right eigenvectors of a matrix.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define LDT   20
#define LWORK (2 * LDT * (10 + LDT))

#define NCASES37 22

/*
 * Embedded test data from LAPACK zec.in (zget37 section, lines 177-368).
 * 22 test cases. Format per case:
 *   n*n complex matrix values stored as (re,im) pairs in row-major order,
 *   then n*(wrin,wiin,sin,sepin) quadruplets as 4 doubles.
 */
static const f64 zget37_data[] = {
    /* Case 0: N=1 */
    0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,

    /* Case 1: N=1 */
    0.0, 1.0,
    0.0, 1.0, 1.0, 1.0,

    /* Case 2: N=2 */
    0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0,

    /* Case 3: N=2 */
    3.0, 0.0,  2.0, 0.0,
    2.0, 0.0,  3.0, 0.0,
    1.0, 0.0, 1.0, 4.0,
    5.0, 0.0, 1.0, 4.0,

    /* Case 4: N=2 */
    3.0, 0.0,  0.0, 2.0,
    0.0, 2.0,  3.0, 0.0,
    3.0, 2.0, 1.0, 4.0,
    3.0, -2.0, 1.0, 4.0,

    /* Case 5: N=5 */
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0,

    /* Case 6: N=5 */
    1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  1.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  1.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  1.0, 0.0,
    1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0,

    /* Case 7: N=5 */
    1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  2.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  3.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  4.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  5.0, 0.0,
    1.0, 0.0, 1.0, 1.0,
    2.0, 0.0, 1.0, 1.0,
    3.0, 0.0, 1.0, 1.0,
    4.0, 0.0, 1.0, 1.0,
    5.0, 0.0, 1.0, 1.0,

    /* Case 8: N=6 */
    0.0, 1.0,  1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 1.0,  1.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 1.0,
    0.0, 1.0, 1.1921e-07, 0.0,
    0.0, 1.0, 2.4074e-35, 0.0,
    0.0, 1.0, 2.4074e-35, 0.0,
    0.0, 1.0, 2.4074e-35, 0.0,
    0.0, 1.0, 2.4074e-35, 0.0,
    0.0, 1.0, 1.1921e-07, 0.0,

    /* Case 9: N=6 */
    0.0, 1.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    1.0, 0.0,  0.0, 1.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  1.0, 0.0,  0.0, 1.0,
    0.0, 1.0, 1.1921e-07, 0.0,
    0.0, 1.0, 2.4074e-35, 0.0,
    0.0, 1.0, 2.4074e-35, 0.0,
    0.0, 1.0, 2.4074e-35, 0.0,
    0.0, 1.0, 2.4074e-35, 0.0,
    0.0, 1.0, 1.1921e-07, 0.0,

    /* Case 10: N=4 */
    9.4480e-01, 1.0,  6.7670e-01, 1.0,  6.9080e-01, 1.0,  5.9650e-01, 1.0,
    5.8760e-01, 1.0,  8.6420e-01, 1.0,  6.7690e-01, 1.0,  7.2600e-02, 1.0,
    7.2560e-01, 1.0,  1.9430e-01, 1.0,  9.6870e-01, 1.0,  2.8310e-01, 1.0,
    2.8490e-01, 1.0,  5.8000e-02, 1.0,  4.8450e-01, 1.0,  7.3610e-01, 1.0,
    2.6014e-01, -1.7813e-01, 8.5279e-01, 3.2881e-01,
    2.8961e-01, 2.0772e-01, 8.4871e-01, 3.2358e-01,
    7.3990e-01, -4.6522e-04, 9.7398e-01, 3.4994e-01,
    2.2242e+00, 3.9709e+00, 9.8325e-01, 4.1429e+00,

    /* Case 11: N=4 */
    2.1130e-01, 9.9330e-01,  8.0960e-01, 4.2370e-01,  4.8320e-01, 1.1670e-01,  6.5380e-01, 4.9430e-01,
    8.2400e-02, 8.3600e-01,  8.4740e-01, 2.6130e-01,  6.1350e-01, 6.2500e-01,  4.8990e-01, 3.6500e-02,
    7.5990e-01, 7.4690e-01,  4.5240e-01, 2.4030e-01,  2.7490e-01, 5.5100e-01,  7.7410e-01, 2.2600e-01,
    8.7000e-03, 3.7800e-02,  8.0750e-01, 3.4050e-01,  8.8070e-01, 3.5500e-01,  9.6260e-01, 8.1590e-01,
    -6.2157e-01, 6.0607e-01, 8.7533e-01, 8.1980e-01,
    2.8890e-01, -2.6354e-01, 8.2538e-01, 8.1086e-01,
    3.8017e-01, 5.4217e-01, 7.4771e-01, 7.0323e-01,
    2.2487e+00, 1.7368e+00, 9.2372e-01, 2.2178e+00,

    /* Case 12: N=3 */
    1.0, 2.0,  3.0, 4.0,  21.0, 22.0,
    43.0, 44.0,  13.0, 14.0,  15.0, 16.0,
    5.0, 6.0,  7.0, 8.0,  25.0, 26.0,
    -7.4775e+00, 6.8803e+00, 3.9550e-01, 1.6583e+01,
    6.7009e+00, -7.8760e+00, 3.9828e-01, 1.6312e+01,
    3.9777e+01, 4.2996e+01, 7.9686e-01, 3.7399e+01,

    /* Case 13: N=4 */
    5.0, 9.0,  5.0, 5.0,  -6.0, -6.0,  -7.0, -7.0,
    3.0, 3.0,  6.0, 10.0,  -5.0, -5.0,  -6.0, -6.0,
    2.0, 2.0,  3.0, 3.0,  -1.0, 3.0,  -5.0, -5.0,
    1.0, 1.0,  2.0, 2.0,  -3.0, -3.0,  0.0, 4.0,
    1.0, 5.0, 2.1822e-01, 7.4651e-01,
    2.0, 6.0, 2.1822e-01, 3.0893e-01,
    3.0, 7.0, 2.1822e-01, 1.8315e-01,
    4.0, 8.0, 2.1822e-01, 6.6350e-01,

    /* Case 14: N=4 */
    3.0, 0.0,  1.0, 0.0,  0.0, 0.0,  0.0, 2.0,
    1.0, 0.0,  3.0, 0.0,  0.0, -2.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 2.0,  1.0, 0.0,  1.0, 0.0,
    0.0, -2.0,  0.0, 0.0,  1.0, 0.0,  1.0, 0.0,
    -8.2843e-01, 1.6979e-07, 1.0, 8.2843e-01,
    4.1744e-07, 7.1526e-08, 1.0, 8.2843e-01,
    4.0, 1.6690e-07, 1.0, 8.2843e-01,
    4.8284e+00, 6.8633e-08, 1.0, 8.2843e-01,

    /* Case 15: N=4 */
    7.0, 0.0,  3.0, 0.0,  1.0, 2.0,  -1.0, 2.0,
    3.0, 0.0,  7.0, 0.0,  1.0, -2.0,  -1.0, -2.0,
    1.0, -2.0,  1.0, 2.0,  7.0, 0.0,  -3.0, 0.0,
    -1.0, -2.0,  -2.0, 2.0,  -3.0, 0.0,  7.0, 0.0,
    -8.0767e-03, -2.5211e-01, 9.9864e-01, 7.7961e+00,
    7.7723e+00, 2.4349e-01, 7.0272e-01, 3.3337e-01,
    8.0, -3.4273e-07, 7.0711e-01, 3.3337e-01,
    1.2236e+01, 8.6188e-03, 9.9021e-01, 3.9429e+00,

    /* Case 16: N=5 */
    1.0, 2.0,  3.0, 4.0,  21.0, 22.0,  23.0, 24.0,  41.0, 42.0,
    43.0, 44.0,  13.0, 14.0,  15.0, 16.0,  33.0, 34.0,  35.0, 36.0,
    5.0, 6.0,  7.0, 8.0,  25.0, 26.0,  27.0, 28.0,  45.0, 46.0,
    47.0, 48.0,  17.0, 18.0,  19.0, 20.0,  37.0, 38.0,  39.0, 40.0,
    9.0, 10.0,  11.0, 12.0,  29.0, 30.0,  31.0, 32.0,  49.0, 50.0,
    -9.4600e+00, 7.2802e+00, 3.1053e-01, 1.1937e+01,
    -7.7912e-06, -1.2743e-05, 2.9408e-01, 1.6030e-05,
    -7.3042e-06, 3.2789e-06, 7.2259e-01, 6.7794e-06,
    7.0733e+00, -9.5584e+00, 3.0911e-01, 1.1891e+01,
    1.2739e+02, 1.3228e+02, 9.2770e-01, 1.2111e+02,

    /* Case 17: N=3 */
    1.0, 1.0,  -1.0, -1.0,  2.0, 2.0,
    0.0, 0.0,  0.0, 1.0,  2.0, 0.0,
    0.0, 0.0,  -1.0, 0.0,  3.0, 1.0,
    1.0, 1.0, 3.0151e-01, 0.0,
    1.0, 1.0, 3.1623e-01, 0.0,
    2.0, 1.0, 2.2361e-01, 1.0,

    /* Case 18: N=4, ISRT=1 */
    -4.0, -2.0,  -5.0, -6.0,  -2.0, -6.0,  0.0, -2.0,
    1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  1.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  1.0, 0.0,  0.0, 0.0,
    -9.9883e-01, -1.0006e+00, 1.3180e-04, 2.4106e-04,
    -1.0012e+00, -9.9945e-01, 1.3140e-04, 2.4041e-04,
    -9.9947e-01, -6.8325e-04, 1.3989e-04, 8.7487e-05,
    -1.0005e+00, 6.8556e-04, 1.4010e-04, 8.7750e-05,

    /* Case 19: N=7 */
    2.0, 4.0,  1.0, 1.0,  6.0, 2.0,  3.0, 3.0,  5.0, 5.0,  2.0, 6.0,  1.0, 1.0,
    1.0, 2.0,  1.0, 3.0,  3.0, 1.0,  5.0, -4.0,  1.0, 1.0,  7.0, 2.0,  2.0, 3.0,
    0.0, 0.0,  3.0, -2.0,  1.0, 1.0,  6.0, 3.0,  2.0, 1.0,  1.0, 4.0,  2.0, 1.0,
    0.0, 0.0,  0.0, 0.0,  2.0, 3.0,  3.0, 1.0,  1.0, 2.0,  2.0, 2.0,  3.0, 1.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  2.0, -1.0,  2.0, 2.0,  3.0, 1.0,  1.0, 3.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  1.0, -1.0,  2.0, 1.0,  2.0, 2.0,
    0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  2.0, -2.0,  1.0, 1.0,
    -2.7081e+00, -2.8029e+00, 6.9734e-01, 3.9279e+00,
    -1.1478e+00, 8.0176e-01, 6.5772e-01, 9.4243e-01,
    -8.0109e-01, 4.9694e+00, 4.6751e-01, 1.3779e+00,
    9.9492e-01, 3.1688e+00, 3.5095e-01, 5.9845e-01,
    2.0809e+00, 1.9341e+00, 4.9042e-01, 3.9035e-01,
    5.3138e+00, 1.2242e+00, 3.0213e-01, 7.1268e-01,
    8.2674e+00, 3.7047e+00, 2.8270e-01, 3.2849e+00,

    /* Case 20: N=5, ISRT=1 */
    0.0, 5.0,  1.0, 2.0,  2.0, 3.0,  -3.0, 6.0,  6.0, 0.0,
    -1.0, 2.0,  0.0, 6.0,  4.0, 5.0,  -3.0, -2.0,  5.0, 0.0,
    -2.0, 3.0,  -4.0, 5.0,  0.0, 7.0,  3.0, 0.0,  2.0, 0.0,
    3.0, 6.0,  3.0, -2.0,  -3.0, 0.0,  0.0, -5.0,  2.0, 1.0,
    -6.0, 0.0,  -5.0, 0.0,  -2.0, 0.0,  -2.0, 1.0,  0.0, 2.0,
    -4.1735e-08, -1.0734e+01, 1.0, 7.7345e+00,
    -2.6397e-07, -2.9991e+00, 1.0, 4.5989e+00,
    1.4565e-07, 1.5998e+00, 1.0, 4.5989e+00,
    -4.4369e-07, 9.3159e+00, 1.0, 7.7161e+00,
    4.0937e-09, 1.7817e+01, 1.0, 8.5013e+00,

    /* Case 21: N=3 */
    2.0, 0.0,  0.0, -1.0,  0.0, 0.0,
    0.0, 1.0,  2.0, 0.0,  0.0, 0.0,
    0.0, 0.0,  0.0, 0.0,  3.0, 0.0,
    1.0, 0.0, 1.0, 2.0,
    3.0, 0.0, 1.0, 0.0,
    3.0, 0.0, 1.0, 0.0,
};

static const INT zget37_n[] = {
    1, 1, 2, 2, 2, 5, 5, 5, 6, 6, 4, 4, 3, 4, 4, 4, 5, 3, 4, 7, 5, 3,
};

static const INT zget37_isrt[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
};

static void rowpairs_to_colmajor_c128(const f64* pairs, c128* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(c128));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = CMPLX(pairs[(i * n + j) * 2],
                                      pairs[(i * n + j) * 2 + 1]);
}

void zget37(f64 rmax[3], INT lmax[3], INT ninfo[3], INT* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE  = 1.0;
    const f64 TWO  = 2.0;
    const f64 EPSIN = 5.9605e-8;

    INT    i, icmp, info, iscl, isrt, j, kmin, m, n;
    f64    bignum, eps, smlnum, tnrm, tol, tolin, v,
           vmax, vmin, vmul;

    INT    select[LDT];
    INT    lcmp[3];
    f64    dum[1], rwork[2 * LDT], s[LDT], sep[LDT], sepin[LDT],
           septmp[LDT], sin_vals[LDT], stmp[LDT], val[3],
           wsrt[LDT];
    c128   cdum[1], le[LDT * LDT], re[LDT * LDT],
           t[LDT * LDT], tmp[LDT * LDT], w[LDT],
           work[LWORK], wtmp[LDT];

    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
    bignum = ONE / smlnum;

    eps = fmax(eps, EPSIN);
    rmax[0] = ZERO;
    rmax[1] = ZERO;
    rmax[2] = ZERO;
    lmax[0] = 0;
    lmax[1] = 0;
    lmax[2] = 0;
    *knt = 0;
    ninfo[0] = 0;
    ninfo[1] = 0;
    ninfo[2] = 0;

    val[0] = sqrt(smlnum);
    val[1] = ONE;
    val[2] = sqrt(bignum);

    INT data_offset = 0;

    for (INT icase = 0; icase < NCASES37; icase++) {
        n = zget37_n[icase];
        isrt = zget37_isrt[icase];

        rowpairs_to_colmajor_c128(&zget37_data[data_offset], tmp, n, LDT);
        data_offset += n * n * 2;

        for (i = 0; i < n; i++) {
            sin_vals[i] = zget37_data[data_offset + i * 4 + 2];
            sepin[i]    = zget37_data[data_offset + i * 4 + 3];
        }
        data_offset += n * 4;

        tnrm = zlange("M", n, n, tmp, LDT, rwork);

        for (iscl = 0; iscl < 3; iscl++) {

            *knt = *knt + 1;
            zlacpy("F", n, n, tmp, LDT, t, LDT);
            vmul = val[iscl];
            for (i = 0; i < n; i++)
                cblas_zdscal(n, vmul, &t[i * LDT], 1);
            if (tnrm == ZERO)
                vmul = ONE;

            zgehrd(n, 0, n - 1, t, LDT, &work[0], &work[n], LWORK - n,
                   &info);
            if (info != 0) {
                lmax[0] = *knt;
                ninfo[0] = ninfo[0] + 1;
                continue;
            }
            for (j = 0; j < n - 2; j++)
                for (i = j + 2; i < n; i++)
                    t[i + j * LDT] = CMPLX(0.0, 0.0);

            zhseqr("S", "N", n, 0, n - 1, t, LDT, w, cdum, 1, work,
                   LWORK, &info);
            if (info != 0) {
                lmax[1] = *knt;
                ninfo[1] = ninfo[1] + 1;
                continue;
            }

            for (i = 0; i < n; i++)
                select[i] = 1;
            ztrevc("Both", "All", select, n, t, LDT, le, LDT, re,
                   LDT, n, &m, work, rwork, &info);

            ztrsna("Both", "All", select, n, t, LDT, le, LDT, re,
                   LDT, s, sep, n, &m, work, n, rwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }

            cblas_zcopy(n, w, 1, wtmp, 1);
            if (isrt == 0) {
                for (i = 0; i < n; i++)
                    wsrt[i] = creal(w[i]);
            } else {
                for (i = 0; i < n; i++)
                    wsrt[i] = cimag(w[i]);
            }
            cblas_dcopy(n, s, 1, stmp, 1);
            cblas_dcopy(n, sep, 1, septmp, 1);
            cblas_dscal(n, ONE / vmul, septmp, 1);
            for (i = 0; i < n - 1; i++) {
                kmin = i;
                vmin = wsrt[i];
                for (j = i + 1; j < n; j++) {
                    if (wsrt[j] < vmin) {
                        kmin = j;
                        vmin = wsrt[j];
                    }
                }
                wsrt[kmin] = wsrt[i];
                wsrt[i] = vmin;
                f64 vcmin = creal(wtmp[i]);
                wtmp[i] = w[kmin];
                wtmp[kmin] = vcmin;
                vmin = stmp[kmin];
                stmp[kmin] = stmp[i];
                stmp[i] = vmin;
                vmin = septmp[kmin];
                septmp[kmin] = septmp[i];
                septmp[i] = vmin;
            }

            v = fmax(TWO * (f64)n * eps * tnrm, smlnum);
            if (tnrm == ZERO)
                v = ONE;
            for (i = 0; i < n; i++) {
                if (v > septmp[i])
                    tol = ONE;
                else
                    tol = v / septmp[i];
                if (v > sepin[i])
                    tolin = ONE;
                else
                    tolin = v / sepin[i];
                tol = fmax(tol, smlnum / eps);
                tolin = fmax(tolin, smlnum / eps);
                if (eps * (sin_vals[i] - tolin) > stmp[i] + tol) {
                    vmax = ONE / eps;
                } else if (sin_vals[i] - tolin > stmp[i] + tol) {
                    vmax = (sin_vals[i] - tolin) / (stmp[i] + tol);
                } else if (sin_vals[i] + tolin < eps * (stmp[i] - tol)) {
                    vmax = ONE / eps;
                } else if (sin_vals[i] + tolin < stmp[i] - tol) {
                    vmax = (stmp[i] - tol) / (sin_vals[i] + tolin);
                } else {
                    vmax = ONE;
                }
                if (vmax > rmax[1]) {
                    rmax[1] = vmax;
                    if (ninfo[1] == 0)
                        lmax[1] = *knt;
                }
            }

            for (i = 0; i < n; i++) {
                if (v > septmp[i] * stmp[i])
                    tol = septmp[i];
                else
                    tol = v / stmp[i];
                if (v > sepin[i] * sin_vals[i])
                    tolin = sepin[i];
                else
                    tolin = v / sin_vals[i];
                tol = fmax(tol, smlnum / eps);
                tolin = fmax(tolin, smlnum / eps);
                if (eps * (sepin[i] - tolin) > septmp[i] + tol) {
                    vmax = ONE / eps;
                } else if (sepin[i] - tolin > septmp[i] + tol) {
                    vmax = (sepin[i] - tolin) / (septmp[i] + tol);
                } else if (sepin[i] + tolin < eps * (septmp[i] - tol)) {
                    vmax = ONE / eps;
                } else if (sepin[i] + tolin < septmp[i] - tol) {
                    vmax = (septmp[i] - tol) / (sepin[i] + tolin);
                } else {
                    vmax = ONE;
                }
                if (vmax > rmax[1]) {
                    rmax[1] = vmax;
                    if (ninfo[1] == 0)
                        lmax[1] = *knt;
                }
            }

            for (i = 0; i < n; i++) {
                if (sin_vals[i] <= (f64)(2 * n) * eps && stmp[i] <=
                    (f64)(2 * n) * eps) {
                    vmax = ONE;
                } else if (eps * sin_vals[i] > stmp[i]) {
                    vmax = ONE / eps;
                } else if (sin_vals[i] > stmp[i]) {
                    vmax = sin_vals[i] / stmp[i];
                } else if (sin_vals[i] < eps * stmp[i]) {
                    vmax = ONE / eps;
                } else if (sin_vals[i] < stmp[i]) {
                    vmax = stmp[i] / sin_vals[i];
                } else {
                    vmax = ONE;
                }
                if (vmax > rmax[2]) {
                    rmax[2] = vmax;
                    if (ninfo[2] == 0)
                        lmax[2] = *knt;
                }
            }

            for (i = 0; i < n; i++) {
                if (sepin[i] <= v && septmp[i] <= v) {
                    vmax = ONE;
                } else if (eps * sepin[i] > septmp[i]) {
                    vmax = ONE / eps;
                } else if (sepin[i] > septmp[i]) {
                    vmax = sepin[i] / septmp[i];
                } else if (sepin[i] < eps * septmp[i]) {
                    vmax = ONE / eps;
                } else if (sepin[i] < septmp[i]) {
                    vmax = septmp[i] / sepin[i];
                } else {
                    vmax = ONE;
                }
                if (vmax > rmax[2]) {
                    rmax[2] = vmax;
                    if (ninfo[2] == 0)
                        lmax[2] = *knt;
                }
            }

            vmax = ZERO;
            dum[0] = -ONE;
            cblas_dcopy(n, dum, 0, stmp, 1);
            cblas_dcopy(n, dum, 0, septmp, 1);
            ztrsna("Eigcond", "All", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, rwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (stmp[i] != s[i])
                    vmax = ONE / eps;
                if (septmp[i] != dum[0])
                    vmax = ONE / eps;
            }

            cblas_dcopy(n, dum, 0, stmp, 1);
            cblas_dcopy(n, dum, 0, septmp, 1);
            ztrsna("Veccond", "All", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, rwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (stmp[i] != dum[0])
                    vmax = ONE / eps;
                if (septmp[i] != sep[i])
                    vmax = ONE / eps;
            }

            for (i = 0; i < n; i++)
                select[i] = 1;
            cblas_dcopy(n, dum, 0, stmp, 1);
            cblas_dcopy(n, dum, 0, septmp, 1);
            ztrsna("Bothcond", "Some", select, n, t, LDT, le, LDT,
                   re, LDT, stmp, septmp, n, &m, work, n, rwork,
                   &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (septmp[i] != sep[i])
                    vmax = ONE / eps;
                if (stmp[i] != s[i])
                    vmax = ONE / eps;
            }

            cblas_dcopy(n, dum, 0, stmp, 1);
            cblas_dcopy(n, dum, 0, septmp, 1);
            ztrsna("Eigcond", "Some", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, rwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (stmp[i] != s[i])
                    vmax = ONE / eps;
                if (septmp[i] != dum[0])
                    vmax = ONE / eps;
            }

            cblas_dcopy(n, dum, 0, stmp, 1);
            cblas_dcopy(n, dum, 0, septmp, 1);
            ztrsna("Veccond", "Some", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, rwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (stmp[i] != dum[0])
                    vmax = ONE / eps;
                if (septmp[i] != sep[i])
                    vmax = ONE / eps;
            }
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }

            for (i = 0; i < n; i++)
                select[i] = 0;
            icmp = 0;
            if (n > 1) {
                icmp = 1;
                lcmp[0] = 1;
                select[1] = 1;
                cblas_zcopy(n, &re[1 * LDT], 1, &re[0 * LDT], 1);
                cblas_zcopy(n, &le[1 * LDT], 1, &le[0 * LDT], 1);
            }
            if (n > 3) {
                icmp = 2;
                lcmp[1] = n - 2;
                select[n - 2] = 1;
                cblas_zcopy(n, &re[(n - 2) * LDT], 1, &re[1 * LDT], 1);
                cblas_zcopy(n, &le[(n - 2) * LDT], 1, &le[1 * LDT], 1);
            }

            cblas_dcopy(icmp, dum, 0, stmp, 1);
            cblas_dcopy(icmp, dum, 0, septmp, 1);
            ztrsna("Bothcond", "Some", select, n, t, LDT, le, LDT,
                   re, LDT, stmp, septmp, n, &m, work, n, rwork,
                   &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < icmp; i++) {
                j = lcmp[i];
                if (septmp[i] != sep[j])
                    vmax = ONE / eps;
                if (stmp[i] != s[j])
                    vmax = ONE / eps;
            }

            cblas_dcopy(icmp, dum, 0, stmp, 1);
            cblas_dcopy(icmp, dum, 0, septmp, 1);
            ztrsna("Eigcond", "Some", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, rwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < icmp; i++) {
                j = lcmp[i];
                if (stmp[i] != s[j])
                    vmax = ONE / eps;
                if (septmp[i] != dum[0])
                    vmax = ONE / eps;
            }

            cblas_dcopy(icmp, dum, 0, stmp, 1);
            cblas_dcopy(icmp, dum, 0, septmp, 1);
            ztrsna("Veccond", "Some", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, rwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < icmp; i++) {
                j = lcmp[i];
                if (stmp[i] != dum[0])
                    vmax = ONE / eps;
                if (septmp[i] != sep[j])
                    vmax = ONE / eps;
            }
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }

        } /* end iscl loop */
    } /* end icase loop */
}
