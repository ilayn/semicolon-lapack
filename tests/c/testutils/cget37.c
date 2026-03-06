/**
 * @file cget37.c
 * @brief CGET37 tests CTRSNA, a routine for estimating condition numbers of
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
 * Embedded test data from LAPACK zec.in (cget37 section, lines 177-368).
 * 22 test cases. Format per case:
 *   n*n complex matrix values stored as (re,im) pairs in row-major order,
 *   then n*(wrin,wiin,sin,sepin) quadruplets as 4 doubles.
 */
static const f32 zget37_data[] = {
    /* Case 0: N=1 */
    0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,

    /* Case 1: N=1 */
    0.0f, 1.0f,
    0.0f, 1.0f, 1.0f, 1.0f,

    /* Case 2: N=2 */
    0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,

    /* Case 3: N=2 */
    3.0f, 0.0f,  2.0f, 0.0f,
    2.0f, 0.0f,  3.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 4.0f,
    5.0f, 0.0f, 1.0f, 4.0f,

    /* Case 4: N=2 */
    3.0f, 0.0f,  0.0f, 2.0f,
    0.0f, 2.0f,  3.0f, 0.0f,
    3.0f, 2.0f, 1.0f, 4.0f,
    3.0f, -2.0f, 1.0f, 4.0f,

    /* Case 5: N=5 */
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,

    /* Case 6: N=5 */
    1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  1.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f,

    /* Case 7: N=5 */
    1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  2.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  3.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  4.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  5.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 1.0f,
    2.0f, 0.0f, 1.0f, 1.0f,
    3.0f, 0.0f, 1.0f, 1.0f,
    4.0f, 0.0f, 1.0f, 1.0f,
    5.0f, 0.0f, 1.0f, 1.0f,

    /* Case 8: N=6 */
    0.0f, 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,
    0.0f, 1.0f, 1.1921e-07f, 0.0f,
    0.0f, 1.0f, 2.4074e-35f, 0.0f,
    0.0f, 1.0f, 2.4074e-35f, 0.0f,
    0.0f, 1.0f, 2.4074e-35f, 0.0f,
    0.0f, 1.0f, 2.4074e-35f, 0.0f,
    0.0f, 1.0f, 1.1921e-07f, 0.0f,

    /* Case 9: N=6 */
    0.0f, 1.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    1.0f, 0.0f,  0.0f, 1.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 1.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 1.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 1.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 1.0f,
    0.0f, 1.0f, 1.1921e-07f, 0.0f,
    0.0f, 1.0f, 2.4074e-35f, 0.0f,
    0.0f, 1.0f, 2.4074e-35f, 0.0f,
    0.0f, 1.0f, 2.4074e-35f, 0.0f,
    0.0f, 1.0f, 2.4074e-35f, 0.0f,
    0.0f, 1.0f, 1.1921e-07f, 0.0f,

    /* Case 10: N=4 */
    9.4480e-01f, 1.0f,  6.7670e-01f, 1.0f,  6.9080e-01f, 1.0f,  5.9650e-01f, 1.0f,
    5.8760e-01f, 1.0f,  8.6420e-01f, 1.0f,  6.7690e-01f, 1.0f,  7.2600e-02f, 1.0f,
    7.2560e-01f, 1.0f,  1.9430e-01f, 1.0f,  9.6870e-01f, 1.0f,  2.8310e-01f, 1.0f,
    2.8490e-01f, 1.0f,  5.8000e-02f, 1.0f,  4.8450e-01f, 1.0f,  7.3610e-01f, 1.0f,
    2.6014e-01f, -1.7813e-01f, 8.5279e-01f, 3.2881e-01f,
    2.8961e-01f, 2.0772e-01f, 8.4871e-01f, 3.2358e-01f,
    7.3990e-01f, -4.6522e-04f, 9.7398e-01f, 3.4994e-01f,
    2.2242e+00f, 3.9709e+00f, 9.8325e-01f, 4.1429e+00f,

    /* Case 11: N=4 */
    2.1130e-01f, 9.9330e-01f,  8.0960e-01f, 4.2370e-01f,  4.8320e-01f, 1.1670e-01f,  6.5380e-01f, 4.9430e-01f,
    8.2400e-02f, 8.3600e-01f,  8.4740e-01f, 2.6130e-01f,  6.1350e-01f, 6.2500e-01f,  4.8990e-01f, 3.6500e-02f,
    7.5990e-01f, 7.4690e-01f,  4.5240e-01f, 2.4030e-01f,  2.7490e-01f, 5.5100e-01f,  7.7410e-01f, 2.2600e-01f,
    8.7000e-03f, 3.7800e-02f,  8.0750e-01f, 3.4050e-01f,  8.8070e-01f, 3.5500e-01f,  9.6260e-01f, 8.1590e-01f,
    -6.2157e-01f, 6.0607e-01f, 8.7533e-01f, 8.1980e-01f,
    2.8890e-01f, -2.6354e-01f, 8.2538e-01f, 8.1086e-01f,
    3.8017e-01f, 5.4217e-01f, 7.4771e-01f, 7.0323e-01f,
    2.2487e+00f, 1.7368e+00f, 9.2372e-01f, 2.2178e+00f,

    /* Case 12: N=3 */
    1.0f, 2.0f,  3.0f, 4.0f,  21.0f, 22.0f,
    43.0f, 44.0f,  13.0f, 14.0f,  15.0f, 16.0f,
    5.0f, 6.0f,  7.0f, 8.0f,  25.0f, 26.0f,
    -7.4775e+00f, 6.8803e+00f, 3.9550e-01f, 1.6583e+01f,
    6.7009e+00f, -7.8760e+00f, 3.9828e-01f, 1.6312e+01f,
    3.9777e+01f, 4.2996e+01f, 7.9686e-01f, 3.7399e+01f,

    /* Case 13: N=4 */
    5.0f, 9.0f,  5.0f, 5.0f,  -6.0f, -6.0f,  -7.0f, -7.0f,
    3.0f, 3.0f,  6.0f, 10.0f,  -5.0f, -5.0f,  -6.0f, -6.0f,
    2.0f, 2.0f,  3.0f, 3.0f,  -1.0f, 3.0f,  -5.0f, -5.0f,
    1.0f, 1.0f,  2.0f, 2.0f,  -3.0f, -3.0f,  0.0f, 4.0f,
    1.0f, 5.0f, 2.1822e-01f, 7.4651e-01f,
    2.0f, 6.0f, 2.1822e-01f, 3.0893e-01f,
    3.0f, 7.0f, 2.1822e-01f, 1.8315e-01f,
    4.0f, 8.0f, 2.1822e-01f, 6.6350e-01f,

    /* Case 14: N=4 */
    3.0f, 0.0f,  1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 2.0f,
    1.0f, 0.0f,  3.0f, 0.0f,  0.0f, -2.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 2.0f,  1.0f, 0.0f,  1.0f, 0.0f,
    0.0f, -2.0f,  0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 0.0f,
    -8.2843e-01f, 1.6979e-07f, 1.0f, 8.2843e-01f,
    4.1744e-07f, 7.1526e-08f, 1.0f, 8.2843e-01f,
    4.0f, 1.6690e-07f, 1.0f, 8.2843e-01f,
    4.8284e+00f, 6.8633e-08f, 1.0f, 8.2843e-01f,

    /* Case 15: N=4 */
    7.0f, 0.0f,  3.0f, 0.0f,  1.0f, 2.0f,  -1.0f, 2.0f,
    3.0f, 0.0f,  7.0f, 0.0f,  1.0f, -2.0f,  -1.0f, -2.0f,
    1.0f, -2.0f,  1.0f, 2.0f,  7.0f, 0.0f,  -3.0f, 0.0f,
    -1.0f, -2.0f,  -2.0f, 2.0f,  -3.0f, 0.0f,  7.0f, 0.0f,
    -8.0767e-03f, -2.5211e-01f, 9.9864e-01f, 7.7961e+00f,
    7.7723e+00f, 2.4349e-01f, 7.0272e-01f, 3.3337e-01f,
    8.0f, -3.4273e-07f, 7.0711e-01f, 3.3337e-01f,
    1.2236e+01f, 8.6188e-03f, 9.9021e-01f, 3.9429e+00f,

    /* Case 16: N=5 */
    1.0f, 2.0f,  3.0f, 4.0f,  21.0f, 22.0f,  23.0f, 24.0f,  41.0f, 42.0f,
    43.0f, 44.0f,  13.0f, 14.0f,  15.0f, 16.0f,  33.0f, 34.0f,  35.0f, 36.0f,
    5.0f, 6.0f,  7.0f, 8.0f,  25.0f, 26.0f,  27.0f, 28.0f,  45.0f, 46.0f,
    47.0f, 48.0f,  17.0f, 18.0f,  19.0f, 20.0f,  37.0f, 38.0f,  39.0f, 40.0f,
    9.0f, 10.0f,  11.0f, 12.0f,  29.0f, 30.0f,  31.0f, 32.0f,  49.0f, 50.0f,
    -9.4600e+00f, 7.2802e+00f, 3.1053e-01f, 1.1937e+01f,
    -7.7912e-06f, -1.2743e-05f, 2.9408e-01f, 1.6030e-05f,
    -7.3042e-06f, 3.2789e-06f, 7.2259e-01f, 6.7794e-06f,
    7.0733e+00f, -9.5584e+00f, 3.0911e-01f, 1.1891e+01f,
    1.2739e+02f, 1.3228e+02f, 9.2770e-01f, 1.2111e+02f,

    /* Case 17: N=3 */
    1.0f, 1.0f,  -1.0f, -1.0f,  2.0f, 2.0f,
    0.0f, 0.0f,  0.0f, 1.0f,  2.0f, 0.0f,
    0.0f, 0.0f,  -1.0f, 0.0f,  3.0f, 1.0f,
    1.0f, 1.0f, 3.0151e-01f, 0.0f,
    1.0f, 1.0f, 3.1623e-01f, 0.0f,
    2.0f, 1.0f, 2.2361e-01f, 1.0f,

    /* Case 18: N=4, ISRT=1 */
    -4.0f, -2.0f,  -5.0f, -6.0f,  -2.0f, -6.0f,  0.0f, -2.0f,
    1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 0.0f,
    -9.9883e-01f, -1.0006e+00f, 1.3180e-04f, 2.4106e-04f,
    -1.0012e+00f, -9.9945e-01f, 1.3140e-04f, 2.4041e-04f,
    -9.9947e-01f, -6.8325e-04f, 1.3989e-04f, 8.7487e-05f,
    -1.0005e+00f, 6.8556e-04f, 1.4010e-04f, 8.7750e-05f,

    /* Case 19: N=7 */
    2.0f, 4.0f,  1.0f, 1.0f,  6.0f, 2.0f,  3.0f, 3.0f,  5.0f, 5.0f,  2.0f, 6.0f,  1.0f, 1.0f,
    1.0f, 2.0f,  1.0f, 3.0f,  3.0f, 1.0f,  5.0f, -4.0f,  1.0f, 1.0f,  7.0f, 2.0f,  2.0f, 3.0f,
    0.0f, 0.0f,  3.0f, -2.0f,  1.0f, 1.0f,  6.0f, 3.0f,  2.0f, 1.0f,  1.0f, 4.0f,  2.0f, 1.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  2.0f, 3.0f,  3.0f, 1.0f,  1.0f, 2.0f,  2.0f, 2.0f,  3.0f, 1.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  2.0f, -1.0f,  2.0f, 2.0f,  3.0f, 1.0f,  1.0f, 3.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  1.0f, -1.0f,  2.0f, 1.0f,  2.0f, 2.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  2.0f, -2.0f,  1.0f, 1.0f,
    -2.7081e+00f, -2.8029e+00f, 6.9734e-01f, 3.9279e+00f,
    -1.1478e+00f, 8.0176e-01f, 6.5772e-01f, 9.4243e-01f,
    -8.0109e-01f, 4.9694e+00f, 4.6751e-01f, 1.3779e+00f,
    9.9492e-01f, 3.1688e+00f, 3.5095e-01f, 5.9845e-01f,
    2.0809e+00f, 1.9341e+00f, 4.9042e-01f, 3.9035e-01f,
    5.3138e+00f, 1.2242e+00f, 3.0213e-01f, 7.1268e-01f,
    8.2674e+00f, 3.7047e+00f, 2.8270e-01f, 3.2849e+00f,

    /* Case 20: N=5, ISRT=1 */
    0.0f, 5.0f,  1.0f, 2.0f,  2.0f, 3.0f,  -3.0f, 6.0f,  6.0f, 0.0f,
    -1.0f, 2.0f,  0.0f, 6.0f,  4.0f, 5.0f,  -3.0f, -2.0f,  5.0f, 0.0f,
    -2.0f, 3.0f,  -4.0f, 5.0f,  0.0f, 7.0f,  3.0f, 0.0f,  2.0f, 0.0f,
    3.0f, 6.0f,  3.0f, -2.0f,  -3.0f, 0.0f,  0.0f, -5.0f,  2.0f, 1.0f,
    -6.0f, 0.0f,  -5.0f, 0.0f,  -2.0f, 0.0f,  -2.0f, 1.0f,  0.0f, 2.0f,
    -4.1735e-08f, -1.0734e+01f, 1.0f, 7.7345e+00f,
    -2.6397e-07f, -2.9991e+00f, 1.0f, 4.5989e+00f,
    1.4565e-07f, 1.5998e+00f, 1.0f, 4.5989e+00f,
    -4.4369e-07f, 9.3159e+00f, 1.0f, 7.7161e+00f,
    4.0937e-09f, 1.7817e+01f, 1.0f, 8.5013e+00f,

    /* Case 21: N=3 */
    2.0f, 0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
    0.0f, 1.0f,  2.0f, 0.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,  3.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 2.0f,
    3.0f, 0.0f, 1.0f, 0.0f,
    3.0f, 0.0f, 1.0f, 0.0f,
};

static const INT zget37_n[] = {
    1, 1, 2, 2, 2, 5, 5, 5, 6, 6, 4, 4, 3, 4, 4, 4, 5, 3, 4, 7, 5, 3,
};

static const INT zget37_isrt[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
};

static void rowpairs_to_colmajor_c128(const f32* pairs, c64* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(c64));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = CMPLXF(pairs[(i * n + j) * 2],
                                      pairs[(i * n + j) * 2 + 1]);
}

void cget37(f32 rmax[3], INT lmax[3], INT ninfo[3], INT* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE  = 1.0f;
    const f32 TWO  = 2.0f;
    const f32 EPSIN = 5.9605e-8f;

    INT    i, icmp, info, iscl, isrt, j, kmin, m, n;
    f32    bignum, eps, smlnum, tnrm, tol, tolin, v,
           vmax, vmin, vmul;

    INT    select[LDT];
    INT    lcmp[3];
    f32    dum[1], rwork[2 * LDT], s[LDT], sep[LDT], sepin[LDT],
           septmp[LDT], sin_vals[LDT], stmp[LDT], val[3],
           wsrt[LDT];
    c64   cdum[1], le[LDT * LDT], re[LDT * LDT],
           t[LDT * LDT], tmp[LDT * LDT], w[LDT],
           work[LWORK], wtmp[LDT];

    eps = slamch("P");
    smlnum = slamch("S") / eps;
    bignum = ONE / smlnum;

    eps = fmaxf(eps, EPSIN);
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

    val[0] = sqrtf(smlnum);
    val[1] = ONE;
    val[2] = sqrtf(bignum);

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

        tnrm = clange("M", n, n, tmp, LDT, rwork);

        for (iscl = 0; iscl < 3; iscl++) {

            *knt = *knt + 1;
            clacpy("F", n, n, tmp, LDT, t, LDT);
            vmul = val[iscl];
            for (i = 0; i < n; i++)
                cblas_csscal(n, vmul, &t[i * LDT], 1);
            if (tnrm == ZERO)
                vmul = ONE;

            cgehrd(n, 0, n - 1, t, LDT, &work[0], &work[n], LWORK - n,
                   &info);
            if (info != 0) {
                lmax[0] = *knt;
                ninfo[0] = ninfo[0] + 1;
                continue;
            }
            for (j = 0; j < n - 2; j++)
                for (i = j + 2; i < n; i++)
                    t[i + j * LDT] = CMPLXF(0.0f, 0.0f);

            chseqr("S", "N", n, 0, n - 1, t, LDT, w, cdum, 1, work,
                   LWORK, &info);
            if (info != 0) {
                lmax[1] = *knt;
                ninfo[1] = ninfo[1] + 1;
                continue;
            }

            for (i = 0; i < n; i++)
                select[i] = 1;
            ctrevc("Both", "All", select, n, t, LDT, le, LDT, re,
                   LDT, n, &m, work, rwork, &info);

            ctrsna("Both", "All", select, n, t, LDT, le, LDT, re,
                   LDT, s, sep, n, &m, work, n, rwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }

            cblas_ccopy(n, w, 1, wtmp, 1);
            if (isrt == 0) {
                for (i = 0; i < n; i++)
                    wsrt[i] = crealf(w[i]);
            } else {
                for (i = 0; i < n; i++)
                    wsrt[i] = cimagf(w[i]);
            }
            cblas_scopy(n, s, 1, stmp, 1);
            cblas_scopy(n, sep, 1, septmp, 1);
            cblas_sscal(n, ONE / vmul, septmp, 1);
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
                f32 vcmin = crealf(wtmp[i]);
                wtmp[i] = w[kmin];
                wtmp[kmin] = vcmin;
                vmin = stmp[kmin];
                stmp[kmin] = stmp[i];
                stmp[i] = vmin;
                vmin = septmp[kmin];
                septmp[kmin] = septmp[i];
                septmp[i] = vmin;
            }

            v = fmaxf(TWO * (f32)n * eps * tnrm, smlnum);
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
                tol = fmaxf(tol, smlnum / eps);
                tolin = fmaxf(tolin, smlnum / eps);
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
                tol = fmaxf(tol, smlnum / eps);
                tolin = fmaxf(tolin, smlnum / eps);
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
                if (sin_vals[i] <= (f32)(2 * n) * eps && stmp[i] <=
                    (f32)(2 * n) * eps) {
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
            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            ctrsna("Eigcond", "All", select, n, t, LDT, le, LDT, re,
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

            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            ctrsna("Veccond", "All", select, n, t, LDT, le, LDT, re,
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
            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            ctrsna("Bothcond", "Some", select, n, t, LDT, le, LDT,
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

            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            ctrsna("Eigcond", "Some", select, n, t, LDT, le, LDT, re,
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

            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            ctrsna("Veccond", "Some", select, n, t, LDT, le, LDT, re,
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
                cblas_ccopy(n, &re[1 * LDT], 1, &re[0 * LDT], 1);
                cblas_ccopy(n, &le[1 * LDT], 1, &le[0 * LDT], 1);
            }
            if (n > 3) {
                icmp = 2;
                lcmp[1] = n - 2;
                select[n - 2] = 1;
                cblas_ccopy(n, &re[(n - 2) * LDT], 1, &re[1 * LDT], 1);
                cblas_ccopy(n, &le[(n - 2) * LDT], 1, &le[1 * LDT], 1);
            }

            cblas_scopy(icmp, dum, 0, stmp, 1);
            cblas_scopy(icmp, dum, 0, septmp, 1);
            ctrsna("Bothcond", "Some", select, n, t, LDT, le, LDT,
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

            cblas_scopy(icmp, dum, 0, stmp, 1);
            cblas_scopy(icmp, dum, 0, septmp, 1);
            ctrsna("Eigcond", "Some", select, n, t, LDT, le, LDT, re,
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

            cblas_scopy(icmp, dum, 0, stmp, 1);
            cblas_scopy(icmp, dum, 0, septmp, 1);
            ctrsna("Veccond", "Some", select, n, t, LDT, le, LDT, re,
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
