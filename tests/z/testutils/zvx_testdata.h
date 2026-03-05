/**
 * @file zvx_testdata.h
 * @brief Precomputed test matrices for ZGEEVX (from LAPACK TESTING/zed.in)
 *
 * 22 matrices with precomputed eigenvalues and reciprocal condition numbers.
 * Used by test_zdrvvx.c for tests 10-11 (condition number accuracy).
 */

#ifndef ZVX_TESTDATA_H
#define ZVX_TESTDATA_H

typedef struct {
    int n;
    int isrt;
    const c128* A;
    const c128* W;
    const f64* rcdein;
    const f64* rcdvin;
} zvx_precomputed_t;

static const c128 zvx_A_0[1] = {CMPLX(0.0, 0.0)};
static const c128 zvx_W_0[1] = {CMPLX(0.0, 0.0)};
static const f64 zvx_rcdein_0[1] = {1.0000e+00};
static const f64 zvx_rcdvin_0[1] = {0.0000e+00};

static const c128 zvx_A_1[1] = {CMPLX(0.0, 1.0)};
static const c128 zvx_W_1[1] = {CMPLX(0.0, 1.0)};
static const f64 zvx_rcdein_1[1] = {1.0000e+00};
static const f64 zvx_rcdvin_1[1] = {1.0000e+00};

static const c128 zvx_A_2[4] = {
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
};
static const c128 zvx_W_2[2] = {CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)};
static const f64 zvx_rcdein_2[2] = {1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_2[2] = {0.0000e+00, 0.0000e+00};

static const c128 zvx_A_3[4] = {
    CMPLX(3.0, 0.0), CMPLX(2.0, 0.0),
    CMPLX(2.0, 0.0), CMPLX(3.0, 0.0)
};
static const c128 zvx_W_3[2] = {CMPLX(1.0, 0.0), CMPLX(5.0, 0.0)};
static const f64 zvx_rcdein_3[2] = {1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_3[2] = {4.0000e+00, 4.0000e+00};

static const c128 zvx_A_4[4] = {
    3.0, CMPLX(0.0, 2.0),
    CMPLX(0.0, 2.0), 3.0
};
static const c128 zvx_W_4[2] = {CMPLX(3.0, 2.0), CMPLX(3.0, -2.0)};
static const f64 zvx_rcdein_4[2] = {1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_4[2] = {4.0000e+00, 4.0000e+00};

static const c128 zvx_A_5[25] = {
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
};
static const c128 zvx_W_5[5] = {CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)};
static const f64 zvx_rcdein_5[5] = {1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_5[5] = {0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00};

static const c128 zvx_A_6[25] = {
    CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
};
static const c128 zvx_W_6[5] = {CMPLX(1.0, 0.0), CMPLX(1.0, 0.0), CMPLX(1.0, 0.0), CMPLX(1.0, 0.0), CMPLX(1.0, 0.0)};
static const f64 zvx_rcdein_6[5] = {1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_6[5] = {0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00};

static const c128 zvx_A_7[25] = {
    CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(2.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(5.0, 0.0)
};
static const c128 zvx_W_7[5] = {CMPLX(1.0, 0.0), CMPLX(2.0, 0.0), CMPLX(3.0, 0.0), CMPLX(4.0, 0.0), CMPLX(5.0, 0.0)};
static const f64 zvx_rcdein_7[5] = {1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_7[5] = {1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00};

static const c128 zvx_A_8[36] = {
    CMPLX(0.0, 1.0), 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, CMPLX(0.0, 1.0), 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, CMPLX(0.0, 1.0), 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, CMPLX(0.0, 1.0), 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, CMPLX(0.0, 1.0), 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, CMPLX(0.0, 1.0)
};
static const c128 zvx_W_8[6] = {CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(0.0, 1.0)};
static const f64 zvx_rcdein_8[6] = {1.1921e-07, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 1.1921e-07};
static const f64 zvx_rcdvin_8[6] = {0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00};

static const c128 zvx_A_9[36] = {
    CMPLX(0.0, 1.0), 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, CMPLX(0.0, 1.0), 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, CMPLX(0.0, 1.0), 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, CMPLX(0.0, 1.0), 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, CMPLX(0.0, 1.0), 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, CMPLX(0.0, 1.0)
};
static const c128 zvx_W_9[6] = {CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(0.0, 1.0)};
static const f64 zvx_rcdein_9[6] = {1.1921e-07, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 1.1921e-07};
static const f64 zvx_rcdvin_9[6] = {0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00};

static const c128 zvx_A_10[16] = {
    CMPLX(9.4480e-01, 1.0), CMPLX(6.7670e-01, 1.0), CMPLX(6.9080e-01, 1.0), CMPLX(5.9650e-01, 1.0),
    CMPLX(5.8760e-01, 1.0), CMPLX(8.6420e-01, 1.0), CMPLX(6.7690e-01, 1.0), CMPLX(7.2600e-02, 1.0),
    CMPLX(7.2560e-01, 1.0), CMPLX(1.9430e-01, 1.0), CMPLX(9.6870e-01, 1.0), CMPLX(2.8310e-01, 1.0),
    CMPLX(2.8490e-01, 1.0), CMPLX(5.8000e-02, 1.0), CMPLX(4.8450e-01, 1.0), CMPLX(7.3610e-01, 1.0)
};
static const c128 zvx_W_10[4] = {CMPLX(2.6014e-01, -1.7813e-01), CMPLX(2.8961e-01, 2.0772e-01), CMPLX(7.3990e-01, -4.6522e-04), CMPLX(2.2242e+00, 3.9709e+00)};
static const f64 zvx_rcdein_10[4] = {8.5279e-01, 8.4871e-01, 9.7398e-01, 9.8325e-01};
static const f64 zvx_rcdvin_10[4] = {3.2881e-01, 3.2358e-01, 3.4994e-01, 4.1429e+00};

static const c128 zvx_A_11[16] = {
    CMPLX(2.1130e-01, 9.9330e-01), CMPLX(8.0960e-01, 4.2370e-01), CMPLX(4.8320e-01, 1.1670e-01), CMPLX(6.5380e-01, 4.9430e-01),
    CMPLX(8.2400e-02, 8.3600e-01), CMPLX(8.4740e-01, 2.6130e-01), CMPLX(6.1350e-01, 6.2500e-01), CMPLX(4.8990e-01, 3.6500e-02),
    CMPLX(7.5990e-01, 7.4690e-01), CMPLX(4.5240e-01, 2.4030e-01), CMPLX(2.7490e-01, 5.5100e-01), CMPLX(7.7410e-01, 2.2600e-01),
    CMPLX(8.7000e-03, 3.7800e-02), CMPLX(8.0750e-01, 3.4050e-01), CMPLX(8.8070e-01, 3.5500e-01), CMPLX(9.6260e-01, 8.1590e-01)
};
static const c128 zvx_W_11[4] = {CMPLX(-6.2157e-01, 6.0607e-01), CMPLX(2.8890e-01, -2.6354e-01), CMPLX(3.8017e-01, 5.4217e-01), CMPLX(2.2487e+00, 1.7368e+00)};
static const f64 zvx_rcdein_11[4] = {8.7533e-01, 8.2538e-01, 7.4771e-01, 9.2372e-01};
static const f64 zvx_rcdvin_11[4] = {8.1980e-01, 8.1086e-01, 7.0323e-01, 2.2178e+00};

static const c128 zvx_A_12[9] = {
    CMPLX( 1.0,  2.0), CMPLX( 3.0,  4.0), CMPLX(21.0, 22.0),
    CMPLX(43.0, 44.0), CMPLX(13.0, 14.0), CMPLX(15.0, 16.0),
    CMPLX( 5.0,  6.0), CMPLX( 7.0,  8.0), CMPLX(25.0, 26.0)
};
static const c128 zvx_W_12[3] = {CMPLX(-7.4775e+00, 6.8803e+00), CMPLX(6.7009e+00, -7.8760e+00), CMPLX(3.9777e+01, 4.2996e+01)};
static const f64 zvx_rcdein_12[3] = {3.9550e-01, 3.9828e-01, 7.9686e-01};
static const f64 zvx_rcdvin_12[3] = {1.6583e+01, 1.6312e+01, 3.7399e+01};

static const c128 zvx_A_13[16] = {
    CMPLX(5.0, 9.0), CMPLX(5.0,  5.0), CMPLX(-6.0, -6.0), CMPLX(-7.0, -7.0),
    CMPLX(3.0, 3.0), CMPLX(6.0, 10.0), CMPLX(-5.0, -5.0), CMPLX(-6.0, -6.0),
    CMPLX(2.0, 2.0), CMPLX(3.0,  3.0), CMPLX(-1.0,  3.0), CMPLX(-5.0, -5.0),
    CMPLX(1.0, 1.0), CMPLX(2.0,  2.0), CMPLX(-3.0, -3.0), CMPLX( 0.0,  4.0)
};
static const c128 zvx_W_13[4] = {CMPLX(1.0, 5.0), CMPLX(2.0, 6.0), CMPLX(3.0, 7.0), CMPLX(4.0, 8.0)};
static const f64 zvx_rcdein_13[4] = {2.1822e-01, 2.1822e-01, 2.1822e-01, 2.1822e-01};
static const f64 zvx_rcdvin_13[4] = {7.4651e-01, 3.0893e-01, 1.8315e-01, 6.6350e-01};

static const c128 zvx_A_14[16] = {
    3.0, 1.0, 0.0, CMPLX(0.0, 2.0),
    1.0, 3.0, CMPLX(0.0, -2.0), 0.0,
    0.0, CMPLX(0.0, 2.0), 1.0, 1.0,
    CMPLX(0.0, -2.0), 0.0, 1.0, 1.0
};
static const c128 zvx_W_14[4] = {CMPLX(-8.2843e-01, 1.6979e-07), CMPLX(4.1744e-07, 7.1526e-08), CMPLX(4.0, 1.6690e-07), CMPLX(4.8284e+00, 6.8633e-08)};
static const f64 zvx_rcdein_14[4] = {1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_14[4] = {8.2843e-01, 8.2843e-01, 8.2843e-01, 8.2843e-01};

static const c128 zvx_A_15[16] = {
    7.0, 3.0, CMPLX(1.0, 2.0), CMPLX(-1.0, 2.0),
    3.0, 7.0, CMPLX(1.0, -2.0), CMPLX(-1.0, -2.0),
    CMPLX(1.0, -2.0), CMPLX(1.0, 2.0), 7.0, -3.0,
    CMPLX(-1.0, -2.0), CMPLX(-2.0, 2.0), -3.0, 7.0
};
static const c128 zvx_W_15[4] = {CMPLX(-8.0767e-03, -2.5211e-01), CMPLX(7.7723e+00, 2.4349e-01), CMPLX(8.0, -3.4273e-07), CMPLX(1.2236e+01, 8.6188e-03)};
static const f64 zvx_rcdein_15[4] = {9.9864e-01, 7.0272e-01, 7.0711e-01, 9.9021e-01};
static const f64 zvx_rcdvin_15[4] = {7.7961e+00, 3.3337e-01, 3.3337e-01, 3.9429e+00};

static const c128 zvx_A_16[25] = {
    CMPLX(1.0, 2.0), CMPLX(3.0, 4.0), CMPLX(21.0, 22.0), CMPLX(23.0, 24.0), CMPLX(41.0, 42.0),
    CMPLX(43.0, 44.0), CMPLX(13.0, 14.0), CMPLX(15.0, 16.0), CMPLX(33.0, 34.0), CMPLX(35.0, 36.0),
    CMPLX(5.0, 6.0), CMPLX(7.0, 8.0), CMPLX(25.0, 26.0), CMPLX(27.0, 28.0), CMPLX(45.0, 46.0),
    CMPLX(47.0, 48.0), CMPLX(17.0, 18.0), CMPLX(19.0, 20.0), CMPLX(37.0, 38.0), CMPLX(39.0, 40.0),
    CMPLX(9.0, 10.0), CMPLX(11.0, 12.0), CMPLX(29.0, 30.0), CMPLX(31.0, 32.0), CMPLX(49.0, 50.0)
};
static const c128 zvx_W_16[5] = {CMPLX(-9.4600e+00, 7.2802e+00), CMPLX(-7.7912e-06, -1.2743e-05), CMPLX(-7.3042e-06, 3.2789e-06), CMPLX(7.0733e+00, -9.5584e+00), CMPLX(1.2739e+02, 1.3228e+02)};
static const f64 zvx_rcdein_16[5] = {3.1053e-01, 2.9408e-01, 7.2259e-01, 3.0911e-01, 9.2770e-01};
static const f64 zvx_rcdvin_16[5] = {1.1937e+01, 1.6030e-05, 6.7794e-06, 1.1891e+01, 1.2111e+02};

static const c128 zvx_A_17[9] = {
    CMPLX(1.0, 1.0), CMPLX(-1.0, -1.0), CMPLX(2.0, 2.0),
                0.0, CMPLX( 0.0,  1.0),             2.0,
                0.0,              -1.0, CMPLX(3.0, 1.0)
};
static const c128 zvx_W_17[3] = {CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(2.0, 1.0)};
static const f64 zvx_rcdein_17[3] = {3.0151e-01, 3.1623e-01, 2.2361e-01};
static const f64 zvx_rcdvin_17[3] = {0.0000e+00, 0.0000e+00, 1.0000e+00};

static const c128 zvx_A_18[16] = {
    CMPLX(-4.0, -2.0), CMPLX(-5.0, -6.0), CMPLX(-2.0, -6.0), CMPLX(0.0, -2.0),
                  1.0,               0.0,               0.0,              0.0,
                  0.0,               1.0,               0.0,              0.0,
                  0.0,               0.0,               1.0,              0.0
};
static const c128 zvx_W_18[4] = {CMPLX(-9.9883e-01, -1.0006e+00), CMPLX(-1.0012e+00, -9.9945e-01), CMPLX(-9.9947e-01, -6.8325e-04), CMPLX(-1.0005e+00, 6.8556e-04)};
static const f64 zvx_rcdein_18[4] = {1.3180e-04, 1.3140e-04, 1.3989e-04, 1.4010e-04};
static const f64 zvx_rcdvin_18[4] = {2.4106e-04, 2.4041e-04, 8.7487e-05, 8.7750e-05};

static const c128 zvx_A_19[49] = {
    CMPLX(2.0, 4.0), CMPLX(1.0, 1.0), CMPLX(6.0, 2.0), CMPLX(3.0, 3.0), CMPLX(5.0, 5.0), CMPLX(2.0, 6.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 2.0), CMPLX(1.0, 3.0), CMPLX(3.0, 1.0), CMPLX(5.0, -4.0), CMPLX(1.0, 1.0), CMPLX(7.0, 2.0), CMPLX(2.0, 3.0),
                0.0, CMPLX(3.0, -2.0), CMPLX(1.0, 1.0), CMPLX(6.0, 3.0), CMPLX(2.0, 1.0), CMPLX(1.0, 4.0), CMPLX(2.0, 1.0),
                0.0,              0.0, CMPLX(2.0, 3.0), CMPLX(3.0, 1.0), CMPLX(1.0, 2.0), CMPLX(2.0, 2.0), CMPLX(3.0, 1.0),
                0.0,              0.0,             0.0, CMPLX(2.0, -1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 1.0), CMPLX(1.0, 3.0),
                0.0,              0.0,             0.0,              0.0, CMPLX(1.0, -1.0), CMPLX(2.0, 1.0), CMPLX(2.0, 2.0),
                0.0,              0.0,             0.0,              0.0,              0.0, CMPLX(2.0, -2.0), CMPLX(1.0, 1.0)
};
static const c128 zvx_W_19[7] = {CMPLX(-2.7081e+00, -2.8029e+00), CMPLX(-1.1478e+00, 8.0176e-01), CMPLX(-8.0109e-01, 4.9694e+00), CMPLX(9.9492e-01, 3.1688e+00), CMPLX(2.0809e+00, 1.9341e+00), CMPLX(5.3138e+00, 1.2242e+00), CMPLX(8.2674e+00, 3.7047e+00)};
static const f64 zvx_rcdein_19[7] = {6.9734e-01, 6.5772e-01, 4.6751e-01, 3.5095e-01, 4.9042e-01, 3.0213e-01, 2.8270e-01};
static const f64 zvx_rcdvin_19[7] = {3.9279e+00, 9.4243e-01, 1.3779e+00, 5.9845e-01, 3.9035e-01, 7.1268e-01, 3.2849e+00};

static const c128 zvx_A_20[25] = {
    CMPLX(0.0, 5.0), CMPLX(1.0, 2.0), CMPLX(2.0, 3.0), CMPLX(-3.0, 6.0), 6.0,
    CMPLX(-1.0, 2.0), CMPLX(0.0, 6.0), CMPLX(4.0, 5.0), CMPLX(-3.0, -2.0), 5.0,
    CMPLX(-2.0, 3.0), CMPLX(-4.0, 5.0), CMPLX(0.0, 7.0), 3.0, 2.0,
    CMPLX(3.0, 6.0), CMPLX(3.0, -2.0), -3.0, CMPLX(0.0, -5.0), CMPLX(2.0, 1.0),
    -6.0, -5.0, -2.0, CMPLX(-2.0, 1.0), CMPLX(0.0, 2.0)
};
static const c128 zvx_W_20[5] = {CMPLX(-4.1735e-08, -1.0734e+01), CMPLX(-2.6397e-07, -2.9991e+00), CMPLX(1.4565e-07, 1.5998e+00), CMPLX(-4.4369e-07, 9.3159e+00), CMPLX(4.0937e-09, 1.7817e+01)};
static const f64 zvx_rcdein_20[5] = {1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_20[5] = {7.7345e+00, 4.5989e+00, 4.5989e+00, 7.7161e+00, 8.5013e+00};

static const c128 zvx_A_21[9] = {
    2.0, CMPLX(0.0, -1.0), 0.0,
    CMPLX(0.0, 1.0), 2.0, 0.0,
    0.0, 0.0, 3.0
};
static const c128 zvx_W_21[3] = {CMPLX(1.0, 0.0), CMPLX(3.0, 0.0), CMPLX(3.0, 0.0)};
static const f64 zvx_rcdein_21[3] = {1.0000e+00, 1.0000e+00, 1.0000e+00};
static const f64 zvx_rcdvin_21[3] = {2.0000e+00, 0.0000e+00, 0.0000e+00};

#define ZVX_NUM_PRECOMPUTED 22

static const zvx_precomputed_t ZVX_PRECOMPUTED[ZVX_NUM_PRECOMPUTED] = {
    {1, 0, zvx_A_0, zvx_W_0, zvx_rcdein_0, zvx_rcdvin_0},
    {1, 0, zvx_A_1, zvx_W_1, zvx_rcdein_1, zvx_rcdvin_1},
    {2, 0, zvx_A_2, zvx_W_2, zvx_rcdein_2, zvx_rcdvin_2},
    {2, 0, zvx_A_3, zvx_W_3, zvx_rcdein_3, zvx_rcdvin_3},
    {2, 0, zvx_A_4, zvx_W_4, zvx_rcdein_4, zvx_rcdvin_4},
    {5, 0, zvx_A_5, zvx_W_5, zvx_rcdein_5, zvx_rcdvin_5},
    {5, 0, zvx_A_6, zvx_W_6, zvx_rcdein_6, zvx_rcdvin_6},
    {5, 0, zvx_A_7, zvx_W_7, zvx_rcdein_7, zvx_rcdvin_7},
    {6, 0, zvx_A_8, zvx_W_8, zvx_rcdein_8, zvx_rcdvin_8},
    {6, 0, zvx_A_9, zvx_W_9, zvx_rcdein_9, zvx_rcdvin_9},
    {4, 0, zvx_A_10, zvx_W_10, zvx_rcdein_10, zvx_rcdvin_10},
    {4, 0, zvx_A_11, zvx_W_11, zvx_rcdein_11, zvx_rcdvin_11},
    {3, 0, zvx_A_12, zvx_W_12, zvx_rcdein_12, zvx_rcdvin_12},
    {4, 0, zvx_A_13, zvx_W_13, zvx_rcdein_13, zvx_rcdvin_13},
    {4, 0, zvx_A_14, zvx_W_14, zvx_rcdein_14, zvx_rcdvin_14},
    {4, 0, zvx_A_15, zvx_W_15, zvx_rcdein_15, zvx_rcdvin_15},
    {5, 0, zvx_A_16, zvx_W_16, zvx_rcdein_16, zvx_rcdvin_16},
    {3, 0, zvx_A_17, zvx_W_17, zvx_rcdein_17, zvx_rcdvin_17},
    {4, 1, zvx_A_18, zvx_W_18, zvx_rcdein_18, zvx_rcdvin_18},
    {7, 0, zvx_A_19, zvx_W_19, zvx_rcdein_19, zvx_rcdvin_19},
    {5, 1, zvx_A_20, zvx_W_20, zvx_rcdein_20, zvx_rcdvin_20},
    {3, 0, zvx_A_21, zvx_W_21, zvx_rcdein_21, zvx_rcdvin_21}
};

#endif /* ZVX_TESTDATA_H */
