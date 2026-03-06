/**
 * @file cvx_testdata.h
 * @brief Precomputed test matrices for CGEEVX (from LAPACK TESTING/zed.in)
 *
 * 22 matrices with precomputed eigenvalues and reciprocal condition numbers.
 * Used by test_zdrvvx.c for tests 10-11 (condition number accuracy).
 */

#ifndef CVX_TESTDATA_H
#define CVX_TESTDATA_H

typedef struct {
    int n;
    int isrt;
    const c64* A;
    const c64* W;
    const f32* rcdein;
    const f32* rcdvin;
} zvx_precomputed_t;

static const c64 cvx_A_0[1] = {CMPLXF(0.0f, 0.0f)};
static const c64 cvx_W_0[1] = {CMPLXF(0.0f, 0.0f)};
static const f32 cvx_rcdein_0[1] = {1.0000e+00f};
static const f32 cvx_rcdvin_0[1] = {0.0000e+00f};

static const c64 cvx_A_1[1] = {CMPLXF(0.0f, 1.0f)};
static const c64 cvx_W_1[1] = {CMPLXF(0.0f, 1.0f)};
static const f32 cvx_rcdein_1[1] = {1.0000e+00f};
static const f32 cvx_rcdvin_1[1] = {1.0000e+00f};

static const c64 cvx_A_2[4] = {
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
};
static const c64 cvx_W_2[2] = {CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)};
static const f32 cvx_rcdein_2[2] = {1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_2[2] = {0.0000e+00f, 0.0000e+00f};

static const c64 cvx_A_3[4] = {
    CMPLXF(3.0f, 0.0f), CMPLXF(2.0f, 0.0f),
    CMPLXF(2.0f, 0.0f), CMPLXF(3.0f, 0.0f)
};
static const c64 cvx_W_3[2] = {CMPLXF(1.0f, 0.0f), CMPLXF(5.0f, 0.0f)};
static const f32 cvx_rcdein_3[2] = {1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_3[2] = {4.0000e+00f, 4.0000e+00f};

static const c64 cvx_A_4[4] = {
    3.0f, CMPLXF(0.0f, 2.0f),
    CMPLXF(0.0f, 2.0f), 3.0f
};
static const c64 cvx_W_4[2] = {CMPLXF(3.0f, 2.0f), CMPLXF(3.0f, -2.0f)};
static const f32 cvx_rcdein_4[2] = {1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_4[2] = {4.0000e+00f, 4.0000e+00f};

static const c64 cvx_A_5[25] = {
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
};
static const c64 cvx_W_5[5] = {CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)};
static const f32 cvx_rcdein_5[5] = {1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_5[5] = {0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f};

static const c64 cvx_A_6[25] = {
    CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
};
static const c64 cvx_W_6[5] = {CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f)};
static const f32 cvx_rcdein_6[5] = {1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_6[5] = {0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f};

static const c64 cvx_A_7[25] = {
    CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 0.0f)
};
static const c64 cvx_W_7[5] = {CMPLXF(1.0f, 0.0f), CMPLXF(2.0f, 0.0f), CMPLXF(3.0f, 0.0f), CMPLXF(4.0f, 0.0f), CMPLXF(5.0f, 0.0f)};
static const f32 cvx_rcdein_7[5] = {1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_7[5] = {1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f};

static const c64 cvx_A_8[36] = {
    CMPLXF(0.0f, 1.0f), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, CMPLXF(0.0f, 1.0f), 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, CMPLXF(0.0f, 1.0f), 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 1.0f), 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 1.0f), 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 1.0f)
};
static const c64 cvx_W_8[6] = {CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f)};
static const f32 cvx_rcdein_8[6] = {1.1921e-07f, 2.4074e-35f, 2.4074e-35f, 2.4074e-35f, 2.4074e-35f, 1.1921e-07f};
static const f32 cvx_rcdvin_8[6] = {0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f};

static const c64 cvx_A_9[36] = {
    CMPLXF(0.0f, 1.0f), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, CMPLXF(0.0f, 1.0f), 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, CMPLXF(0.0f, 1.0f), 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, CMPLXF(0.0f, 1.0f), 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, CMPLXF(0.0f, 1.0f), 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, CMPLXF(0.0f, 1.0f)
};
static const c64 cvx_W_9[6] = {CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f)};
static const f32 cvx_rcdein_9[6] = {1.1921e-07f, 2.4074e-35f, 2.4074e-35f, 2.4074e-35f, 2.4074e-35f, 1.1921e-07f};
static const f32 cvx_rcdvin_9[6] = {0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f};

static const c64 cvx_A_10[16] = {
    CMPLXF(9.4480e-01f, 1.0f), CMPLXF(6.7670e-01f, 1.0f), CMPLXF(6.9080e-01f, 1.0f), CMPLXF(5.9650e-01f, 1.0f),
    CMPLXF(5.8760e-01f, 1.0f), CMPLXF(8.6420e-01f, 1.0f), CMPLXF(6.7690e-01f, 1.0f), CMPLXF(7.2600e-02f, 1.0f),
    CMPLXF(7.2560e-01f, 1.0f), CMPLXF(1.9430e-01f, 1.0f), CMPLXF(9.6870e-01f, 1.0f), CMPLXF(2.8310e-01f, 1.0f),
    CMPLXF(2.8490e-01f, 1.0f), CMPLXF(5.8000e-02f, 1.0f), CMPLXF(4.8450e-01f, 1.0f), CMPLXF(7.3610e-01f, 1.0f)
};
static const c64 cvx_W_10[4] = {CMPLXF(2.6014e-01f, -1.7813e-01f), CMPLXF(2.8961e-01f, 2.0772e-01f), CMPLXF(7.3990e-01f, -4.6522e-04f), CMPLXF(2.2242e+00f, 3.9709e+00f)};
static const f32 cvx_rcdein_10[4] = {8.5279e-01f, 8.4871e-01f, 9.7398e-01f, 9.8325e-01f};
static const f32 cvx_rcdvin_10[4] = {3.2881e-01f, 3.2358e-01f, 3.4994e-01f, 4.1429e+00f};

static const c64 cvx_A_11[16] = {
    CMPLXF(2.1130e-01f, 9.9330e-01f), CMPLXF(8.0960e-01f, 4.2370e-01f), CMPLXF(4.8320e-01f, 1.1670e-01f), CMPLXF(6.5380e-01f, 4.9430e-01f),
    CMPLXF(8.2400e-02f, 8.3600e-01f), CMPLXF(8.4740e-01f, 2.6130e-01f), CMPLXF(6.1350e-01f, 6.2500e-01f), CMPLXF(4.8990e-01f, 3.6500e-02f),
    CMPLXF(7.5990e-01f, 7.4690e-01f), CMPLXF(4.5240e-01f, 2.4030e-01f), CMPLXF(2.7490e-01f, 5.5100e-01f), CMPLXF(7.7410e-01f, 2.2600e-01f),
    CMPLXF(8.7000e-03f, 3.7800e-02f), CMPLXF(8.0750e-01f, 3.4050e-01f), CMPLXF(8.8070e-01f, 3.5500e-01f), CMPLXF(9.6260e-01f, 8.1590e-01f)
};
static const c64 cvx_W_11[4] = {CMPLXF(-6.2157e-01f, 6.0607e-01f), CMPLXF(2.8890e-01f, -2.6354e-01f), CMPLXF(3.8017e-01f, 5.4217e-01f), CMPLXF(2.2487e+00f, 1.7368e+00f)};
static const f32 cvx_rcdein_11[4] = {8.7533e-01f, 8.2538e-01f, 7.4771e-01f, 9.2372e-01f};
static const f32 cvx_rcdvin_11[4] = {8.1980e-01f, 8.1086e-01f, 7.0323e-01f, 2.2178e+00f};

static const c64 cvx_A_12[9] = {
    CMPLXF( 1.0f,  2.0f), CMPLXF( 3.0f,  4.0f), CMPLXF(21.0f, 22.0f),
    CMPLXF(43.0f, 44.0f), CMPLXF(13.0f, 14.0f), CMPLXF(15.0f, 16.0f),
    CMPLXF( 5.0f,  6.0f), CMPLXF( 7.0f,  8.0f), CMPLXF(25.0f, 26.0f)
};
static const c64 cvx_W_12[3] = {CMPLXF(-7.4775e+00f, 6.8803e+00f), CMPLXF(6.7009e+00f, -7.8760e+00f), CMPLXF(3.9777e+01f, 4.2996e+01f)};
static const f32 cvx_rcdein_12[3] = {3.9550e-01f, 3.9828e-01f, 7.9686e-01f};
static const f32 cvx_rcdvin_12[3] = {1.6583e+01f, 1.6312e+01f, 3.7399e+01f};

static const c64 cvx_A_13[16] = {
    CMPLXF(5.0f, 9.0f), CMPLXF(5.0f,  5.0f), CMPLXF(-6.0f, -6.0f), CMPLXF(-7.0f, -7.0f),
    CMPLXF(3.0f, 3.0f), CMPLXF(6.0f, 10.0f), CMPLXF(-5.0f, -5.0f), CMPLXF(-6.0f, -6.0f),
    CMPLXF(2.0f, 2.0f), CMPLXF(3.0f,  3.0f), CMPLXF(-1.0f,  3.0f), CMPLXF(-5.0f, -5.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f,  2.0f), CMPLXF(-3.0f, -3.0f), CMPLXF( 0.0f,  4.0f)
};
static const c64 cvx_W_13[4] = {CMPLXF(1.0f, 5.0f), CMPLXF(2.0f, 6.0f), CMPLXF(3.0f, 7.0f), CMPLXF(4.0f, 8.0f)};
static const f32 cvx_rcdein_13[4] = {2.1822e-01f, 2.1822e-01f, 2.1822e-01f, 2.1822e-01f};
static const f32 cvx_rcdvin_13[4] = {7.4651e-01f, 3.0893e-01f, 1.8315e-01f, 6.6350e-01f};

static const c64 cvx_A_14[16] = {
    3.0f, 1.0f, 0.0f, CMPLXF(0.0f, 2.0f),
    1.0f, 3.0f, CMPLXF(0.0f, -2.0f), 0.0f,
    0.0f, CMPLXF(0.0f, 2.0f), 1.0f, 1.0f,
    CMPLXF(0.0f, -2.0f), 0.0f, 1.0f, 1.0f
};
static const c64 cvx_W_14[4] = {CMPLXF(-8.2843e-01f, 1.6979e-07f), CMPLXF(4.1744e-07f, 7.1526e-08f), CMPLXF(4.0f, 1.6690e-07f), CMPLXF(4.8284e+00f, 6.8633e-08f)};
static const f32 cvx_rcdein_14[4] = {1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_14[4] = {8.2843e-01f, 8.2843e-01f, 8.2843e-01f, 8.2843e-01f};

static const c64 cvx_A_15[16] = {
    7.0f, 3.0f, CMPLXF(1.0f, 2.0f), CMPLXF(-1.0f, 2.0f),
    3.0f, 7.0f, CMPLXF(1.0f, -2.0f), CMPLXF(-1.0f, -2.0f),
    CMPLXF(1.0f, -2.0f), CMPLXF(1.0f, 2.0f), 7.0f, -3.0f,
    CMPLXF(-1.0f, -2.0f), CMPLXF(-2.0f, 2.0f), -3.0f, 7.0f
};
static const c64 cvx_W_15[4] = {CMPLXF(-8.0767e-03f, -2.5211e-01f), CMPLXF(7.7723e+00f, 2.4349e-01f), CMPLXF(8.0f, -3.4273e-07f), CMPLXF(1.2236e+01f, 8.6188e-03f)};
static const f32 cvx_rcdein_15[4] = {9.9864e-01f, 7.0272e-01f, 7.0711e-01f, 9.9021e-01f};
static const f32 cvx_rcdvin_15[4] = {7.7961e+00f, 3.3337e-01f, 3.3337e-01f, 3.9429e+00f};

static const c64 cvx_A_16[25] = {
    CMPLXF(1.0f, 2.0f), CMPLXF(3.0f, 4.0f), CMPLXF(21.0f, 22.0f), CMPLXF(23.0f, 24.0f), CMPLXF(41.0f, 42.0f),
    CMPLXF(43.0f, 44.0f), CMPLXF(13.0f, 14.0f), CMPLXF(15.0f, 16.0f), CMPLXF(33.0f, 34.0f), CMPLXF(35.0f, 36.0f),
    CMPLXF(5.0f, 6.0f), CMPLXF(7.0f, 8.0f), CMPLXF(25.0f, 26.0f), CMPLXF(27.0f, 28.0f), CMPLXF(45.0f, 46.0f),
    CMPLXF(47.0f, 48.0f), CMPLXF(17.0f, 18.0f), CMPLXF(19.0f, 20.0f), CMPLXF(37.0f, 38.0f), CMPLXF(39.0f, 40.0f),
    CMPLXF(9.0f, 10.0f), CMPLXF(11.0f, 12.0f), CMPLXF(29.0f, 30.0f), CMPLXF(31.0f, 32.0f), CMPLXF(49.0f, 50.0f)
};
static const c64 cvx_W_16[5] = {CMPLXF(-9.4600e+00f, 7.2802e+00f), CMPLXF(-7.7912e-06f, -1.2743e-05f), CMPLXF(-7.3042e-06f, 3.2789e-06f), CMPLXF(7.0733e+00f, -9.5584e+00f), CMPLXF(1.2739e+02f, 1.3228e+02f)};
static const f32 cvx_rcdein_16[5] = {3.1053e-01f, 2.9408e-01f, 7.2259e-01f, 3.0911e-01f, 9.2770e-01f};
static const f32 cvx_rcdvin_16[5] = {1.1937e+01f, 1.6030e-05f, 6.7794e-06f, 1.1891e+01f, 1.2111e+02f};

static const c64 cvx_A_17[9] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(2.0f, 2.0f),
                0.0f, CMPLXF( 0.0f,  1.0f),             2.0f,
                0.0f,              -1.0f, CMPLXF(3.0f, 1.0f)
};
static const c64 cvx_W_17[3] = {CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 1.0f)};
static const f32 cvx_rcdein_17[3] = {3.0151e-01f, 3.1623e-01f, 2.2361e-01f};
static const f32 cvx_rcdvin_17[3] = {0.0000e+00f, 0.0000e+00f, 1.0000e+00f};

static const c64 cvx_A_18[16] = {
    CMPLXF(-4.0f, -2.0f), CMPLXF(-5.0f, -6.0f), CMPLXF(-2.0f, -6.0f), CMPLXF(0.0f, -2.0f),
                  1.0f,               0.0f,               0.0f,              0.0f,
                  0.0f,               1.0f,               0.0f,              0.0f,
                  0.0f,               0.0f,               1.0f,              0.0f
};
static const c64 cvx_W_18[4] = {CMPLXF(-9.9883e-01f, -1.0006e+00f), CMPLXF(-1.0012e+00f, -9.9945e-01f), CMPLXF(-9.9947e-01f, -6.8325e-04f), CMPLXF(-1.0005e+00f, 6.8556e-04f)};
static const f32 cvx_rcdein_18[4] = {1.3180e-04f, 1.3140e-04f, 1.3989e-04f, 1.4010e-04f};
static const f32 cvx_rcdvin_18[4] = {2.4106e-04f, 2.4041e-04f, 8.7487e-05f, 8.7750e-05f};

static const c64 cvx_A_19[49] = {
    CMPLXF(2.0f, 4.0f), CMPLXF(1.0f, 1.0f), CMPLXF(6.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(5.0f, 5.0f), CMPLXF(2.0f, 6.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 2.0f), CMPLXF(1.0f, 3.0f), CMPLXF(3.0f, 1.0f), CMPLXF(5.0f, -4.0f), CMPLXF(1.0f, 1.0f), CMPLXF(7.0f, 2.0f), CMPLXF(2.0f, 3.0f),
                0.0f, CMPLXF(3.0f, -2.0f), CMPLXF(1.0f, 1.0f), CMPLXF(6.0f, 3.0f), CMPLXF(2.0f, 1.0f), CMPLXF(1.0f, 4.0f), CMPLXF(2.0f, 1.0f),
                0.0f,              0.0f, CMPLXF(2.0f, 3.0f), CMPLXF(3.0f, 1.0f), CMPLXF(1.0f, 2.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 1.0f),
                0.0f,              0.0f,             0.0f, CMPLXF(2.0f, -1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 1.0f), CMPLXF(1.0f, 3.0f),
                0.0f,              0.0f,             0.0f,              0.0f, CMPLXF(1.0f, -1.0f), CMPLXF(2.0f, 1.0f), CMPLXF(2.0f, 2.0f),
                0.0f,              0.0f,             0.0f,              0.0f,              0.0f, CMPLXF(2.0f, -2.0f), CMPLXF(1.0f, 1.0f)
};
static const c64 cvx_W_19[7] = {CMPLXF(-2.7081e+00f, -2.8029e+00f), CMPLXF(-1.1478e+00f, 8.0176e-01f), CMPLXF(-8.0109e-01f, 4.9694e+00f), CMPLXF(9.9492e-01f, 3.1688e+00f), CMPLXF(2.0809e+00f, 1.9341e+00f), CMPLXF(5.3138e+00f, 1.2242e+00f), CMPLXF(8.2674e+00f, 3.7047e+00f)};
static const f32 cvx_rcdein_19[7] = {6.9734e-01f, 6.5772e-01f, 4.6751e-01f, 3.5095e-01f, 4.9042e-01f, 3.0213e-01f, 2.8270e-01f};
static const f32 cvx_rcdvin_19[7] = {3.9279e+00f, 9.4243e-01f, 1.3779e+00f, 5.9845e-01f, 3.9035e-01f, 7.1268e-01f, 3.2849e+00f};

static const c64 cvx_A_20[25] = {
    CMPLXF(0.0f, 5.0f), CMPLXF(1.0f, 2.0f), CMPLXF(2.0f, 3.0f), CMPLXF(-3.0f, 6.0f), 6.0f,
    CMPLXF(-1.0f, 2.0f), CMPLXF(0.0f, 6.0f), CMPLXF(4.0f, 5.0f), CMPLXF(-3.0f, -2.0f), 5.0f,
    CMPLXF(-2.0f, 3.0f), CMPLXF(-4.0f, 5.0f), CMPLXF(0.0f, 7.0f), 3.0f, 2.0f,
    CMPLXF(3.0f, 6.0f), CMPLXF(3.0f, -2.0f), -3.0f, CMPLXF(0.0f, -5.0f), CMPLXF(2.0f, 1.0f),
    -6.0f, -5.0f, -2.0f, CMPLXF(-2.0f, 1.0f), CMPLXF(0.0f, 2.0f)
};
static const c64 cvx_W_20[5] = {CMPLXF(-4.1735e-08f, -1.0734e+01f), CMPLXF(-2.6397e-07f, -2.9991e+00f), CMPLXF(1.4565e-07f, 1.5998e+00f), CMPLXF(-4.4369e-07f, 9.3159e+00f), CMPLXF(4.0937e-09f, 1.7817e+01f)};
static const f32 cvx_rcdein_20[5] = {1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_20[5] = {7.7345e+00f, 4.5989e+00f, 4.5989e+00f, 7.7161e+00f, 8.5013e+00f};

static const c64 cvx_A_21[9] = {
    2.0f, CMPLXF(0.0f, -1.0f), 0.0f,
    CMPLXF(0.0f, 1.0f), 2.0f, 0.0f,
    0.0f, 0.0f, 3.0f
};
static const c64 cvx_W_21[3] = {CMPLXF(1.0f, 0.0f), CMPLXF(3.0f, 0.0f), CMPLXF(3.0f, 0.0f)};
static const f32 cvx_rcdein_21[3] = {1.0000e+00f, 1.0000e+00f, 1.0000e+00f};
static const f32 cvx_rcdvin_21[3] = {2.0000e+00f, 0.0000e+00f, 0.0000e+00f};

#define ZVX_NUM_PRECOMPUTED 22

static const zvx_precomputed_t ZVX_PRECOMPUTED[ZVX_NUM_PRECOMPUTED] = {
    {1, 0, cvx_A_0, cvx_W_0, cvx_rcdein_0, cvx_rcdvin_0},
    {1, 0, cvx_A_1, cvx_W_1, cvx_rcdein_1, cvx_rcdvin_1},
    {2, 0, cvx_A_2, cvx_W_2, cvx_rcdein_2, cvx_rcdvin_2},
    {2, 0, cvx_A_3, cvx_W_3, cvx_rcdein_3, cvx_rcdvin_3},
    {2, 0, cvx_A_4, cvx_W_4, cvx_rcdein_4, cvx_rcdvin_4},
    {5, 0, cvx_A_5, cvx_W_5, cvx_rcdein_5, cvx_rcdvin_5},
    {5, 0, cvx_A_6, cvx_W_6, cvx_rcdein_6, cvx_rcdvin_6},
    {5, 0, cvx_A_7, cvx_W_7, cvx_rcdein_7, cvx_rcdvin_7},
    {6, 0, cvx_A_8, cvx_W_8, cvx_rcdein_8, cvx_rcdvin_8},
    {6, 0, cvx_A_9, cvx_W_9, cvx_rcdein_9, cvx_rcdvin_9},
    {4, 0, cvx_A_10, cvx_W_10, cvx_rcdein_10, cvx_rcdvin_10},
    {4, 0, cvx_A_11, cvx_W_11, cvx_rcdein_11, cvx_rcdvin_11},
    {3, 0, cvx_A_12, cvx_W_12, cvx_rcdein_12, cvx_rcdvin_12},
    {4, 0, cvx_A_13, cvx_W_13, cvx_rcdein_13, cvx_rcdvin_13},
    {4, 0, cvx_A_14, cvx_W_14, cvx_rcdein_14, cvx_rcdvin_14},
    {4, 0, cvx_A_15, cvx_W_15, cvx_rcdein_15, cvx_rcdvin_15},
    {5, 0, cvx_A_16, cvx_W_16, cvx_rcdein_16, cvx_rcdvin_16},
    {3, 0, cvx_A_17, cvx_W_17, cvx_rcdein_17, cvx_rcdvin_17},
    {4, 1, cvx_A_18, cvx_W_18, cvx_rcdein_18, cvx_rcdvin_18},
    {7, 0, cvx_A_19, cvx_W_19, cvx_rcdein_19, cvx_rcdvin_19},
    {5, 1, cvx_A_20, cvx_W_20, cvx_rcdein_20, cvx_rcdvin_20},
    {3, 0, cvx_A_21, cvx_W_21, cvx_rcdein_21, cvx_rcdvin_21}
};

#endif /* CVX_TESTDATA_H */
