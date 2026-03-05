/**
 * @file zsx_testdata.h
 * @brief Precomputed test matrices for ZGEESX (from LAPACK TESTING/zed.in)
 *
 * 20 matrices with precomputed reciprocal condition numbers.
 * Used by test_zdrvsx.c for tests 16-17 (condition number accuracy).
 */

#ifndef ZSX_TESTDATA_H
#define ZSX_TESTDATA_H

static const c128 zsx_A_0[1] = {CMPLX(0.0, 0.0)};
static const INT zsx_islct_0[1] = {0};

static const c128 zsx_A_1[1] = {CMPLX(1.0, 0.0)};
static const INT zsx_islct_1[1] = {0};

static const c128 zsx_A_2[25] = {
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
};
static const INT zsx_islct_2[3] = {1, 2, 3};

static const c128 zsx_A_3[25] = {
    CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
};
static const INT zsx_islct_3[3] = {0, 2, 4};

static const c128 zsx_A_4[25] = {
    CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(2.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(5.0, 0.0)
};
static const INT zsx_islct_4[2] = {1, 3};

static const c128 zsx_A_5[36] = {
    CMPLX(0.0, 1.0), 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, CMPLX(0.0, 1.0), 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, CMPLX(0.0, 1.0), 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, CMPLX(0.0, 1.0), 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, CMPLX(0.0, 1.0), 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, CMPLX(0.0, 1.0)
};
static const INT zsx_islct_5[3] = {2, 3, 5};

static const c128 zsx_A_6[36] = {
    CMPLX(0.0, 1.0), 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, CMPLX(0.0, 1.0), 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, CMPLX(0.0, 1.0), 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, CMPLX(0.0, 1.0), 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, CMPLX(0.0, 1.0), 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, CMPLX(0.0, 1.0)
};
static const INT zsx_islct_6[3] = {0, 2, 4};

static const c128 zsx_A_7[16] = {
    CMPLX(9.4480e-01, 1.0), CMPLX(6.7670e-01, 1.0), CMPLX(6.9080e-01, 1.0), CMPLX(5.9650e-01, 1.0),
    CMPLX(5.8760e-01, 1.0), CMPLX(8.6420e-01, 1.0), CMPLX(6.7690e-01, 1.0), CMPLX(7.2600e-02, 1.0),
    CMPLX(7.2560e-01, 1.0), CMPLX(1.9430e-01, 1.0), CMPLX(9.6870e-01, 1.0), CMPLX(2.8310e-01, 1.0),
    CMPLX(2.8490e-01, 1.0), CMPLX(5.8000e-02, 1.0), CMPLX(4.8450e-01, 1.0), CMPLX(7.3610e-01, 1.0)
};
static const INT zsx_islct_7[2] = {2, 3};

static const c128 zsx_A_8[16] = {
    CMPLX(2.1130e-01, 9.9330e-01), CMPLX(8.0960e-01, 4.2370e-01), CMPLX(4.8320e-01, 1.1670e-01), CMPLX(6.5380e-01, 4.9430e-01),
    CMPLX(8.2400e-02, 8.3600e-01), CMPLX(8.4740e-01, 2.6130e-01), CMPLX(6.1350e-01, 6.2500e-01), CMPLX(4.8990e-01, 3.6500e-02),
    CMPLX(7.5990e-01, 7.4690e-01), CMPLX(4.5240e-01, 2.4030e-01), CMPLX(2.7490e-01, 5.5100e-01), CMPLX(7.7410e-01, 2.2600e-01),
    CMPLX(8.7000e-03, 3.7800e-02), CMPLX(8.0750e-01, 3.4050e-01), CMPLX(8.8070e-01, 3.5500e-01), CMPLX(9.6260e-01, 8.1590e-01)
};
static const INT zsx_islct_8[2] = {1, 2};

static const c128 zsx_A_9[9] = {
    CMPLX( 1.0,  2.0), CMPLX( 3.0,  4.0), CMPLX(21.0, 22.0),
    CMPLX(43.0, 44.0), CMPLX(13.0, 14.0), CMPLX(15.0, 16.0),
    CMPLX( 5.0,  6.0), CMPLX( 7.0,  8.0), CMPLX(25.0, 26.0)
};
static const INT zsx_islct_9[2] = {1, 2};

static const c128 zsx_A_10[16] = {
    CMPLX(5.0, 9.0), CMPLX(5.0,  5.0), CMPLX(-6.0, -6.0), CMPLX(-7.0, -7.0),
    CMPLX(3.0, 3.0), CMPLX(6.0, 10.0), CMPLX(-5.0, -5.0), CMPLX(-6.0, -6.0),
    CMPLX(2.0, 2.0), CMPLX(3.0,  3.0), CMPLX(-1.0,  3.0), CMPLX(-5.0, -5.0),
    CMPLX(1.0, 1.0), CMPLX(2.0,  2.0), CMPLX(-3.0, -3.0), CMPLX( 0.0,  4.0)
};
static const INT zsx_islct_10[2] = {0, 2};

static const c128 zsx_A_11[16] = {
    3.0, 1.0, 0.0, CMPLX(0.0, 2.0),
    1.0, 3.0, CMPLX(0.0, -2.0), 0.0,
    0.0, CMPLX(0.0, 2.0), 1.0, 1.0,
    CMPLX(0.0, -2.0), 0.0, 1.0, 1.0
};
static const INT zsx_islct_11[3] = {0, 2, 3};

static const c128 zsx_A_12[16] = {
    7.0, 3.0, CMPLX(1.0, 2.0), CMPLX(-1.0, 2.0),
    3.0, 7.0, CMPLX(1.0, -2.0), CMPLX(-1.0, -2.0),
    CMPLX(1.0, -2.0), CMPLX(1.0, 2.0), 7.0, -3.0,
    CMPLX(-1.0, -2.0), CMPLX(-2.0, 2.0), -3.0, 7.0
};
static const INT zsx_islct_12[2] = {1, 2};

static const c128 zsx_A_13[25] = {
    CMPLX(1.0, 2.0), CMPLX(3.0, 4.0), CMPLX(21.0, 22.0), CMPLX(23.0, 24.0), CMPLX(41.0, 42.0),
    CMPLX(43.0, 44.0), CMPLX(13.0, 14.0), CMPLX(15.0, 16.0), CMPLX(33.0, 34.0), CMPLX(35.0, 36.0),
    CMPLX(5.0, 6.0), CMPLX(7.0, 8.0), CMPLX(25.0, 26.0), CMPLX(27.0, 28.0), CMPLX(45.0, 46.0),
    CMPLX(47.0, 48.0), CMPLX(17.0, 18.0), CMPLX(19.0, 20.0), CMPLX(37.0, 38.0), CMPLX(39.0, 40.0),
    CMPLX(9.0, 10.0), CMPLX(11.0, 12.0), CMPLX(29.0, 30.0), CMPLX(31.0, 32.0), CMPLX(49.0, 50.0)
};
static const INT zsx_islct_13[2] = {1, 2};

static const c128 zsx_A_14[9] = {
    CMPLX(1.0, 1.0), CMPLX(-1.0, -1.0), CMPLX(2.0, 2.0),
                0.0, CMPLX( 0.0,  1.0),             2.0,
                0.0,              -1.0, CMPLX(3.0, 1.0)
};
static const INT zsx_islct_14[2] = {0, 1};

static const c128 zsx_A_15[16] = {
    CMPLX(-4.0, -2.0), CMPLX(-5.0, -6.0), CMPLX(-2.0, -6.0), CMPLX(0.0, -2.0),
                  1.0,               0.0,               0.0,              0.0,
                  0.0,               1.0,               0.0,              0.0,
                  0.0,               0.0,               1.0,              0.0
};
static const INT zsx_islct_15[2] = {0, 2};

static const c128 zsx_A_16[49] = {
    CMPLX(2.0, 4.0), CMPLX(1.0, 1.0), CMPLX(6.0, 2.0), CMPLX(3.0, 3.0), CMPLX(5.0, 5.0), CMPLX(2.0, 6.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 2.0), CMPLX(1.0, 3.0), CMPLX(3.0, 1.0), CMPLX(5.0, -4.0), CMPLX(1.0, 1.0), CMPLX(7.0, 2.0), CMPLX(2.0, 3.0),
                0.0, CMPLX(3.0, -2.0), CMPLX(1.0, 1.0), CMPLX(6.0, 3.0), CMPLX(2.0, 1.0), CMPLX(1.0, 4.0), CMPLX(2.0, 1.0),
                0.0,              0.0, CMPLX(2.0, 3.0), CMPLX(3.0, 1.0), CMPLX(1.0, 2.0), CMPLX(2.0, 2.0), CMPLX(3.0, 1.0),
                0.0,              0.0,             0.0, CMPLX(2.0, -1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 1.0), CMPLX(1.0, 3.0),
                0.0,              0.0,             0.0,              0.0, CMPLX(1.0, -1.0), CMPLX(2.0, 1.0), CMPLX(2.0, 2.0),
                0.0,              0.0,             0.0,              0.0,              0.0, CMPLX(2.0, -2.0), CMPLX(1.0, 1.0)
};
static const INT zsx_islct_16[4] = {0, 3, 5, 6};

static const c128 zsx_A_17[25] = {
    CMPLX(0.0, 5.0), CMPLX(1.0, 2.0), CMPLX(2.0, 3.0), CMPLX(-3.0, 6.0), 6.0,
    CMPLX(-1.0, 2.0), CMPLX(0.0, 6.0), CMPLX(4.0, 5.0), CMPLX(-3.0, -2.0), 5.0,
    CMPLX(-2.0, 3.0), CMPLX(-4.0, 5.0), CMPLX(0.0, 7.0), 3.0, 2.0,
    CMPLX(3.0, 6.0), CMPLX(3.0, -2.0), -3.0, CMPLX(0.0, -5.0), CMPLX(2.0, 1.0),
    -6.0, -5.0, -2.0, CMPLX(-2.0, 1.0), CMPLX(0.0, 2.0)
};
static const INT zsx_islct_17[3] = {0, 2, 4};

static const c128 zsx_A_18[64] = {
    CMPLX(0.0, 1.0), 1.0, 0.0, 0.0, CMPLX(0.0, 1.0), 1.0, CMPLX(0.0, 1.0), 1.0,
    0.0, CMPLX(0.0, 1.0), 1.0, 0.0, CMPLX(0.0, 2.0), 2.0, CMPLX(0.0, 2.0), 2.0,
    0.0, 0.0, CMPLX(0.0, 1.0), 1.0, CMPLX(0.0, 3.0), 3.0, CMPLX(0.0, 3.0), 3.0,
    0.0, 0.0, 0.0, CMPLX(0.0, 1.0), CMPLX(0.0, 4.0), 4.0, CMPLX(0.0, 4.0), 4.0,
    0.0, 0.0, 0.0, 0.0, CMPLX(0.0, 9.5000e-01), 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, CMPLX(0.0, 9.5000e-01), 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, CMPLX(0.0, 9.5000e-01), 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, CMPLX(0.0, 9.5000e-01)
};
static const INT zsx_islct_18[4] = {0, 1, 2, 3};

static const c128 zsx_A_19[9] = {
    2.0, CMPLX(0.0, -1.0), 0.0,
    CMPLX(0.0, 1.0), 2.0, 0.0,
    0.0, 0.0, 3.0
};
static const INT zsx_islct_19[2] = {1, 2};

typedef struct {
    int n;
    int isrt;
    int nslct;
    const INT* islct;
    const c128* A;
    f64 rcdein;
    f64 rcdvin;
} zsx_precomputed_t;

#define ZSX_NUM_PRECOMPUTED 20

static const zsx_precomputed_t ZSX_PRECOMPUTED[ZSX_NUM_PRECOMPUTED] = {
    {1, 0, 1, zsx_islct_0, zsx_A_0, 1.0000e+00, 0.0000e+00},
    {1, 0, 1, zsx_islct_1, zsx_A_1, 1.0000e+00, 1.0000e+00},
    {5, 0, 3, zsx_islct_2, zsx_A_2, 1.0000e+00, 2.9582e-31},
    {5, 0, 3, zsx_islct_3, zsx_A_3, 1.0000e+00, 1.0000e+00},
    {5, 0, 2, zsx_islct_4, zsx_A_4, 1.0000e+00, 1.0000e+00},
    {6, 1, 3, zsx_islct_5, zsx_A_5, 1.0000e+00, 2.0000e+00},
    {6, 0, 3, zsx_islct_6, zsx_A_6, 1.0000e+00, 2.0000e+00},
    {4, 0, 2, zsx_islct_7, zsx_A_7, 9.6350e-01, 3.3122e-01},
    {4, 0, 2, zsx_islct_8, zsx_A_8, 8.4053e-01, 7.4754e-01},
    {3, 0, 2, zsx_islct_9, zsx_A_9, 3.9550e-01, 2.0464e+01},
    {4, 0, 2, zsx_islct_10, zsx_A_10, 3.3333e-01, 1.2569e-01},
    {4, 0, 3, zsx_islct_11, zsx_A_11, 1.0000e+00, 8.2843e-01},
    {4, 0, 2, zsx_islct_12, zsx_A_12, 9.8985e-01, 4.1447e+00},
    {5, 1, 2, zsx_islct_13, zsx_A_13, 3.1088e-01, 4.6912e+00},
    {3, 0, 2, zsx_islct_14, zsx_A_14, 2.2361e-01, 1.0000e+00},
    {4, 1, 2, zsx_islct_15, zsx_A_15, 7.2803e-05, 1.1947e-04},
    {7, 0, 4, zsx_islct_16, zsx_A_16, 3.7241e-01, 5.2080e-01},
    {5, 1, 3, zsx_islct_17, zsx_A_17, 1.0000e+00, 4.5989e+00},
    {8, 1, 4, zsx_islct_18, zsx_A_18, 9.5269e-12, 2.9360e-11},
    {3, 0, 2, zsx_islct_19, zsx_A_19, 1.0000e+00, 2.0000e+00}
};

#endif /* ZSX_TESTDATA_H */
