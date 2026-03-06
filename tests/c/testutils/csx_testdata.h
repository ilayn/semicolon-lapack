/**
 * @file csx_testdata.h
 * @brief Precomputed test matrices for CGEESX (from LAPACK TESTING/zed.in)
 *
 * 20 matrices with precomputed reciprocal condition numbers.
 * Used by test_zdrvsx.c for tests 16-17 (condition number accuracy).
 */

#ifndef CSX_TESTDATA_H
#define CSX_TESTDATA_H

static const c64 csx_A_0[1] = {CMPLXF(0.0f, 0.0f)};
static const INT csx_islct_0[1] = {0};

static const c64 csx_A_1[1] = {CMPLXF(1.0f, 0.0f)};
static const INT csx_islct_1[1] = {0};

static const c64 csx_A_2[25] = {
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
};
static const INT csx_islct_2[3] = {1, 2, 3};

static const c64 csx_A_3[25] = {
    CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
};
static const INT csx_islct_3[3] = {0, 2, 4};

static const c64 csx_A_4[25] = {
    CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 0.0f)
};
static const INT csx_islct_4[2] = {1, 3};

static const c64 csx_A_5[36] = {
    CMPLXF(0.0f, 1.0f), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, CMPLXF(0.0f, 1.0f), 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, CMPLXF(0.0f, 1.0f), 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 1.0f), 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 1.0f), 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 1.0f)
};
static const INT csx_islct_5[3] = {2, 3, 5};

static const c64 csx_A_6[36] = {
    CMPLXF(0.0f, 1.0f), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, CMPLXF(0.0f, 1.0f), 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, CMPLXF(0.0f, 1.0f), 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, CMPLXF(0.0f, 1.0f), 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, CMPLXF(0.0f, 1.0f), 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, CMPLXF(0.0f, 1.0f)
};
static const INT csx_islct_6[3] = {0, 2, 4};

static const c64 csx_A_7[16] = {
    CMPLXF(9.4480e-01f, 1.0f), CMPLXF(6.7670e-01f, 1.0f), CMPLXF(6.9080e-01f, 1.0f), CMPLXF(5.9650e-01f, 1.0f),
    CMPLXF(5.8760e-01f, 1.0f), CMPLXF(8.6420e-01f, 1.0f), CMPLXF(6.7690e-01f, 1.0f), CMPLXF(7.2600e-02f, 1.0f),
    CMPLXF(7.2560e-01f, 1.0f), CMPLXF(1.9430e-01f, 1.0f), CMPLXF(9.6870e-01f, 1.0f), CMPLXF(2.8310e-01f, 1.0f),
    CMPLXF(2.8490e-01f, 1.0f), CMPLXF(5.8000e-02f, 1.0f), CMPLXF(4.8450e-01f, 1.0f), CMPLXF(7.3610e-01f, 1.0f)
};
static const INT csx_islct_7[2] = {2, 3};

static const c64 csx_A_8[16] = {
    CMPLXF(2.1130e-01f, 9.9330e-01f), CMPLXF(8.0960e-01f, 4.2370e-01f), CMPLXF(4.8320e-01f, 1.1670e-01f), CMPLXF(6.5380e-01f, 4.9430e-01f),
    CMPLXF(8.2400e-02f, 8.3600e-01f), CMPLXF(8.4740e-01f, 2.6130e-01f), CMPLXF(6.1350e-01f, 6.2500e-01f), CMPLXF(4.8990e-01f, 3.6500e-02f),
    CMPLXF(7.5990e-01f, 7.4690e-01f), CMPLXF(4.5240e-01f, 2.4030e-01f), CMPLXF(2.7490e-01f, 5.5100e-01f), CMPLXF(7.7410e-01f, 2.2600e-01f),
    CMPLXF(8.7000e-03f, 3.7800e-02f), CMPLXF(8.0750e-01f, 3.4050e-01f), CMPLXF(8.8070e-01f, 3.5500e-01f), CMPLXF(9.6260e-01f, 8.1590e-01f)
};
static const INT csx_islct_8[2] = {1, 2};

static const c64 csx_A_9[9] = {
    CMPLXF( 1.0f,  2.0f), CMPLXF( 3.0f,  4.0f), CMPLXF(21.0f, 22.0f),
    CMPLXF(43.0f, 44.0f), CMPLXF(13.0f, 14.0f), CMPLXF(15.0f, 16.0f),
    CMPLXF( 5.0f,  6.0f), CMPLXF( 7.0f,  8.0f), CMPLXF(25.0f, 26.0f)
};
static const INT csx_islct_9[2] = {1, 2};

static const c64 csx_A_10[16] = {
    CMPLXF(5.0f, 9.0f), CMPLXF(5.0f,  5.0f), CMPLXF(-6.0f, -6.0f), CMPLXF(-7.0f, -7.0f),
    CMPLXF(3.0f, 3.0f), CMPLXF(6.0f, 10.0f), CMPLXF(-5.0f, -5.0f), CMPLXF(-6.0f, -6.0f),
    CMPLXF(2.0f, 2.0f), CMPLXF(3.0f,  3.0f), CMPLXF(-1.0f,  3.0f), CMPLXF(-5.0f, -5.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f,  2.0f), CMPLXF(-3.0f, -3.0f), CMPLXF( 0.0f,  4.0f)
};
static const INT csx_islct_10[2] = {0, 2};

static const c64 csx_A_11[16] = {
    3.0f, 1.0f, 0.0f, CMPLXF(0.0f, 2.0f),
    1.0f, 3.0f, CMPLXF(0.0f, -2.0f), 0.0f,
    0.0f, CMPLXF(0.0f, 2.0f), 1.0f, 1.0f,
    CMPLXF(0.0f, -2.0f), 0.0f, 1.0f, 1.0f
};
static const INT csx_islct_11[3] = {0, 2, 3};

static const c64 csx_A_12[16] = {
    7.0f, 3.0f, CMPLXF(1.0f, 2.0f), CMPLXF(-1.0f, 2.0f),
    3.0f, 7.0f, CMPLXF(1.0f, -2.0f), CMPLXF(-1.0f, -2.0f),
    CMPLXF(1.0f, -2.0f), CMPLXF(1.0f, 2.0f), 7.0f, -3.0f,
    CMPLXF(-1.0f, -2.0f), CMPLXF(-2.0f, 2.0f), -3.0f, 7.0f
};
static const INT csx_islct_12[2] = {1, 2};

static const c64 csx_A_13[25] = {
    CMPLXF(1.0f, 2.0f), CMPLXF(3.0f, 4.0f), CMPLXF(21.0f, 22.0f), CMPLXF(23.0f, 24.0f), CMPLXF(41.0f, 42.0f),
    CMPLXF(43.0f, 44.0f), CMPLXF(13.0f, 14.0f), CMPLXF(15.0f, 16.0f), CMPLXF(33.0f, 34.0f), CMPLXF(35.0f, 36.0f),
    CMPLXF(5.0f, 6.0f), CMPLXF(7.0f, 8.0f), CMPLXF(25.0f, 26.0f), CMPLXF(27.0f, 28.0f), CMPLXF(45.0f, 46.0f),
    CMPLXF(47.0f, 48.0f), CMPLXF(17.0f, 18.0f), CMPLXF(19.0f, 20.0f), CMPLXF(37.0f, 38.0f), CMPLXF(39.0f, 40.0f),
    CMPLXF(9.0f, 10.0f), CMPLXF(11.0f, 12.0f), CMPLXF(29.0f, 30.0f), CMPLXF(31.0f, 32.0f), CMPLXF(49.0f, 50.0f)
};
static const INT csx_islct_13[2] = {1, 2};

static const c64 csx_A_14[9] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(2.0f, 2.0f),
                0.0f, CMPLXF( 0.0f,  1.0f),             2.0f,
                0.0f,              -1.0f, CMPLXF(3.0f, 1.0f)
};
static const INT csx_islct_14[2] = {0, 1};

static const c64 csx_A_15[16] = {
    CMPLXF(-4.0f, -2.0f), CMPLXF(-5.0f, -6.0f), CMPLXF(-2.0f, -6.0f), CMPLXF(0.0f, -2.0f),
                  1.0f,               0.0f,               0.0f,              0.0f,
                  0.0f,               1.0f,               0.0f,              0.0f,
                  0.0f,               0.0f,               1.0f,              0.0f
};
static const INT csx_islct_15[2] = {0, 2};

static const c64 csx_A_16[49] = {
    CMPLXF(2.0f, 4.0f), CMPLXF(1.0f, 1.0f), CMPLXF(6.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(5.0f, 5.0f), CMPLXF(2.0f, 6.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 2.0f), CMPLXF(1.0f, 3.0f), CMPLXF(3.0f, 1.0f), CMPLXF(5.0f, -4.0f), CMPLXF(1.0f, 1.0f), CMPLXF(7.0f, 2.0f), CMPLXF(2.0f, 3.0f),
                0.0f, CMPLXF(3.0f, -2.0f), CMPLXF(1.0f, 1.0f), CMPLXF(6.0f, 3.0f), CMPLXF(2.0f, 1.0f), CMPLXF(1.0f, 4.0f), CMPLXF(2.0f, 1.0f),
                0.0f,              0.0f, CMPLXF(2.0f, 3.0f), CMPLXF(3.0f, 1.0f), CMPLXF(1.0f, 2.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 1.0f),
                0.0f,              0.0f,             0.0f, CMPLXF(2.0f, -1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 1.0f), CMPLXF(1.0f, 3.0f),
                0.0f,              0.0f,             0.0f,              0.0f, CMPLXF(1.0f, -1.0f), CMPLXF(2.0f, 1.0f), CMPLXF(2.0f, 2.0f),
                0.0f,              0.0f,             0.0f,              0.0f,              0.0f, CMPLXF(2.0f, -2.0f), CMPLXF(1.0f, 1.0f)
};
static const INT csx_islct_16[4] = {0, 3, 5, 6};

static const c64 csx_A_17[25] = {
    CMPLXF(0.0f, 5.0f), CMPLXF(1.0f, 2.0f), CMPLXF(2.0f, 3.0f), CMPLXF(-3.0f, 6.0f), 6.0f,
    CMPLXF(-1.0f, 2.0f), CMPLXF(0.0f, 6.0f), CMPLXF(4.0f, 5.0f), CMPLXF(-3.0f, -2.0f), 5.0f,
    CMPLXF(-2.0f, 3.0f), CMPLXF(-4.0f, 5.0f), CMPLXF(0.0f, 7.0f), 3.0f, 2.0f,
    CMPLXF(3.0f, 6.0f), CMPLXF(3.0f, -2.0f), -3.0f, CMPLXF(0.0f, -5.0f), CMPLXF(2.0f, 1.0f),
    -6.0f, -5.0f, -2.0f, CMPLXF(-2.0f, 1.0f), CMPLXF(0.0f, 2.0f)
};
static const INT csx_islct_17[3] = {0, 2, 4};

static const c64 csx_A_18[64] = {
    CMPLXF(0.0f, 1.0f), 1.0f, 0.0f, 0.0f, CMPLXF(0.0f, 1.0f), 1.0f, CMPLXF(0.0f, 1.0f), 1.0f,
    0.0f, CMPLXF(0.0f, 1.0f), 1.0f, 0.0f, CMPLXF(0.0f, 2.0f), 2.0f, CMPLXF(0.0f, 2.0f), 2.0f,
    0.0f, 0.0f, CMPLXF(0.0f, 1.0f), 1.0f, CMPLXF(0.0f, 3.0f), 3.0f, CMPLXF(0.0f, 3.0f), 3.0f,
    0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 4.0f), 4.0f, CMPLXF(0.0f, 4.0f), 4.0f,
    0.0f, 0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 9.5000e-01f), 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 9.5000e-01f), 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 9.5000e-01f), 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, CMPLXF(0.0f, 9.5000e-01f)
};
static const INT csx_islct_18[4] = {0, 1, 2, 3};

static const c64 csx_A_19[9] = {
    2.0f, CMPLXF(0.0f, -1.0f), 0.0f,
    CMPLXF(0.0f, 1.0f), 2.0f, 0.0f,
    0.0f, 0.0f, 3.0f
};
static const INT csx_islct_19[2] = {1, 2};

typedef struct {
    int n;
    int isrt;
    int nslct;
    const INT* islct;
    const c64* A;
    f32 rcdein;
    f32 rcdvin;
} zsx_precomputed_t;

#define ZSX_NUM_PRECOMPUTED 20

static const zsx_precomputed_t ZSX_PRECOMPUTED[ZSX_NUM_PRECOMPUTED] = {
    {1, 0, 1, csx_islct_0, csx_A_0, 1.0000e+00f, 0.0000e+00f},
    {1, 0, 1, csx_islct_1, csx_A_1, 1.0000e+00f, 1.0000e+00f},
    {5, 0, 3, csx_islct_2, csx_A_2, 1.0000e+00f, 2.9582e-31f},
    {5, 0, 3, csx_islct_3, csx_A_3, 1.0000e+00f, 1.0000e+00f},
    {5, 0, 2, csx_islct_4, csx_A_4, 1.0000e+00f, 1.0000e+00f},
    {6, 1, 3, csx_islct_5, csx_A_5, 1.0000e+00f, 2.0000e+00f},
    {6, 0, 3, csx_islct_6, csx_A_6, 1.0000e+00f, 2.0000e+00f},
    {4, 0, 2, csx_islct_7, csx_A_7, 9.6350e-01f, 3.3122e-01f},
    {4, 0, 2, csx_islct_8, csx_A_8, 8.4053e-01f, 7.4754e-01f},
    {3, 0, 2, csx_islct_9, csx_A_9, 3.9550e-01f, 2.0464e+01f},
    {4, 0, 2, csx_islct_10, csx_A_10, 3.3333e-01f, 1.2569e-01f},
    {4, 0, 3, csx_islct_11, csx_A_11, 1.0000e+00f, 8.2843e-01f},
    {4, 0, 2, csx_islct_12, csx_A_12, 9.8985e-01f, 4.1447e+00f},
    {5, 1, 2, csx_islct_13, csx_A_13, 3.1088e-01f, 4.6912e+00f},
    {3, 0, 2, csx_islct_14, csx_A_14, 2.2361e-01f, 1.0000e+00f},
    {4, 1, 2, csx_islct_15, csx_A_15, 7.2803e-05f, 1.1947e-04f},
    {7, 0, 4, csx_islct_16, csx_A_16, 3.7241e-01f, 5.2080e-01f},
    {5, 1, 3, csx_islct_17, csx_A_17, 1.0000e+00f, 4.5989e+00f},
    {8, 1, 4, csx_islct_18, csx_A_18, 9.5269e-12f, 2.9360e-11f},
    {3, 0, 2, csx_islct_19, csx_A_19, 1.0000e+00f, 2.0000e+00f}
};

#endif /* CSX_TESTDATA_H */
