/**
 * @file cgvx_testdata.h
 * @brief Precomputed test matrix pairs for CGGEVX (from LAPACK TESTING/zgd.in)
 *
 * 2 matrix pairs with precomputed eigenvalue/eigenvector condition numbers.
 * Used by test_zdrgvx.c for read-in tests (condition number accuracy).
 */

#ifndef CGVX_TESTDATA_H
#define CGVX_TESTDATA_H

typedef struct {
    INT n;
    const c64* A;
    const c64* B;
    const f32* dtru;
    const f32* diftru;
} zgvx_precomputed_t;

/* Test case 0: N=4
 * A (column-major, transposed from row-major zgd.in):
 *   Row 0: (2,6)  (2,5)    (3,-10)   (4,7)
 *   Row 1: (0,0)  (9,2)    (16,-24)  (7,-7)
 *   Row 2: (0,0)  (0,0)    (8,-3)    (9,-8)
 *   Row 3: (0,0)  (0,0)    (0,0)     (10,-16)
 */
static const c64 cgvx_A_0[16] = {
    CMPLXF(2.0f, 6.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(2.0f, 5.0f), CMPLXF(9.0f, 2.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(3.0f, -10.0f), CMPLXF(16.0f, -24.0f), CMPLXF(8.0f, -3.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(4.0f, 7.0f), CMPLXF(7.0f, -7.0f), CMPLXF(9.0f, -8.0f), CMPLXF(10.0f, -16.0f)
};

/* B (column-major):
 *   Row 0: (-9,1)  (-1,-8)  (-1,10)  (2,-6)
 *   Row 1: (0,0)   (-1,4)   (1,16)   (-6,4)
 *   Row 2: (0,0)   (0,0)    (1,-14)  (-1,6)
 *   Row 3: (0,0)   (0,0)    (0,0)    (8,4)
 */
static const c64 cgvx_B_0[16] = {
    CMPLXF(-9.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, -8.0f), CMPLXF(-1.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, 10.0f), CMPLXF(1.0f, 16.0f), CMPLXF(1.0f, -14.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(2.0f, -6.0f), CMPLXF(-6.0f, 4.0f), CMPLXF(-1.0f, 6.0f), CMPLXF(8.0f, 4.0f)
};

static const f32 cgvx_dtru_0[4] = {5.2612e+00f, 8.0058e-01f, 1.4032e+00f, 4.0073e+00f};
static const f32 cgvx_diftru_0[4] = {1.1787e+00f, 3.3139e+00f, 1.1835e+00f, 2.0777e+00f};

/* Test case 1: N=4
 * A (column-major):
 *   Row 0: (1,8)  (2,4)    (3,-13)   (4,4)
 *   Row 1: (0,0)  (5,7)    (6,-24)   (7,-3)
 *   Row 2: (0,0)  (0,0)    (8,3)     (9,-5)
 *   Row 3: (0,0)  (0,0)    (0,0)     (10,16)
 */
static const c64 cgvx_A_1[16] = {
    CMPLXF(1.0f, 8.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(2.0f, 4.0f), CMPLXF(5.0f, 7.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(3.0f, -13.0f), CMPLXF(6.0f, -24.0f), CMPLXF(8.0f, 3.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(4.0f, 4.0f), CMPLXF(7.0f, -3.0f), CMPLXF(9.0f, -5.0f), CMPLXF(10.0f, 16.0f)
};

/* B (column-major):
 *   Row 0: (-1,9)  (-1,-1)  (-1,1)   (-1,-6)
 *   Row 1: (0,0)   (-1,4)   (-1,16)  (-1,-24)
 *   Row 2: (0,0)   (0,0)    (1,-11)  (-1,6)
 *   Row 3: (0,0)   (0,0)    (0,0)    (1,4)
 */
static const c64 cgvx_B_1[16] = {
    CMPLXF(-1.0f, 9.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, 1.0f), CMPLXF(-1.0f, 16.0f), CMPLXF(1.0f, -11.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, -6.0f), CMPLXF(-1.0f, -24.0f), CMPLXF(-1.0f, 6.0f), CMPLXF(1.0f, 4.0f)
};

static const f32 cgvx_dtru_1[4] = {4.9068e+00f, 1.6813e+00f, 3.4636e+00f, 5.2436e+00f};
static const f32 cgvx_diftru_1[4] = {1.0386e+00f, 1.4728e+00f, 2.0029e+00f, 9.8365e-01f};

#define ZGVX_NUM_PRECOMPUTED 2

static const zgvx_precomputed_t ZGVX_PRECOMPUTED[ZGVX_NUM_PRECOMPUTED] = {
    {4, cgvx_A_0, cgvx_B_0, cgvx_dtru_0, cgvx_diftru_0},
    {4, cgvx_A_1, cgvx_B_1, cgvx_dtru_1, cgvx_diftru_1}
};

#endif /* CGVX_TESTDATA_H */
