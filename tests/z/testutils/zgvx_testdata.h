/**
 * @file zgvx_testdata.h
 * @brief Precomputed test matrix pairs for ZGGEVX (from LAPACK TESTING/zgd.in)
 *
 * 2 matrix pairs with precomputed eigenvalue/eigenvector condition numbers.
 * Used by test_zdrgvx.c for read-in tests (condition number accuracy).
 */

#ifndef ZGVX_TESTDATA_H
#define ZGVX_TESTDATA_H

typedef struct {
    INT n;
    const c128* A;
    const c128* B;
    const f64* dtru;
    const f64* diftru;
} zgvx_precomputed_t;

/* Test case 0: N=4
 * A (column-major, transposed from row-major zgd.in):
 *   Row 0: (2,6)  (2,5)    (3,-10)   (4,7)
 *   Row 1: (0,0)  (9,2)    (16,-24)  (7,-7)
 *   Row 2: (0,0)  (0,0)    (8,-3)    (9,-8)
 *   Row 3: (0,0)  (0,0)    (0,0)     (10,-16)
 */
static const c128 zgvx_A_0[16] = {
    CMPLX(2.0, 6.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(2.0, 5.0), CMPLX(9.0, 2.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(3.0, -10.0), CMPLX(16.0, -24.0), CMPLX(8.0, -3.0), CMPLX(0.0, 0.0),
    CMPLX(4.0, 7.0), CMPLX(7.0, -7.0), CMPLX(9.0, -8.0), CMPLX(10.0, -16.0)
};

/* B (column-major):
 *   Row 0: (-9,1)  (-1,-8)  (-1,10)  (2,-6)
 *   Row 1: (0,0)   (-1,4)   (1,16)   (-6,4)
 *   Row 2: (0,0)   (0,0)    (1,-14)  (-1,6)
 *   Row 3: (0,0)   (0,0)    (0,0)    (8,4)
 */
static const c128 zgvx_B_0[16] = {
    CMPLX(-9.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, -8.0), CMPLX(-1.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, 10.0), CMPLX(1.0, 16.0), CMPLX(1.0, -14.0), CMPLX(0.0, 0.0),
    CMPLX(2.0, -6.0), CMPLX(-6.0, 4.0), CMPLX(-1.0, 6.0), CMPLX(8.0, 4.0)
};

static const f64 zgvx_dtru_0[4] = {5.2612e+00, 8.0058e-01, 1.4032e+00, 4.0073e+00};
static const f64 zgvx_diftru_0[4] = {1.1787e+00, 3.3139e+00, 1.1835e+00, 2.0777e+00};

/* Test case 1: N=4
 * A (column-major):
 *   Row 0: (1,8)  (2,4)    (3,-13)   (4,4)
 *   Row 1: (0,0)  (5,7)    (6,-24)   (7,-3)
 *   Row 2: (0,0)  (0,0)    (8,3)     (9,-5)
 *   Row 3: (0,0)  (0,0)    (0,0)     (10,16)
 */
static const c128 zgvx_A_1[16] = {
    CMPLX(1.0, 8.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(2.0, 4.0), CMPLX(5.0, 7.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(3.0, -13.0), CMPLX(6.0, -24.0), CMPLX(8.0, 3.0), CMPLX(0.0, 0.0),
    CMPLX(4.0, 4.0), CMPLX(7.0, -3.0), CMPLX(9.0, -5.0), CMPLX(10.0, 16.0)
};

/* B (column-major):
 *   Row 0: (-1,9)  (-1,-1)  (-1,1)   (-1,-6)
 *   Row 1: (0,0)   (-1,4)   (-1,16)  (-1,-24)
 *   Row 2: (0,0)   (0,0)    (1,-11)  (-1,6)
 *   Row 3: (0,0)   (0,0)    (0,0)    (1,4)
 */
static const c128 zgvx_B_1[16] = {
    CMPLX(-1.0, 9.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, -1.0), CMPLX(-1.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, 1.0), CMPLX(-1.0, 16.0), CMPLX(1.0, -11.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, -6.0), CMPLX(-1.0, -24.0), CMPLX(-1.0, 6.0), CMPLX(1.0, 4.0)
};

static const f64 zgvx_dtru_1[4] = {4.9068e+00, 1.6813e+00, 3.4636e+00, 5.2436e+00};
static const f64 zgvx_diftru_1[4] = {1.0386e+00, 1.4728e+00, 2.0029e+00, 9.8365e-01};

#define ZGVX_NUM_PRECOMPUTED 2

static const zgvx_precomputed_t ZGVX_PRECOMPUTED[ZGVX_NUM_PRECOMPUTED] = {
    {4, zgvx_A_0, zgvx_B_0, zgvx_dtru_0, zgvx_diftru_0},
    {4, zgvx_A_1, zgvx_B_1, zgvx_dtru_1, zgvx_diftru_1}
};

#endif /* ZGVX_TESTDATA_H */
