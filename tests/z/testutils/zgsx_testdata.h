/**
 * @file zgsx_testdata.h
 * @brief Precomputed test matrix pairs for ZGGESX (from LAPACK TESTING/zgd.in)
 *
 * 2 matrix pairs with precomputed condition numbers for eigenvalue cluster
 * and deflating subspace. Used by test_zdrgsx.c for read-in tests.
 */

#ifndef ZGSX_TESTDATA_H
#define ZGSX_TESTDATA_H

typedef struct {
    INT mplusn;
    INT n;
    const c128* A;
    const c128* B;
    f64 pltru;
    f64 diftru;
} zgsx_precomputed_t;

/* Test case 0: MPLUSN=4, N=2
 * A (row-major in zgd.in, transposed to column-major here):
 *   Row 0: (2,6)  (2,5)    (3,-10)   (4,7)
 *   Row 1: (0,0)  (9,2)    (16,-24)  (7,-7)
 *   Row 2: (0,0)  (0,0)    (8,-3)    (9,-8)
 *   Row 3: (0,0)  (0,0)    (0,0)     (10,-16)
 */
static const c128 zgsx_A_0[16] = {
    CMPLX(2.0, 6.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(2.0, 5.0), CMPLX(9.0, 2.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(3.0, -10.0), CMPLX(16.0, -24.0), CMPLX(8.0, -3.0), CMPLX(0.0, 0.0),
    CMPLX(4.0, 7.0), CMPLX(7.0, -7.0), CMPLX(9.0, -8.0), CMPLX(10.0, -16.0)
};

/* B (row-major in zgd.in, transposed to column-major here):
 *   Row 0: (-9,1)  (-1,-8)  (-1,10)  (2,-6)
 *   Row 1: (0,0)   (-1,4)   (1,16)   (-6,4)
 *   Row 2: (0,0)   (0,0)    (1,-14)  (-1,6)
 *   Row 3: (0,0)   (0,0)    (0,0)    (8,4)
 */
static const c128 zgsx_B_0[16] = {
    CMPLX(-9.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, -8.0), CMPLX(-1.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, 10.0), CMPLX(1.0, 16.0), CMPLX(1.0, -14.0), CMPLX(0.0, 0.0),
    CMPLX(2.0, -6.0), CMPLX(-6.0, 4.0), CMPLX(-1.0, 6.0), CMPLX(8.0, 4.0)
};

/* Test case 1: MPLUSN=4, N=2
 * A (row-major in zgd.in, transposed to column-major here):
 *   Row 0: (1,8)  (2,4)    (3,-13)   (4,4)
 *   Row 1: (0,0)  (5,7)    (6,-24)   (7,-3)
 *   Row 2: (0,0)  (0,0)    (8,3)     (9,-5)
 *   Row 3: (0,0)  (0,0)    (0,0)     (10,16)
 */
static const c128 zgsx_A_1[16] = {
    CMPLX(1.0, 8.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(2.0, 4.0), CMPLX(5.0, 7.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(3.0, -13.0), CMPLX(6.0, -24.0), CMPLX(8.0, 3.0), CMPLX(0.0, 0.0),
    CMPLX(4.0, 4.0), CMPLX(7.0, -3.0), CMPLX(9.0, -5.0), CMPLX(10.0, 16.0)
};

/* B (row-major in zgd.in, transposed to column-major here):
 *   Row 0: (-1,9)  (-1,-1)  (-1,1)   (-1,-6)
 *   Row 1: (0,0)   (-1,4)   (-1,16)  (-1,-24)
 *   Row 2: (0,0)   (0,0)    (1,-11)  (-1,6)
 *   Row 3: (0,0)   (0,0)    (0,0)    (1,4)
 */
static const c128 zgsx_B_1[16] = {
    CMPLX(-1.0, 9.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, -1.0), CMPLX(-1.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, 1.0), CMPLX(-1.0, 16.0), CMPLX(1.0, -11.0), CMPLX(0.0, 0.0),
    CMPLX(-1.0, -6.0), CMPLX(-1.0, -24.0), CMPLX(-1.0, 6.0), CMPLX(1.0, 4.0)
};

#define ZGSX_NUM_PRECOMPUTED 2

static const zgsx_precomputed_t ZGSX_PRECOMPUTED[ZGSX_NUM_PRECOMPUTED] = {
    {4, 2, zgsx_A_0, zgsx_B_0, 7.6883e-02, 2.1007e-01},
    {4, 2, zgsx_A_1, zgsx_B_1, 4.2067e-01, 4.9338e+00}
};

#endif /* ZGSX_TESTDATA_H */
