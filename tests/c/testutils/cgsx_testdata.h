/**
 * @file cgsx_testdata.h
 * @brief Precomputed test matrix pairs for CGGESX (from LAPACK TESTING/zgd.in)
 *
 * 2 matrix pairs with precomputed condition numbers for eigenvalue cluster
 * and deflating subspace. Used by test_zdrgsx.c for read-in tests.
 */

#ifndef CGSX_TESTDATA_H
#define CGSX_TESTDATA_H

typedef struct {
    INT mplusn;
    INT n;
    const c64* A;
    const c64* B;
    f32 pltru;
    f32 diftru;
} zgsx_precomputed_t;

/* Test case 0: MPLUSN=4, N=2
 * A (row-major in zgd.in, transposed to column-major here):
 *   Row 0: (2,6)  (2,5)    (3,-10)   (4,7)
 *   Row 1: (0,0)  (9,2)    (16,-24)  (7,-7)
 *   Row 2: (0,0)  (0,0)    (8,-3)    (9,-8)
 *   Row 3: (0,0)  (0,0)    (0,0)     (10,-16)
 */
static const c64 cgsx_A_0[16] = {
    CMPLXF(2.0f, 6.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(2.0f, 5.0f), CMPLXF(9.0f, 2.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(3.0f, -10.0f), CMPLXF(16.0f, -24.0f), CMPLXF(8.0f, -3.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(4.0f, 7.0f), CMPLXF(7.0f, -7.0f), CMPLXF(9.0f, -8.0f), CMPLXF(10.0f, -16.0f)
};

/* B (row-major in zgd.in, transposed to column-major here):
 *   Row 0: (-9,1)  (-1,-8)  (-1,10)  (2,-6)
 *   Row 1: (0,0)   (-1,4)   (1,16)   (-6,4)
 *   Row 2: (0,0)   (0,0)    (1,-14)  (-1,6)
 *   Row 3: (0,0)   (0,0)    (0,0)    (8,4)
 */
static const c64 cgsx_B_0[16] = {
    CMPLXF(-9.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, -8.0f), CMPLXF(-1.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, 10.0f), CMPLXF(1.0f, 16.0f), CMPLXF(1.0f, -14.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(2.0f, -6.0f), CMPLXF(-6.0f, 4.0f), CMPLXF(-1.0f, 6.0f), CMPLXF(8.0f, 4.0f)
};

/* Test case 1: MPLUSN=4, N=2
 * A (row-major in zgd.in, transposed to column-major here):
 *   Row 0: (1,8)  (2,4)    (3,-13)   (4,4)
 *   Row 1: (0,0)  (5,7)    (6,-24)   (7,-3)
 *   Row 2: (0,0)  (0,0)    (8,3)     (9,-5)
 *   Row 3: (0,0)  (0,0)    (0,0)     (10,16)
 */
static const c64 cgsx_A_1[16] = {
    CMPLXF(1.0f, 8.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(2.0f, 4.0f), CMPLXF(5.0f, 7.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(3.0f, -13.0f), CMPLXF(6.0f, -24.0f), CMPLXF(8.0f, 3.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(4.0f, 4.0f), CMPLXF(7.0f, -3.0f), CMPLXF(9.0f, -5.0f), CMPLXF(10.0f, 16.0f)
};

/* B (row-major in zgd.in, transposed to column-major here):
 *   Row 0: (-1,9)  (-1,-1)  (-1,1)   (-1,-6)
 *   Row 1: (0,0)   (-1,4)   (-1,16)  (-1,-24)
 *   Row 2: (0,0)   (0,0)    (1,-11)  (-1,6)
 *   Row 3: (0,0)   (0,0)    (0,0)    (1,4)
 */
static const c64 cgsx_B_1[16] = {
    CMPLXF(-1.0f, 9.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, 1.0f), CMPLXF(-1.0f, 16.0f), CMPLXF(1.0f, -11.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(-1.0f, -6.0f), CMPLXF(-1.0f, -24.0f), CMPLXF(-1.0f, 6.0f), CMPLXF(1.0f, 4.0f)
};

#define ZGSX_NUM_PRECOMPUTED 2

static const zgsx_precomputed_t ZGSX_PRECOMPUTED[ZGSX_NUM_PRECOMPUTED] = {
    {4, 2, cgsx_A_0, cgsx_B_0, 7.6883e-02f, 2.1007e-01f},
    {4, 2, cgsx_A_1, cgsx_B_1, 4.2067e-01f, 4.9338e+00f}
};

#endif /* CGSX_TESTDATA_H */
